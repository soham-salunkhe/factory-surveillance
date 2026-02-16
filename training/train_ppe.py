"""
PPE Training Script — YOLOv8 fine-tuning for Helmet & Mask detection.

Optimized for Mac M4 (MPS backend). Fine-tunes YOLOv8m pretrained weights
for 4-class PPE detection: helmet, no_helmet, mask, no_mask.

The original pretrained model is NOT modified — we train a new model head
using transfer learning.

Usage:
    # Full training (100 epochs)
    python training/train_ppe.py

    # Quick smoke test
    python training/train_ppe.py --epochs 2 --batch 4

    # Resume interrupted training
    python training/train_ppe.py --resume runs/detect/ppe_latest/weights/last.pt

    # Custom settings
    python training/train_ppe.py --model yolov8m.pt --epochs 150 --batch 16 --imgsz 640
"""

import os
import sys
import argparse
import platform
import shutil
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command-line arguments for PPE training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Helmet & Mask PPE Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 100-epoch training on Mac M4
  python training/train_ppe.py

  # Quick test (2 epochs)
  python training/train_ppe.py --epochs 2 --batch 4

  # Resume training
  python training/train_ppe.py --resume runs/detect/ppe_latest/weights/last.pt

  # Use larger model
  python training/train_ppe.py --model yolov8l.pt --batch 4
        """
    )
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Base model weights (default: yolov8m.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="Dataset YAML config (default: auto-detect)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8 for Mac M4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'mps' for Mac, '0' for CUDA, 'cpu' (auto-detected)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers (default: 4 for Mac)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (path to last.pt)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project save directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze first N backbone layers (default: 10 for transfer learning)")
    return parser.parse_args()


def detect_device() -> str:
    """Auto-detect the best available device."""
    import torch

    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected — using MPS backend")
        return "mps"
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 NVIDIA GPU detected — using CUDA ({gpu_name})")
        return "0"
    else:
        print("💻 No GPU found — using CPU (training will be slow!)")
        return "cpu"


def detect_dataset_yaml(project_root: Path) -> str:
    """Auto-detect dataset YAML file."""
    # Check for auto-generated data.yaml from download script
    auto_yaml = project_root / "datasets" / "ppe_helmet_mask" / "data.yaml"
    if auto_yaml.exists():
        return str(auto_yaml)

    # Fall back to the manual YAML
    manual_yaml = project_root / "datasets" / "ppe_helmet_mask.yaml"
    if manual_yaml.exists():
        return str(manual_yaml)

    print("❌ No dataset YAML found!")
    print("   Run 'python training/download_dataset.py' first to download the dataset.")
    sys.exit(1)


def get_ppe_hyperparameters() -> dict:
    """
    Get hyperparameters optimized for PPE (helmet/mask) detection.

    Tuned for:
    - Small-to-medium objects (helmets, masks on heads)
    - Industrial/construction environments
    - Transfer learning from COCO pretrained weights
    """
    return {
        # ── Optimizer ──
        "optimizer": "AdamW",
        "lr0": 0.001,            # Initial learning rate
        "lrf": 0.01,             # Final LR = lr0 * lrf
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5.0,    # Longer warmup for transfer learning
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # ── Loss ──
        "box": 7.5,              # Box loss weight
        "cls": 1.5,              # Higher cls weight — classification is key for PPE
        "dfl": 1.5,              # Distribution focal loss weight

        # ── Augmentation (PPE-optimized) ──
        "hsv_h": 0.01,           # Very small hue shift (preserve helmet/vest colors)
        "hsv_s": 0.5,            # Moderate saturation
        "hsv_v": 0.4,            # Brightness variation (indoor/outdoor)
        "degrees": 10.0,         # Rotation (heads tilt)
        "translate": 0.15,       # Translation
        "scale": 0.5,            # Scale (detect at various distances)
        "shear": 2.0,            # Shear
        "perspective": 0.0001,   # Slight perspective
        "flipud": 0.0,           # No vertical flip (helmets are always on top)
        "fliplr": 0.5,           # Horizontal flip OK
        "mosaic": 1.0,           # Mosaic augmentation
        "mixup": 0.15,           # MixUp
        "copy_paste": 0.1,       # Copy-paste augmentation

        # ── Other ──
        "close_mosaic": 15,      # Disable mosaic in last 15 epochs for fine detail
        "label_smoothing": 0.0,
    }


def train(args):
    """Execute the PPE training pipeline."""
    from ultralytics import YOLO

    project_root = Path(__file__).parent.parent

    # Auto-detect device
    if args.device is None:
        args.device = detect_device()
    else:
        print(f"📱 Using device: {args.device}")

    # Auto-detect dataset
    if args.data is None:
        args.data = detect_dataset_yaml(project_root)

    # Set experiment name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"ppe_{timestamp}"

    print()
    print("=" * 70)
    print("🏭 Factory AI — PPE Helmet & Mask Training")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Classes:     helmet, no_helmet, mask, no_mask")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch}")
    print(f"  Image Size:  {args.imgsz}")
    print(f"  Device:      {args.device}")
    print(f"  Workers:     {args.workers}")
    print(f"  Patience:    {args.patience}")
    print(f"  Freeze:      {args.freeze} backbone layers")
    print(f"  Output:      {args.project}/{args.name}")
    print("=" * 70)

    # Get PPE-optimized hyperparameters
    hyp = get_ppe_hyperparameters()

    # Load model
    if args.resume:
        print(f"\n🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        print(f"\n📦 Loading pretrained base model: {args.model}")
        print(f"   (Original COCO model is NOT modified — training new head for PPE)")
        model = YOLO(args.model)

        if args.freeze > 0:
            print(f"🧊 Will freeze first {args.freeze} backbone layers (transfer learning)")

        # ── Start training ──
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print(f"   Estimated time: ~{args.epochs * 3}–{args.epochs * 8} minutes on Mac M4\n")

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            save=True,
            save_period=10,         # Checkpoint every 10 epochs
            plots=True,             # Generate training plots
            freeze=args.freeze if args.freeze > 0 else None,

            # PPE-optimized hyperparameters
            **hyp,
        )

    # ── Post-training ──
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)

    # Copy best weights to models/
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        dest = str(models_dir / "best_ppe.pt")
        shutil.copy2(best_path, dest)
        print(f"📦 Best PPE model copied to: {dest}")

        # Also copy as latest for easy reference
        latest = str(models_dir / "latest_ppe.pt")
        shutil.copy2(best_path, latest)
    else:
        print(f"⚠️  Best weights not found at: {best_path}")

    # ── Validate ──
    print("\n📊 Running validation...")
    try:
        metrics = model.val()
        print(f"\n{'─' * 40}")
        print(f"  📈 Validation Results")
        print(f"{'─' * 40}")
        print(f"   mAP@50:      {metrics.box.map50:.4f}")
        print(f"   mAP@50-95:   {metrics.box.map:.4f}")
        print(f"   Precision:    {metrics.box.mp:.4f}")
        print(f"   Recall:       {metrics.box.mr:.4f}")
        print(f"{'─' * 40}")

        # Per-class metrics
        print(f"\n  📋 Per-Class Results:")
        class_names = ["helmet", "no_helmet", "mask", "no_mask"]
        if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) == len(class_names):
            for i, name in enumerate(class_names):
                print(f"   {name:12s}  AP@50: {metrics.box.ap50[i]:.4f}")
    except Exception as e:
        print(f"   ⚠️  Validation error: {e}")

    print(f"\n📂 Results saved to: {args.project}/{args.name}")
    print(f"   Weights:  {args.project}/{args.name}/weights/")
    print(f"   Plots:    {args.project}/{args.name}/")
    print("=" * 70)

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
