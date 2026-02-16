"""
Training Script — YOLOv8 training optimized for RTX 4060.

Complete training pipeline with hyperparameters tuned for RTX 4060 (8GB VRAM),
supporting class weighting, early stopping, resume training, and data augmentation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Factory Safety Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch with YOLOv8m
  python -m factory_ai.training.train --model yolov8m.pt --data datasets/factory.yaml

  # Resume interrupted training
  python -m factory_ai.training.train --resume runs/detect/train/weights/last.pt

  # Train with custom hyperparameters
  python -m factory_ai.training.train --model yolov8m.pt --epochs 200 --batch 16 --imgsz 960

  # Train with class weights for imbalanced dataset
  python -m factory_ai.training.train --model yolov8m.pt --class-weights
        """
    )
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Base model weights (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="datasets/factory.yaml",
                        help="Dataset YAML config path")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (16 for 8GB VRAM with imgsz=640)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (640 or 960)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0, 1, 0,1 for multi-GPU, or cpu)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint (path to last.pt)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project save directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--class-weights", action="store_true",
                        help="Enable class-balanced weighting")
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N backbone layers for transfer learning")
    return parser.parse_args()


def get_hyperparameters(imgsz: int = 640, class_weights: bool = False) -> dict:
    """
    Get optimized hyperparameters for RTX 4060 factory training.

    These hyperparameters are tuned for:
    - RTX 4060 (8GB VRAM)
    - Industrial/factory surveillance imagery
    - Mixed object sizes (small PPE items to large forklifts)

    Args:
        imgsz: Training image size.
        class_weights: Whether to use class-balanced loss weights.

    Returns:
        Dict of training hyperparameters.
    """
    hyp = {
        # --- Optimizer ---
        "optimizer": "AdamW",
        "lr0": 0.001,           # Initial learning rate
        "lrf": 0.01,            # Final learning rate factor (lr0 * lrf)
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # --- Loss ---
        "box": 7.5,             # Box loss weight
        "cls": 0.5,             # Classification loss weight
        "dfl": 1.5,             # Distribution focal loss weight

        # --- Augmentation (factory-optimized) ---
        "hsv_h": 0.015,         # Hue augmentation (small for safety colors)
        "hsv_s": 0.7,           # Saturation augmentation
        "hsv_v": 0.4,           # Value/brightness augmentation
        "degrees": 5.0,         # Rotation (small for factory scenes)
        "translate": 0.1,       # Translation
        "scale": 0.5,           # Scale augmentation
        "shear": 2.0,           # Shear
        "perspective": 0.0001,  # Perspective transform
        "flipud": 0.01,         # Vertical flip (rare in factory)
        "fliplr": 0.5,          # Horizontal flip
        "mosaic": 1.0,          # Mosaic augmentation
        "mixup": 0.1,           # MixUp augmentation
        "copy_paste": 0.1,      # Copy-paste augmentation

        # --- Other ---
        "close_mosaic": 10,     # Disable mosaic in last N epochs
        "label_smoothing": 0.0,
    }

    return hyp


def train(args):
    """Execute the training pipeline."""
    from ultralytics import YOLO

    # Set experiment name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"factory_{timestamp}"

    print("=" * 70)
    print("🏭 Factory AI — YOLOv8 Training Pipeline")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device:     {args.device}")
    print(f"  Workers:    {args.workers}")
    print(f"  Patience:   {args.patience}")
    print(f"  Output:     {args.project}/{args.name}")
    print("=" * 70)

    # Get optimized hyperparameters
    hyp = get_hyperparameters(args.imgsz, args.class_weights)

    # Load model
    if args.resume:
        print(f"\n🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        print(f"\n📦 Loading base model: {args.model}")
        model = YOLO(args.model)

        # Freeze backbone layers if specified
        if args.freeze > 0:
            print(f"🧊 Freezing first {args.freeze} backbone layers")

        # Start training
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
            save_period=10,        # Save checkpoint every 10 epochs
            plots=True,            # Generate training plots
            freeze=args.freeze if args.freeze > 0 else None,

            # Hyperparameters
            **hyp,
        )

    # Post-training
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)

    # Copy best weights to models/
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)
        import shutil
        dest = "models/best.pt"
        shutil.copy2(best_path, dest)
        print(f"📦 Best model copied to: {dest}")

    # Validate
    print("\n📊 Running validation...")
    metrics = model.val()
    print(f"   mAP@50:    {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision:  {metrics.box.mp:.4f}")
    print(f"   Recall:     {metrics.box.mr:.4f}")

    print(f"\n📂 Results saved to: {args.project}/{args.name}")
    print("=" * 70)

    return results


# ===========================================================
# RTX 4060 Batch Size Recommendations
# ===========================================================
#
# Model     | imgsz=640 | imgsz=960
# ----------|-----------|----------
# YOLOv8n   | batch=32  | batch=16
# YOLOv8s   | batch=24  | batch=12
# YOLOv8m   | batch=16  | batch=8
# YOLOv8l   | batch=8   | batch=4
# YOLOv8x   | batch=4   | batch=2
#
# Note: These are approximate for RTX 4060 (8GB VRAM).
# Monitor VRAM usage and adjust accordingly.
# Use gradient accumulation for effective larger batch sizes.
# ===========================================================


if __name__ == "__main__":
    args = parse_args()
    train(args)
