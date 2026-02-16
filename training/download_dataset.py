"""
Download & Merge Roboflow Datasets for Helmet/Mask PPE Detection.

Downloads datasets from Roboflow Universe, remaps class labels to a
unified 4-class schema, and merges them into a single dataset directory.

Classes:
    0: helmet
    1: no_helmet
    2: mask
    3: no_mask

Usage:
    python training/download_dataset.py
    python training/download_dataset.py --api-key YOUR_KEY
    python training/download_dataset.py --output datasets/ppe_helmet_mask
"""

import os
import sys
import glob
import shutil
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Unified class mapping ──
UNIFIED_CLASSES = {
    0: "helmet",
    1: "no_helmet",
    2: "mask",
    3: "no_mask",
}

# ── Dataset sources from Roboflow Universe ──
# Each entry: (workspace, project, version, class_remap_dict)
# class_remap_dict maps source class names → unified class index
DATASET_SOURCES = [
    {
        "workspace": "roboflow-universe-projects",
        "project": "personal-protective-equipment-combined-model",
        "version": 4,
        "class_remap": {
            "helmet": 0,
            "Helmet": 0,
            "hard-hat": 0,
            "Hardhat": 0,
            "head": 1,         # head without helmet = no_helmet
            "NO-Hardhat": 1,
            "no-helmet": 1,
            "no_helmet": 1,
            "mask": 2,
            "Mask": 2,
            "face_mask": 2,
            "with_mask": 2,
            "NO-Mask": 3,
            "no-mask": 3,
            "no_mask": 3,
            "without_mask": 3,
            "face_no_mask": 3,
        },
    },
    {
        "workspace": "joseph-nelson",
        "project": "mask-wearing",
        "version": 4,
        "class_remap": {
            "mask": 2,
            "Mask": 2,
            "with_mask": 2,
            "mask_weared_incorrect": 3,  # incorrect = effectively no mask
            "no-mask": 3,
            "no_mask": 3,
            "without_mask": 3,
            "WithoutMask": 3,
            "WithMask": 2,
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Roboflow datasets for helmet/mask PPE detection"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="ScDHuZFCVx9f4bWx81Cx",
        help="Roboflow API key",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/ppe_helmet_mask)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=6000,
        help="Minimum total images target",
    )
    return parser.parse_args()


def download_single_dataset(api_key: str, source: dict, download_dir: str) -> str:
    """
    Download a single dataset from Roboflow Universe.

    Returns the path to the downloaded dataset directory.
    """
    from roboflow import Roboflow

    print(f"\n📥 Downloading: {source['workspace']}/{source['project']} v{source['version']}")

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(source["workspace"])
    project = workspace.project(source["project"])
    version = project.version(source["version"])

    dataset = version.download(
        model_format="yolov8",
        location=download_dir,
        overwrite=True,
    )

    print(f"   ✅ Downloaded to: {download_dir}")
    return download_dir


def remap_labels(label_dir: str, class_remap: dict, source_classes: list = None):
    """
    Remap class indices in YOLO label files to unified class indices.

    If source_classes is provided, maps old_index -> source_classes[old_index] -> class_remap.
    Otherwise, tries to remap based on old index being already a name.
    """
    if not os.path.isdir(label_dir):
        print(f"   ⚠️  Label directory not found: {label_dir}")
        return 0, 0

    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    remapped = 0
    skipped = 0

    for label_file in label_files:
        if os.path.basename(label_file) == "classes.txt":
            continue

        new_lines = []
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            old_class_idx = int(parts[0])

            # Map old index → class name → unified index
            if source_classes and old_class_idx < len(source_classes):
                class_name = source_classes[old_class_idx]
                if class_name in class_remap:
                    new_class_idx = class_remap[class_name]
                    parts[0] = str(new_class_idx)
                    new_lines.append(" ".join(parts) + "\n")
                    remapped += 1
                else:
                    skipped += 1  # class not in our target set, skip
            else:
                skipped += 1

        # Write remapped labels (only keep lines with target classes)
        with open(label_file, "w") as f:
            f.writelines(new_lines)

        # Remove empty label files
        if not new_lines:
            os.remove(label_file)

    return remapped, skipped


def read_source_classes(dataset_dir: str) -> list:
    """Read class names from data.yaml or classes.txt in a downloaded dataset."""
    import yaml

    # Try data.yaml first
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            # Convert {0: 'class0', 1: 'class1'} to list
            max_idx = max(names.keys()) if names else -1
            class_list = [""] * (max_idx + 1)
            for idx, name in names.items():
                class_list[idx] = name
            return class_list
        return names

    # Try README.roboflow.txt or classes.txt
    for fname in ["classes.txt", "README.roboflow.txt"]:
        fpath = os.path.join(dataset_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                return [line.strip() for line in f if line.strip()]

    return []


def merge_into_output(source_dir: str, output_dir: str, split: str):
    """
    Copy images and labels from source split into output directory.

    Handles both `train/images` and `images/train` layouts.
    """
    # Detect directory layout
    possible_img_dirs = [
        os.path.join(source_dir, split, "images"),
        os.path.join(source_dir, split, "labels"),
        os.path.join(source_dir, "images", split),
        os.path.join(source_dir, "labels", split),
    ]

    # Find images directory
    img_src = None
    lbl_src = None
    for d in [os.path.join(source_dir, split, "images"),
              os.path.join(source_dir, "images", split)]:
        if os.path.isdir(d):
            img_src = d
            break

    for d in [os.path.join(source_dir, split, "labels"),
              os.path.join(source_dir, "labels", split)]:
        if os.path.isdir(d):
            lbl_src = d
            break

    if not img_src:
        print(f"   ⚠️  No images found for split '{split}' in {source_dir}")
        return 0

    img_dst = os.path.join(output_dir, split, "images")
    lbl_dst = os.path.join(output_dir, split, "labels")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    copied = 0
    for img_file in os.listdir(img_src):
        ext = os.path.splitext(img_file)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        stem = os.path.splitext(img_file)[0]
        lbl_file = stem + ".txt"

        # Only copy images that have a corresponding label file
        if lbl_src and os.path.exists(os.path.join(lbl_src, lbl_file)):
            # Check label file is not empty
            lbl_path = os.path.join(lbl_src, lbl_file)
            if os.path.getsize(lbl_path) == 0:
                continue

            # Use prefix to avoid filename collision between datasets
            shutil.copy2(
                os.path.join(img_src, img_file),
                os.path.join(img_dst, img_file),
            )
            shutil.copy2(lbl_path, os.path.join(lbl_dst, lbl_file))
            copied += 1

    return copied


def count_images(output_dir: str) -> dict:
    """Count images per split."""
    counts = {}
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(output_dir, split, "images")
        if os.path.isdir(img_dir):
            count = len([f for f in os.listdir(img_dir)
                        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
            counts[split] = count
        else:
            counts[split] = 0
    return counts


def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    output_dir = args.output or str(project_root / "datasets" / "ppe_helmet_mask")
    temp_dir = str(project_root / "datasets" / "_temp_downloads")

    print("=" * 70)
    print("🏭 Factory AI — Roboflow Dataset Downloader")
    print("=" * 70)
    print(f"  Output:      {output_dir}")
    print(f"  Min images:  {args.min_images}")
    print(f"  Sources:     {len(DATASET_SOURCES)} datasets")
    print("=" * 70)

    # Clean output directory
    if os.path.exists(output_dir):
        print(f"\n🗑️  Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Clean temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    total_images = 0

    for i, source in enumerate(DATASET_SOURCES):
        print(f"\n{'─' * 70}")
        print(f"📦 Dataset {i + 1}/{len(DATASET_SOURCES)}: {source['project']}")
        print(f"{'─' * 70}")

        dl_dir = os.path.join(temp_dir, f"dataset_{i}")

        try:
            download_single_dataset(args.api_key, source, dl_dir)
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
            print(f"   Skipping this dataset...")
            continue

        # Read source class names
        source_classes = read_source_classes(dl_dir)
        if source_classes:
            print(f"   📋 Source classes: {source_classes}")
        else:
            print(f"   ⚠️  Could not read source classes, attempting direct remap...")

        # Remap labels for each split
        for split in ["train", "valid", "test"]:
            for lbl_dir_candidate in [
                os.path.join(dl_dir, split, "labels"),
                os.path.join(dl_dir, "labels", split),
            ]:
                if os.path.isdir(lbl_dir_candidate):
                    remapped, skipped = remap_labels(
                        lbl_dir_candidate,
                        source["class_remap"],
                        source_classes,
                    )
                    print(f"   🔄 [{split}] Remapped: {remapped} labels, Skipped: {skipped}")
                    break

        # Merge into output
        for split in ["train", "valid", "test"]:
            copied = merge_into_output(dl_dir, output_dir, split)
            total_images += copied
            if copied > 0:
                print(f"   📂 [{split}] Copied: {copied} images")

        # Check if we've hit target
        if total_images >= args.min_images:
            print(f"\n✅ Reached target: {total_images} images ≥ {args.min_images}")
            break

    # Clean up temp
    print(f"\n🧹 Cleaning temp directory...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Final counts
    counts = count_images(output_dir)
    total = sum(counts.values())

    print(f"\n{'=' * 70}")
    print(f"📊 Dataset Summary")
    print(f"{'=' * 70}")
    print(f"  Train:       {counts.get('train', 0)} images")
    print(f"  Validation:  {counts.get('valid', 0)} images")
    print(f"  Test:        {counts.get('test', 0)} images")
    print(f"  Total:       {total} images")
    print(f"  Classes:     {list(UNIFIED_CLASSES.values())}")
    print(f"  Output:      {output_dir}")
    print(f"{'=' * 70}")

    if total < args.min_images:
        print(f"\n⚠️  Warning: Only {total} images downloaded (target: {args.min_images})")
        print(f"   You may need to add more Roboflow dataset sources.")
    else:
        print(f"\n✅ Successfully prepared {total} images for training!")

    # Write a data.yaml in the output directory for convenience
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# PPE Helmet/Mask Detection Dataset\n")
        f.write(f"# Auto-generated by download_dataset.py\n\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"nc: {len(UNIFIED_CLASSES)}\n\n")
        f.write(f"names:\n")
        for idx, name in UNIFIED_CLASSES.items():
            f.write(f"  {idx}: {name}\n")

    print(f"📝 Dataset YAML written to: {yaml_path}")

    return total


if __name__ == "__main__":
    total = main()
    sys.exit(0 if total > 0 else 1)
