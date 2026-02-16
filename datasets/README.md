# Dataset Preparation Guide

## 📸 Data Collection

### Factory Footage Sources
1. **Existing CCTV**: Export footage from your factory's security cameras
2. **Dedicated recording**: Set up cameras at key locations (loading docks, machinery areas, exits)
3. **Simulated scenarios**: Stage PPE violations, near-misses, and hazardous situations safely
4. **Shift variation**: Capture footage across different shifts, lighting, and weather conditions

### Recording Tips
- **Resolution**: Minimum 1080p (1920×1080)
- **Frame rate**: 25-30 FPS
- **Duration**: At least 2-4 hours per camera angle
- **Diversity**: Multiple angles, zoom levels, worker densities

---

## 🏷️ Annotation

### Tools
| Tool | Type | Best For |
|------|------|----------|
| [Roboflow](https://roboflow.com) | Cloud | Team collaboration, auto-augmentation, hosted datasets |
| [LabelImg](https://github.com/heartexlabs/labelImg) | Desktop | Offline annotation, YOLO format native |
| [CVAT](https://www.cvat.ai) | Self-hosted | Large teams, video annotation, interpolation |
| [Label Studio](https://labelstud.io) | Self-hosted | Custom workflows, ML-assisted labeling |

### Annotation Format (YOLO)
Each image needs a `.txt` file with the same name:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]:
```
1 0.45 0.62 0.12 0.35
15 0.70 0.50 0.15 0.60
```

### Annotation Guidelines
- Label every visible instance, even partially occluded
- Use tight bounding boxes (minimal background)
- Be consistent with class definitions
- Annotate at least 500-1000 images per class for good results

---

## 📦 Public Dataset Sources

| Dataset | Classes | Link |
|---------|---------|------|
| **COCO 2017** | person, car, truck (base) | [cocodataset.org](https://cocodataset.org) |
| **Safety Helmet** | helmet, no_helmet, person | [Roboflow Universe](https://universe.roboflow.com/search?q=safety+helmet) |
| **Fire Detection** | fire, smoke | [Roboflow Universe](https://universe.roboflow.com/search?q=fire+detection) |
| **PPE Detection** | vest, helmet, goggles | [Roboflow Universe](https://universe.roboflow.com/search?q=ppe+detection) |
| **Forklift** | forklift, person | [Roboflow Universe](https://universe.roboflow.com/search?q=forklift) |
| **Fall Detection** | fall, lying | [Roboflow Universe](https://universe.roboflow.com/search?q=fall+detection) |

---

## 🔀 Merging Multiple Datasets

### Steps
1. **Remap class IDs** to match `factory.yaml` numbering (0–25)
2. **Standardize format** — all to YOLO `.txt` format
3. **Combine** images and labels into `train/`, `val/`, `test/`
4. **Deduplicate** — remove near-identical images
5. **Verify** — spot-check labels for correctness

### Remapping Script
```python
import os
import glob

# Mapping: {source_class_id: factory_class_id}
CLASS_MAP = {
    0: 0,   # helmet -> helmet
    1: 1,   # no_helmet -> no_helmet
    2: 15,  # person -> person
}

for txt_file in glob.glob("source_labels/*.txt"):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    remapped = []
    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        if cls_id in CLASS_MAP:
            parts[0] = str(CLASS_MAP[cls_id])
            remapped.append(" ".join(parts))
    
    with open(txt_file, "w") as f:
        f.write("\n".join(remapped) + "\n")
```

---

## ⚖️ Class Balancing

### Problem
Factory datasets are typically imbalanced:
- **Common**: person, helmet, safety_vest (thousands of instances)
- **Rare**: fire, fall, hand_inside_machine (few instances)

### Solutions
1. **Oversample** rare classes by duplicating their images
2. **Augment** rare classes more aggressively
3. **Use focal loss** (built into YOLOv8 by default)
4. **Weighted sampling** — increase weight for rare classes in training
5. **Synthetic data** — generate rare scenarios via simulation or image editing

### Target Distribution
Aim for at most 10:1 ratio between most and least common classes.

---

## 🔧 Data Augmentation for Industrial Environments

### YOLOv8 Built-in Augmentations
Already configured in `training/train.py`:
- Mosaic (combines 4 images)
- MixUp (blends 2 images)
- Copy-paste (copies objects between images)
- HSV shifts, rotation, scaling, flipping

### Recommended Additional Augmentations (via Roboflow or Albumentations)
| Augmentation | Why |
|-------------|-----|
| **Brightness ±30%** | Simulates different lighting conditions |
| **Blur (motion + Gaussian)** | Camera shake / low-quality CCTV |
| **Noise (Gaussian)** | Sensor noise in low light |
| **Rain / fog overlay** | Weather on outdoor cameras |
| **Cutout / random erase** | Occlusion robustness |
| **CLAHE** | Contrast-limited histogram equalization for dark areas |

### Directory Structure After Preparation
```
datasets/
├── factory.yaml
├── images/
│   ├── train/    (70% — ~5000+ images)
│   ├── val/      (20% — ~1500+ images)
│   └── test/     (10% — ~750+ images)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Validation
```bash
# Verify dataset integrity
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt').val(data='datasets/factory.yaml')"
```
