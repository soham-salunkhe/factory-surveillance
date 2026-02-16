# 🏭 Factory AI — Industrial Safety Monitoring System

**Real-time AI-powered surveillance for smart factories using YOLOv8 + CUDA acceleration.**

Detects PPE violations, fire/smoke hazards, unsafe worker behavior, forklift collision risks, and restricted zone intrusions from live CCTV/RTSP streams — optimized for NVIDIA RTX 4060.

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to project
cd factory_ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run with webcam (default)
python main.py

# 5. Open dashboard
# http://localhost:5000  (admin/factory2024)
```

### Run with RTSP Camera
```bash
python main.py --source "rtsp://admin:pass@192.168.1.100:554/stream"
```

### Run with TensorRT (fastest)
```bash
# Export first
python -m factory_ai.models.export --weights models/best.pt --format engine --half
# Run with engine
python main.py --weights models/best.engine
```

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| **PPE Detection** | Helmet, vest, gloves, goggles, mask violations |
| **Hazard Detection** | Fire, smoke, sparks, gas leak, oil spill |
| **Behavior Analysis** | Fall, lying person, running, unsafe climbing |
| **Collision Prevention** | Forklift–human proximity alerts |
| **Zone Intrusion** | Polygon-based restricted area monitoring |
| **Multi-Camera** | Multiple RTSP/webcam streams |
| **Object Tracking** | ByteTrack with dwell time & trajectory |
| **Real-time Dashboard** | Live video + GPU stats + incident log |
| **Telegram Alerts** | Photo + caption alerts via Telegram bot |
| **CSV Logging** | Structured incident records |
| **FP16 Inference** | Half-precision for 2x speed on RTX 4060 |
| **TensorRT** | Optimized engine for maximum FPS |
| **Face Blurring** | Privacy-preserving head region blur |
| **Log Encryption** | Fernet symmetric encryption |

---

## 🏗️ Architecture

```
factory_ai/
├── configs/settings.yaml      # All tunables
├── datasets/
│   ├── factory.yaml           # 26 detection classes
│   └── README.md              # Dataset preparation guide
├── models/export.py           # TensorRT/ONNX export
├── training/train.py          # Training with RTX 4060 optimizations
├── inference/
│   ├── detect.py              # YOLOv8 detection engine (FP16)
│   ├── tracker.py             # ByteTrack with history
│   ├── zone_engine.py         # Zone intrusion + proximity
│   ├── alert_engine.py        # Multi-channel alerts
│   ├── cuda_utils.py          # GPU setup & monitoring
│   └── video_stream.py        # Threaded RTSP capture
├── dashboard/                 # Flask web UI
├── utils/
│   ├── logger.py              # Structured logging
│   ├── privacy.py             # Face blur + encryption
│   └── metrics.py             # FPS, mAP, precision
├── main.py                    # Entry point
├── Dockerfile
└── requirements.txt
```

### Pipeline Flow
```
Camera → VideoStream (thread) → DetectionEngine (FP16/TensorRT)
    → ObjectTracker (ByteTrack) → ZoneEngine (intrusion + proximity)
    → AlertEngine (thread: Telegram/CSV/siren) + Dashboard (Flask)
```

---

## ⚡ CUDA Optimization (RTX 4060)

| Optimization | Detail |
|-------------|--------|
| **CUDA Device** | Auto-detect, `torch.cuda.set_device()` |
| **cuDNN Benchmark** | `cudnn.benchmark = True` for optimal conv kernels |
| **FP16 Half-Precision** | `model.half()` + `torch.cuda.amp.autocast()` |
| **TensorRT Export** | `yolo export format=engine half=True` |
| **Warmup** | 3 dummy inferences to initialize CUDA kernels |
| **VRAM Monitoring** | NVML-based GPU utilization, temp, power |
| **Threaded I/O** | Separate threads for capture, inference, alerts |

### Expected Performance (RTX 4060)
| Model | Format | FPS (640px) | Latency |
|-------|--------|-------------|---------|
| YOLOv8m | PyTorch FP16 | ~45-60 | ~18ms |
| YOLOv8m | TensorRT FP16 | ~80-120 | ~10ms |
| YOLOv8s | TensorRT FP16 | ~120-160 | ~7ms |
| YOLOv8n | TensorRT FP16 | ~180-250 | ~5ms |

---

## 🎯 Detection Classes (26 total)

### 🦺 PPE Violations (0–9)
`helmet` `no_helmet` `safety_vest` `no_vest` `gloves` `no_gloves` `goggles` `no_goggles` `mask` `no_mask`

### 🔥 Hazard Detection (10–14)
`fire` `smoke` `sparks` `gas_leak` `oil_spill`

### 👷 Worker Safety (15–20)
`person` `fall` `lying_person` `running` `unsafe_climb` `restricted_area_intrusion`

### 🚜 Machinery Risk (21–25)
`forklift` `heavy_machine` `blocked_exit` `conveyor_belt` `hand_inside_machine`

---

## 🏋️ Training

### Quick Train
```bash
python -m factory_ai.training.train \
    --model yolov8m.pt \
    --data datasets/factory.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### RTX 4060 Batch Size Guide
| Model | imgsz=640 | imgsz=960 |
|-------|-----------|-----------|
| YOLOv8n | batch=32 | batch=16 |
| YOLOv8s | batch=24 | batch=12 |
| YOLOv8m | batch=16 | batch=8 |
| YOLOv8l | batch=8 | batch=4 |
| YOLOv8x | batch=4 | batch=2 |

### Resume Training
```bash
python -m factory_ai.training.train --resume runs/detect/train/weights/last.pt
```

### Dataset Preparation
See [datasets/README.md](datasets/README.md) for detailed instructions on:
- Collecting factory footage
- Annotation with Roboflow/LabelImg
- Merging multiple datasets
- Class balancing strategies
- Augmentation for industrial environments

---

## 🚨 Alert System

### Channels
| Channel | Config Key | Description |
|---------|-----------|-------------|
| **Snapshot** | `alerts.snapshot` | Saves annotated JPEG to `logs/snapshots/` |
| **CSV** | `alerts.csv` | Appends to `logs/incidents.csv` |
| **Telegram** | `alerts.telegram` | Photo + caption via Bot API |
| **Siren** | `alerts.siren` | Plays WAV via system audio |
| **REST API** | `alerts.rest_api` | POST JSON to external endpoint |

### Telegram Setup
1. Message [@BotFather](https://t.me/botfather) → `/newbot`
2. Copy the bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
4. Update `configs/settings.yaml`:
```yaml
alerts:
  telegram:
    enabled: true
    bot_token: "123456:ABC-DEF..."
    chat_id: "987654321"
```

### Collision Risk Logic
```
IF forklift detected
AND person detected
AND centroid_distance < 150px
→ TRIGGER "Collision Risk Alert"
→ Save snapshot + Send Telegram + Log CSV
```

---

## 🖥️ Deployment Options

### Local PC (RTX 4060)
```bash
python main.py --source 0  # Default mode
```

### Multi-Camera
Edit `configs/settings.yaml`:
```yaml
sources:
  - name: "Main Floor"
    url: "rtsp://admin:pass@cam1:554/stream"
    enabled: true
  - name: "Loading Dock"
    url: "rtsp://admin:pass@cam2:554/stream"
    enabled: true
```

### Docker
```bash
docker build -t factory-ai .
docker run --gpus all -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    factory-ai
```

### TensorRT Deployment
```bash
# Export (run once per GPU architecture)
python -m factory_ai.models.export --weights models/best.pt --format engine --half

# Run with engine
python main.py --weights models/best.engine
```

### ONNX Export
```bash
python -m factory_ai.models.export --weights models/best.pt --format onnx
```

---

## 📊 Performance Metrics

### Model Evaluation
```bash
# Run validation
python -c "
from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.val(data='datasets/factory.yaml')
print(f'mAP@50: {results.box.map50:.4f}')
print(f'mAP@50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
"
```

### Benchmark Formats
```bash
python -m factory_ai.models.export --format benchmark --weights models/best.pt
```

---

## 🔐 Privacy Features

| Feature | Config | Description |
|---------|--------|-------------|
| **Face Blur** | `privacy.face_blur.enabled` | Gaussian blur on head region |
| **Log Encryption** | `privacy.log_encryption.enabled` | Fernet AES encryption |
| **Basic Auth** | `dashboard.auth` | Username/password for dashboard |
| **Local Storage** | Default | All data stays on-premises |

---

## 🧠 Advanced Features (Future Integration)

### LSTM Behavior Anomaly Detection
Use track trajectory sequences as input to an LSTM network for detecting unusual movement patterns (e.g., erratic walking, prolonged loitering in danger zones). Feed centroid histories from `ObjectTracker` into a sequence classifier.

### Autoencoder-Based Anomaly Detection
Train a convolutional autoencoder on "normal" factory scenes. High reconstruction error indicates anomalies (unexpected objects, unusual configurations). Useful for detecting novel hazards not in the training set.

### Vision Transformer (ViT) for Crowd Anomaly
For large factory floors with many workers, use a ViT-based classifier on scene-level features to detect crowd anomalies: stampedes, mass evacuations, or unusual clustering.

### Sound Anomaly Detection
Integrate microphone input with an audio classifier (e.g., YAMNet) to detect explosion sounds, machinery failures, screams, or glass breaking. Fuse with visual detections for higher confidence alerts.

### Thermal Camera Integration
Add FLIR/thermal camera streams as additional input channels. Thermal data excels at fire detection, overheating machinery, and human detection in smoke-filled environments.

---

## 💰 Startup MVP & Pricing Model

### MVP Feature Set (Month 1–3)
1. **Core Detection**: PPE + fire/smoke (2 categories)
2. **Single Camera**: One RTSP stream
3. **Dashboard**: Live feed + incident log
4. **Alerts**: Telegram + CSV logging
5. **Hardware**: RTX 4060 workstation

### Pricing Tiers

| Tier | Cameras | Features | Price/mo |
|------|---------|----------|----------|
| **Starter** | 1-2 | PPE + Hazard detection, Basic alerts | $299 |
| **Professional** | 4-8 | All detections, Zones, Multi-channel alerts | $799 |
| **Enterprise** | 16+ | Custom models, API, Multi-site, SLA | $1,999+ |

### Revenue Streams
- **SaaS License**: Monthly camera-based pricing
- **Setup Fee**: One-time installation + model training ($2,000–5,000)
- **Custom Training**: Domain-specific model fine-tuning
- **Hardware Bundle**: Pre-configured GPU workstations
- **Support**: 24/7 support contracts

### Scaling Roadmap
1. **Phase 1** (MVP): Single factory, RTX 4060, core detections
2. **Phase 2**: Multi-camera, TensorRT, mobile alerts
3. **Phase 3**: Multi-site, cloud dashboard, custom models
4. **Phase 4**: Edge deployment (Jetson), thermal/audio fusion
5. **Phase 5**: SaaS platform, marketplace for industry-specific models

---

## 🛠️ CLI Reference

```bash
# Standard run
python main.py

# With options
python main.py --config configs/settings.yaml \
               --source "rtsp://ip/stream" \
               --weights models/best.engine \
               --confidence 0.5 \
               --no-display \
               --save-video output.mp4

# Training
python -m factory_ai.training.train --help

# Export
python -m factory_ai.models.export --help
```

---

## 📄 License

This project is for educational and commercial use. Ensure compliance with local surveillance and privacy regulations when deploying in production.

---

**Built with ❤️ using YOLOv8 + PyTorch + CUDA**
