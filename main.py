#!/usr/bin/env python3
"""
Factory AI — Main Entry Point

Orchestrates the complete real-time industrial surveillance pipeline:
    1. Load configuration
    2. Initialize CUDA / GPU
    3. Load YOLOv8 model
    4. Start video stream(s)
    5. Initialize tracker, zone engine, alert engine
    6. Start dashboard server
    7. Run inference loop (detect → track → zones → proximity → alert)
    8. Graceful shutdown on Ctrl+C

Usage:
    python main.py
    python main.py --config configs/settings.yaml
    python main.py --source 0                   # Webcam
    python main.py --source rtsp://ip/stream     # RTSP
    python main.py --weights models/best.engine  # TensorRT
    python main.py --no-dashboard                # Headless mode
"""

import os
import sys
import time
import signal
import argparse
import logging
from pathlib import Path

import cv2
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from inference.cuda_utils import CUDAManager
from inference.video_stream import VideoStream, MultiStreamManager
from inference.detect import DetectionEngine
from inference.tracker import ObjectTracker
from inference.zone_engine import ZoneEngine
from inference.alert_engine import AlertEngine
from dashboard.app import DashboardServer
from utils.logger import setup_logging, IncidentLogger
from utils.privacy import blur_faces
from utils.metrics import FPSCounter, InferenceTimer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Factory AI — Real-Time Industrial Safety Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/settings.yaml",
                        help="Path to configuration YAML")
    parser.add_argument("--source", type=str, default=None,
                        help="Override video source (RTSP URL, file, or webcam index)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Override model weights path")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Override confidence threshold")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable web dashboard")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV window display")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save annotated output to video file")
    return parser.parse_args()


class FactoryAI:
    """
    Main orchestrator for the Factory AI Surveillance System.

    Manages the complete pipeline from video capture through
    detection, tracking, zone checking, and alert dispatching.
    """

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.running = False

        # Components (initialized in setup())
        self.cuda_manager = None
        self.detection_engine = None
        self.tracker = None
        self.zone_engine = None
        self.alert_engine = None
        self.dashboard = None
        self.stream = None
        self.incident_logger = None
        self.fps_counter = FPSCounter(window_size=60)
        self.inference_timer = InferenceTimer()
        self.video_writer = None

    def setup(self):
        """Initialize all system components."""
        logger = logging.getLogger("factory_ai")

        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║         🏭  Factory AI Safety Monitoring System         ║")
        print("║              Powered by YOLOv8 + CUDA                   ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        # ── 1. CUDA Setup ──
        device_cfg = self.config.get("device", {})
        self.cuda_manager = CUDAManager(
            gpu_id=device_cfg.get("gpu_id", 0),
            fp16=device_cfg.get("fp16", True),
            cudnn_benchmark=device_cfg.get("cudnn_benchmark", True),
        )
        device = self.cuda_manager.setup()
        self.cuda_manager.start_monitoring(
            interval=self.config.get("performance", {}).get("gpu_monitor_interval", 5)
        )

        # ── 2. Detection Engine ──
        model_cfg = self.config.get("model", {})
        weights = self.args.weights or model_cfg.get("weights", "yolov8m.pt")
        confidence = self.args.confidence or model_cfg.get("confidence", 0.45)

        self.detection_engine = DetectionEngine(
            weights=weights,
            device=self.cuda_manager.get_device_string(),
            fp16=self.cuda_manager.fp16,
            confidence=confidence,
            iou_threshold=model_cfg.get("iou_threshold", 0.5),
            imgsz=model_cfg.get("imgsz", 640),
            classes=model_cfg.get("classes"),
        )

        # ── 3. Tracker ──
        tracker_cfg = self.config.get("tracker", {})
        self.tracker = ObjectTracker(
            tracker_type=tracker_cfg.get("type", "bytetrack"),
            track_buffer=tracker_cfg.get("track_buffer", 30),
            match_threshold=tracker_cfg.get("match_threshold", 0.8),
        )

        # ── 4. Zone Engine ──
        self.zone_engine = ZoneEngine(
            zones_config=self.config.get("zones", []),
            proximity_config=self.config.get("proximity"),
        )

        # ── 5. Alert Engine ──
        self.alert_engine = AlertEngine(self.config.get("alerts", {}))
        self.alert_engine.start()

        # ── 6. Incident Logger ──
        log_cfg = self.config.get("logging", {})
        self.incident_logger = IncidentLogger(
            filepath=os.path.join(log_cfg.get("log_dir", "logs"), "incidents.jsonl")
        )

        # ── 7. Video Stream ──
        sources = self.config.get("sources", [{"name": "Webcam", "url": 0, "enabled": True}])

        if self.args.source is not None:
            # Override with CLI source
            try:
                source = int(self.args.source)
            except ValueError:
                source = self.args.source
            self.stream = VideoStream(source=source, name="CLI Source")
        else:
            # Use first enabled source from config
            for src in sources:
                if src.get("enabled", True):
                    self.stream = VideoStream(source=src["url"], name=src["name"])
                    break

        if self.stream is None:
            logger.error("No video source configured!")
            sys.exit(1)

        self.stream.start()

        # ── 8. Dashboard ──
        dashboard_cfg = self.config.get("dashboard", {})
        if dashboard_cfg.get("enabled", True) and not self.args.no_dashboard:
            self.dashboard = DashboardServer(dashboard_cfg)
            self.dashboard.start()

        # ── 9. Video Writer (optional) ──
        if self.args.save_video:
            logger.info(f"📹 Video output: {self.args.save_video}")

        logger.info("")
        logger.info("✅ All systems initialized. Starting inference loop...")
        logger.info("   Press Ctrl+C to stop.")
        logger.info("")

    def run(self):
        """Main inference loop."""
        self.running = True
        logger = logging.getLogger("factory_ai")

        privacy_cfg = self.config.get("privacy", {})
        face_blur_enabled = privacy_cfg.get("face_blur", {}).get("enabled", False)
        blur_strength = privacy_cfg.get("face_blur", {}).get("blur_strength", 51)

        try:
            while self.running:
                # ── Read frame ──
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # ── Run detection + tracking ──
                with self.inference_timer:
                    detections = self.tracker.update(self.detection_engine, frame)

                # ── FPS tick ──
                self.fps_counter.tick()

                # ── Check zone intrusions ──
                intrusion_events = self.zone_engine.check_intrusions(detections)

                # ── Check proximity alerts ──
                proximity_events = self.zone_engine.check_proximity(detections)

                # ── Process detections for alerts ──
                self._process_detection_alerts(detections, frame)

                # ── Process zone intrusion alerts ──
                for event in intrusion_events:
                    det = event["detection"]
                    self.alert_engine.trigger({
                        "type": "zone_intrusion",
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "message": event["message"],
                        "frame": frame,
                        "camera": self.stream.name,
                        "zone": event["zone"],
                    })
                    self.incident_logger.log(
                        event_type="zone_intrusion",
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        camera=self.stream.name,
                        bbox=det["bbox"],
                        zone=event["zone"],
                    )

                # ── Process proximity alerts ──
                for event in proximity_events:
                    self.alert_engine.trigger({
                        "type": "proximity_alert",
                        "class_name": event["rule"],
                        "confidence": 1.0,
                        "message": event["message"],
                        "frame": frame,
                        "camera": self.stream.name,
                    })
                    self.incident_logger.log(
                        event_type="proximity_alert",
                        class_name=event["rule"],
                        confidence=1.0,
                        camera=self.stream.name,
                        extra={"distance": event["distance"]},
                    )

                # ── Annotate frame ──
                annotated = self.detection_engine.annotate_frame(frame, detections)
                annotated = self.zone_engine.draw_zones(annotated)

                # Draw proximity alerts
                for event in proximity_events:
                    annotated = self.zone_engine.draw_proximity_alert(annotated, event)

                # Face blur (privacy)
                if face_blur_enabled:
                    annotated = blur_faces(annotated, detections, blur_strength=blur_strength)

                # Add FPS overlay
                fps = self.fps_counter.get()
                latency = self.inference_timer.last_ms
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} | Latency: {latency:.1f}ms | Objects: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # ── Update dashboard ──
                if self.dashboard:
                    self.dashboard.update_frame(annotated)
                    self._update_dashboard_stats(detections, fps, latency)

                    # Send incidents to dashboard
                    for event in intrusion_events + proximity_events:
                        self.dashboard.add_incident({
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "type": event.get("type", "unknown"),
                            "class_name": event.get("detection", {}).get("class_name", event.get("rule", "")),
                            "confidence": event.get("detection", {}).get("confidence", 1.0),
                            "camera": self.stream.name,
                            "message": event.get("message", ""),
                        })

                # ── Display window ──
                if not self.args.no_display:
                    # Resize for display if needed
                    display = annotated
                    h, w = display.shape[:2]
                    if w > 1280:
                        scale = 1280 / w
                        display = cv2.resize(display, None, fx=scale, fy=scale)

                    cv2.imshow("Factory AI — Safety Monitoring", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit key pressed.")
                        break
                    elif key == ord("s"):
                        # Manual snapshot
                        snap_path = f"logs/snapshots/manual_{int(time.time())}.jpg"
                        os.makedirs("logs/snapshots", exist_ok=True)
                        cv2.imwrite(snap_path, annotated)
                        logger.info(f"📸 Manual snapshot: {snap_path}")

                # ── Save video ──
                if self.video_writer:
                    self.video_writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("\n⚡ Ctrl+C received. Shutting down...")

        finally:
            self.shutdown()

    def _process_detection_alerts(self, detections: list, frame):
        """Process detections for PPE, hazard, and behavior alerts."""
        for det in detections:
            category = det.get("category", "other")

            # Alert on PPE violations
            if category == "ppe_violation":
                self.alert_engine.trigger({
                    "type": "ppe_violation",
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "message": f"🦺 PPE VIOLATION: {det['class_name']} detected",
                    "frame": frame,
                    "camera": self.stream.name,
                    "bbox": det["bbox"],
                })
                self.incident_logger.log(
                    event_type="ppe_violation",
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    camera=self.stream.name,
                    bbox=det["bbox"],
                    track_id=det.get("track_id"),
                )

            # Alert on hazards (fire, smoke, etc.)
            elif category == "hazard":
                self.alert_engine.trigger({
                    "type": "hazard",
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "message": f"🔥 HAZARD DETECTED: {det['class_name']}!",
                    "frame": frame,
                    "camera": self.stream.name,
                    "bbox": det["bbox"],
                })
                self.incident_logger.log(
                    event_type="hazard",
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    camera=self.stream.name,
                    bbox=det["bbox"],
                )

            # Alert on unsafe behavior
            elif category == "unsafe_behavior":
                self.alert_engine.trigger({
                    "type": "unsafe_behavior",
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "message": f"⚠️ UNSAFE BEHAVIOR: {det['class_name']}",
                    "frame": frame,
                    "camera": self.stream.name,
                    "bbox": det["bbox"],
                })
                self.incident_logger.log(
                    event_type="unsafe_behavior",
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    camera=self.stream.name,
                    bbox=det["bbox"],
                    track_id=det.get("track_id"),
                )

    def _update_dashboard_stats(self, detections: list, fps: float, latency: float):
        """Update dashboard with latest statistics."""
        gpu_stats = self.cuda_manager.get_stats()

        # Count categories
        categories = {}
        for det in detections:
            cat = det.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        self.dashboard.update_stats({
            "fps": fps,
            "inference_ms": latency,
            "detection_count": len(detections),
            "tracking_count": self.tracker.active_count,
            "categories": categories,
            "alert_count": self.alert_engine.stats["total_alerts"],
            "gpu": {
                "device_name": gpu_stats.device_name,
                "vram_used_mb": gpu_stats.vram_used_mb,
                "vram_total_mb": gpu_stats.vram_total_mb,
                "vram_percent": gpu_stats.vram_percent,
                "gpu_utilization": gpu_stats.gpu_utilization,
                "temperature": gpu_stats.temperature,
                "power_draw_w": gpu_stats.power_draw_w,
            },
        })

    def shutdown(self):
        """Graceful shutdown of all components."""
        logger = logging.getLogger("factory_ai")
        logger.info("")
        logger.info("🛑 Shutting down Factory AI...")

        self.running = False

        if self.stream:
            self.stream.stop()
        if self.alert_engine:
            self.alert_engine.stop()
        if self.cuda_manager:
            self.cuda_manager.stop_monitoring()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        logger.info("✅ Factory AI stopped. Goodbye!")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print(f"   Copy configs/settings.yaml and configure your settings.")
        sys.exit(1)

    config = load_config(config_path)

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        log_dir=os.path.join(PROJECT_ROOT, log_cfg.get("log_dir", "logs")),
        level=log_cfg.get("level", "INFO"),
        max_log_size_mb=log_cfg.get("max_log_size_mb", 50),
        backup_count=log_cfg.get("backup_count", 5),
    )

    # Change to project directory
    os.chdir(PROJECT_ROOT)

    # Register signal handler for graceful shutdown
    system = FactoryAI(config, args)
    signal.signal(signal.SIGINT, lambda s, f: setattr(system, "running", False))

    # Run
    system.setup()
    system.run()


if __name__ == "__main__":
    main()
