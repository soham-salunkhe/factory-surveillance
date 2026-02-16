"""
YOLOv8 Detection Engine — FP16 mixed precision inference with CUDA.

Core detection module that wraps Ultralytics YOLOv8 for real-time
industrial safety monitoring. Supports FP16 half-precision, configurable
confidence/NMS thresholds, and batch inference.
"""

import logging
import time
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    YOLOv8 detection engine optimized for RTX 4060.

    Features:
        - FP16 half-precision inference
        - Mixed precision with torch.cuda.amp.autocast()
        - Configurable confidence and NMS thresholds
        - Batch inference support
        - Automatic model loading (PyTorch, TensorRT, ONNX)

    Usage:
        engine = DetectionEngine(
            weights="models/best.pt",
            device="cuda:0",
            fp16=True,
            confidence=0.45,
            iou_threshold=0.5,
            imgsz=640
        )
        detections = engine.detect(frame)
    """

    # Detection class categories for alert routing
    PPE_VIOLATIONS = {"no_helmet", "no_vest", "no_gloves", "no_goggles", "no_mask"}
    HAZARDS = {"fire", "smoke", "sparks", "gas_leak", "oil_spill"}
    UNSAFE_BEHAVIOR = {"fall", "lying_person", "running", "unsafe_climb", "restricted_area_intrusion"}
    MACHINERY_RISK = {"forklift", "heavy_machine", "blocked_exit", "conveyor_belt", "hand_inside_machine"}

    def __init__(
        self,
        weights: str = "yolov8m.pt",
        device: str = "cuda:0",
        fp16: bool = True,
        confidence: float = 0.45,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        classes: Optional[list] = None,
    ):
        """
        Initialize the detection engine.

        Args:
            weights: Path to model weights (.pt, .engine, .onnx).
            device: Torch device string.
            fp16: Enable FP16 half-precision.
            confidence: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            imgsz: Inference image size.
            classes: List of class indices to detect (None = all).
        """
        self.weights = weights
        self.device = device
        self.fp16 = fp16
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.classes = classes
        self.model = None
        self._class_names = {}

        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model with optimizations."""
        from ultralytics import YOLO

        logger.info(f"📦 Loading model: {self.weights}")
        logger.info(f"   Device: {self.device} | FP16: {self.fp16} | ImgSz: {self.imgsz}")

        self.model = YOLO(self.weights)

        # Move to device and apply optimizations
        device_type = self.device.split(":")[0] if ":" in self.device else self.device
        if device_type == "cuda" and torch.cuda.is_available():
            if self.weights.endswith(".pt") and self.fp16:
                self.model.model.half()
                logger.info("✅ Model converted to FP16 half-precision")

        # Cache class names
        if hasattr(self.model, "names"):
            self._class_names = self.model.names
            logger.info(f"   Classes: {len(self._class_names)} loaded")

        # Warmup inference to initialize CUDA kernels
        self._warmup()

    def _warmup(self, n: int = 3):
        """Run warmup inferences to initialize CUDA kernels and cuDNN auto-tuner."""
        logger.info(f"🔥 Warming up model ({n} iterations)...")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(
                dummy,
                device=self.device,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        logger.info("✅ Warmup complete")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.

        Args:
            frame: Input BGR frame from OpenCV.

        Returns:
            List of detection dicts:
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'category': str  # ppe, hazard, unsafe, machinery, other
                }
        """
        # Run inference
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False,
        )

        return self._parse_results(results)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of BGR frames.

        Returns:
            List of detection lists, one per frame.
        """
        results = self.model.predict(
            frames,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False,
        )

        return [self._parse_results([r]) for r in results]

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml") -> List[Dict[str, Any]]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: Input BGR frame.
            tracker: Tracker config name (bytetrack.yaml or botsort.yaml).

        Returns:
            List of detection dicts with additional 'track_id' field.
        """
        results = self.model.track(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.classes,
            tracker=tracker,
            persist=True,
            verbose=False,
        )

        return self._parse_results(results, include_track_id=True)

    def _parse_results(
        self, results, include_track_id: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse Ultralytics results into standardized detection dicts."""
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                det = {
                    "bbox": [round(c, 1) for c in bbox],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "category": self._get_category(cls_name),
                }

                # Add track ID if tracking
                if include_track_id and boxes.id is not None:
                    det["track_id"] = int(boxes.id[i].cpu())
                elif include_track_id:
                    det["track_id"] = -1

                detections.append(det)

        return detections

    def _get_category(self, class_name: str) -> str:
        """Categorize a detection class name."""
        if class_name in self.PPE_VIOLATIONS:
            return "ppe_violation"
        elif class_name in self.HAZARDS:
            return "hazard"
        elif class_name in self.UNSAFE_BEHAVIOR:
            return "unsafe_behavior"
        elif class_name in self.MACHINERY_RISK:
            return "machinery_risk"
        return "other"

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Input BGR frame.
            detections: List of detection dicts from detect().
            show_labels: Whether to show class names.
            show_confidence: Whether to show confidence scores.

        Returns:
            Annotated frame copy.
        """
        result = frame.copy()

        # Color mapping by category
        colors = {
            "ppe_violation": (0, 0, 255),      # Red
            "hazard": (0, 69, 255),             # Orange
            "unsafe_behavior": (0, 165, 255),   # Orange-yellow
            "machinery_risk": (0, 255, 255),    # Yellow
            "other": (0, 255, 0),               # Green
        }

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = colors.get(det.get("category", "other"), (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Build label
            label_parts = []
            if show_labels:
                label_parts.append(det["class_name"])
            if show_confidence:
                label_parts.append(f"{det['confidence']:.0%}")
            if "track_id" in det and det["track_id"] >= 0:
                label_parts.append(f"ID:{det['track_id']}")

            label = " | ".join(label_parts)

            if label:
                # Background rectangle for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

        return result

    @property
    def class_names(self) -> dict:
        """Get the model's class name mapping."""
        return self._class_names
