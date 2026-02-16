"""
Object Tracker — ByteTrack integration via Ultralytics.

Wraps YOLOv8's built-in ByteTrack/BoTSORT tracker with track history
management for trajectory analysis, dwell time, and behavior monitoring.
"""

import logging
import time
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class TrackInfo:
    """Stores tracking metadata for a single object."""

    def __init__(self, track_id: int, class_name: str, max_history: int = 90):
        self.track_id = track_id
        self.class_name = class_name
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.centroids: deque = deque(maxlen=max_history)
        self.bboxes: deque = deque(maxlen=max_history)
        self.confidences: deque = deque(maxlen=max_history)
        self.frame_count = 0

    def update(self, bbox: list, confidence: float):
        """Update track with new detection."""
        self.last_seen = time.time()
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        self.frame_count += 1

        # Compute centroid (bottom-center for ground contact point)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = y2  # Bottom center
        self.centroids.append((cx, cy))

    @property
    def dwell_time(self) -> float:
        """Time in seconds since first detection."""
        return time.time() - self.first_seen

    @property
    def current_centroid(self) -> Optional[Tuple[float, float]]:
        """Latest centroid position."""
        return self.centroids[-1] if self.centroids else None

    @property
    def current_bbox(self) -> Optional[list]:
        """Latest bounding box."""
        return self.bboxes[-1] if self.bboxes else None

    @property
    def speed(self) -> float:
        """Estimated speed in pixels/frame based on last 5 centroids."""
        if len(self.centroids) < 2:
            return 0.0
        points = list(self.centroids)[-5:]
        total_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            total_dist += (dx**2 + dy**2) ** 0.5
        return total_dist / (len(points) - 1)

    @property
    def is_stationary(self) -> bool:
        """Whether the object has moved very little recently."""
        return self.speed < 2.0


class ObjectTracker:
    """
    Multi-object tracker with history and behavior analysis.

    Wraps the detection engine's built-in ByteTrack/BoTSORT and adds:
    - Per-object track history (centroids, bboxes, timestamps)
    - Dwell time calculation
    - Speed estimation
    - Stationary detection (potential fall/lying person)
    - Track cleanup for lost objects

    Usage:
        tracker = ObjectTracker(tracker_type="bytetrack")
        detections = tracker.update(engine, frame)
        for det in detections:
            info = tracker.get_track_info(det["track_id"])
            print(f"ID {info.track_id}: dwell={info.dwell_time:.1f}s speed={info.speed:.1f}")
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        max_history: int = 90,
        lost_timeout: float = 10.0,
    ):
        """
        Args:
            tracker_type: 'bytetrack' or 'botsort'.
            track_buffer: Frames to keep lost tracks in the tracker.
            match_threshold: IoU threshold for track matching.
            max_history: Max centroid history per track.
            lost_timeout: Seconds before removing a lost track.
        """
        self.tracker_config = f"{tracker_type}.yaml"
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.max_history = max_history
        self.lost_timeout = lost_timeout

        # Track storage: track_id -> TrackInfo
        self._tracks: Dict[int, TrackInfo] = {}
        self._active_ids: set = set()

    def update(self, engine, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection + tracking on a frame and update track histories.

        Args:
            engine: DetectionEngine instance.
            frame: Input BGR frame.

        Returns:
            List of detection dicts with track_id and enriched metadata.
        """
        # Run detection with tracking
        detections = engine.track(frame, tracker=self.tracker_config)

        current_ids = set()

        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id < 0:
                continue

            current_ids.add(track_id)

            # Create or update track info
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackInfo(
                    track_id=track_id,
                    class_name=det["class_name"],
                    max_history=self.max_history,
                )
                logger.debug(f"New track: ID={track_id} class={det['class_name']}")

            self._tracks[track_id].update(det["bbox"], det["confidence"])

            # Enrich detection with tracking metadata
            info = self._tracks[track_id]
            det["dwell_time"] = round(info.dwell_time, 1)
            det["speed"] = round(info.speed, 1)
            det["is_stationary"] = info.is_stationary
            det["centroid"] = info.current_centroid

        self._active_ids = current_ids

        # Cleanup lost tracks
        self._cleanup_lost_tracks()

        return detections

    def _cleanup_lost_tracks(self):
        """Remove tracks that haven't been seen recently."""
        now = time.time()
        lost_ids = [
            tid for tid, info in self._tracks.items()
            if tid not in self._active_ids and (now - info.last_seen) > self.lost_timeout
        ]
        for tid in lost_ids:
            del self._tracks[tid]
            logger.debug(f"Removed lost track: ID={tid}")

    def get_track_info(self, track_id: int) -> Optional[TrackInfo]:
        """Get tracking metadata for a specific ID."""
        return self._tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, TrackInfo]:
        """Get all active tracks."""
        return {
            tid: info for tid, info in self._tracks.items()
            if tid in self._active_ids
        }

    def get_tracks_by_class(self, class_name: str) -> List[TrackInfo]:
        """Get all active tracks of a specific class."""
        return [
            info for tid, info in self._tracks.items()
            if tid in self._active_ids and info.class_name == class_name
        ]

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_ids)

    @property
    def total_tracks(self) -> int:
        """Total unique objects tracked."""
        return len(self._tracks)
