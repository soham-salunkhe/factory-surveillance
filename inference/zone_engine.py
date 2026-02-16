"""
Zone Engine — Restricted area intrusion and proximity detection.

Implements polygon-based zone monitoring and bounding box distance
calculations for forklift–human collision risk detection.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Shapely for polygon operations (fallback to cv2 if unavailable)
try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    logger.warning("Shapely not installed. Using OpenCV for polygon checks. Install: pip install shapely")


class Zone:
    """Represents a restricted/monitored zone."""

    def __init__(self, name: str, polygon: List[List[int]], alert_classes: List[int] = None):
        """
        Args:
            name: Human-readable zone name.
            polygon: List of [x, y] vertices.
            alert_classes: Class IDs that trigger alerts in this zone.
        """
        self.name = name
        self.polygon_points = np.array(polygon, dtype=np.int32)
        self.alert_classes = set(alert_classes or [])

        # Create Shapely polygon if available
        if HAS_SHAPELY:
            self._shapely_poly = ShapelyPolygon(polygon)
        else:
            self._shapely_poly = None

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this zone."""
        if HAS_SHAPELY and self._shapely_poly:
            return self._shapely_poly.contains(Point(x, y))
        else:
            # Fallback to OpenCV pointPolygonTest
            result = cv2.pointPolygonTest(
                self.polygon_points.reshape(-1, 1, 2).astype(np.float32),
                (float(x), float(y)),
                False
            )
            return result >= 0

    def should_alert(self, class_id: int) -> bool:
        """Check if this class should trigger an alert in this zone."""
        if not self.alert_classes:
            return True  # Alert for all classes if none specified
        return class_id in self.alert_classes


class ZoneEngine:
    """
    Zone-based intrusion detection and proximity monitoring.

    Features:
        - Polygon zone intrusion detection (point-in-polygon)
        - Bounding box distance-based proximity alerts
        - Visual zone overlay on frames
        - Configurable alert classes per zone

    Usage:
        engine = ZoneEngine(zones_config, proximity_config)
        intrusions = engine.check_intrusions(detections)
        proximity_alerts = engine.check_proximity(detections)
        annotated = engine.draw_zones(frame)
    """

    def __init__(self, zones_config: list, proximity_config: dict = None):
        """
        Args:
            zones_config: List of zone dicts from settings.yaml.
            proximity_config: Proximity detection config from settings.yaml.
        """
        self.zones: List[Zone] = []
        self.proximity_rules: List[dict] = []

        # Load zones
        for zc in (zones_config or []):
            zone = Zone(
                name=zc["name"],
                polygon=zc["polygon"],
                alert_classes=zc.get("alert_classes"),
            )
            self.zones.append(zone)
            logger.info(f"📍 Zone loaded: '{zone.name}' ({len(zc['polygon'])} vertices)")

        # Load proximity rules
        if proximity_config:
            for rule_name, rule in proximity_config.items():
                self.proximity_rules.append({
                    "name": rule_name,
                    "class_a": rule["class_a"],
                    "class_b": rule["class_b"],
                    "threshold": rule["distance_threshold"],
                    "message": rule["alert_message"],
                })
                logger.info(f"📏 Proximity rule: '{rule_name}' (threshold={rule['distance_threshold']}px)")

    def check_intrusions(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """
        Check if any detections are inside restricted zones.

        Uses the bottom-center of the bounding box as the reference point
        (approximate ground contact point).

        Args:
            detections: List of detection dicts.

        Returns:
            List of intrusion events:
                {
                    'type': 'zone_intrusion',
                    'zone': str,
                    'detection': dict,
                    'point': (x, y),
                    'message': str
                }
        """
        events = []

        for det in detections:
            # Bottom-center of bounding box (ground contact point)
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = y2  # Bottom of bbox

            for zone in self.zones:
                if zone.contains_point(cx, cy):
                    class_id = det.get("class_id", -1)
                    if zone.should_alert(class_id):
                        event = {
                            "type": "zone_intrusion",
                            "zone": zone.name,
                            "detection": det,
                            "point": (cx, cy),
                            "message": (
                                f"🚫 ZONE INTRUSION: {det['class_name']} "
                                f"detected in '{zone.name}'"
                            ),
                        }
                        events.append(event)

        return events

    def check_proximity(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """
        Check proximity between specified class pairs.

        Measures Euclidean distance between bounding box centroids:
            If class_a detected AND class_b detected AND distance < threshold
            → Trigger proximity alert

        Args:
            detections: List of detection dicts.

        Returns:
            List of proximity alert events:
                {
                    'type': 'proximity_alert',
                    'rule': str,
                    'object_a': dict,
                    'object_b': dict,
                    'distance': float,
                    'message': str
                }
        """
        events = []

        for rule in self.proximity_rules:
            # Find all detections of class_a and class_b
            class_a_dets = [d for d in detections if d.get("class_id") == rule["class_a"]]
            class_b_dets = [d for d in detections if d.get("class_id") == rule["class_b"]]

            # Check all pairs
            for det_a in class_a_dets:
                cx_a, cy_a = self._get_centroid(det_a["bbox"])

                for det_b in class_b_dets:
                    cx_b, cy_b = self._get_centroid(det_b["bbox"])

                    distance = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

                    if distance < rule["threshold"]:
                        event = {
                            "type": "proximity_alert",
                            "rule": rule["name"],
                            "object_a": det_a,
                            "object_b": det_b,
                            "distance": round(distance, 1),
                            "message": rule["message"],
                        }
                        events.append(event)

        return events

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Draw zone overlays on the frame.

        Args:
            frame: Input BGR frame.
            alpha: Transparency of zone fill (0=transparent, 1=opaque).

        Returns:
            Frame with zone overlays.
        """
        result = frame.copy()
        overlay = frame.copy()

        for zone in self.zones:
            # Fill polygon with semi-transparent color
            cv2.fillPoly(overlay, [zone.polygon_points], (0, 0, 200))

            # Draw polygon outline
            cv2.polylines(result, [zone.polygon_points], True, (0, 0, 255), 2)

            # Zone label
            centroid = zone.polygon_points.mean(axis=0).astype(int)
            cv2.putText(
                result,
                f"⛔ {zone.name}",
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Blend overlay
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        return result

    def draw_proximity_alert(
        self,
        frame: np.ndarray,
        event: Dict,
    ) -> np.ndarray:
        """Draw a proximity alert visualization on the frame."""
        result = frame.copy()

        bbox_a = [int(c) for c in event["object_a"]["bbox"]]
        bbox_b = [int(c) for c in event["object_b"]["bbox"]]

        cx_a, cy_a = self._get_centroid(event["object_a"]["bbox"])
        cx_b, cy_b = self._get_centroid(event["object_b"]["bbox"])

        # Draw line between objects
        cv2.line(result, (int(cx_a), int(cy_a)), (int(cx_b), int(cy_b)), (0, 0, 255), 3)

        # Draw distance label
        mid_x = int((cx_a + cx_b) / 2)
        mid_y = int((cy_a + cy_b) / 2)
        cv2.putText(
            result,
            f"{event['distance']:.0f}px",
            (mid_x, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw warning icon
        cv2.putText(
            result,
            "!! COLLISION RISK !!",
            (mid_x - 80, mid_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return result

    @staticmethod
    def _get_centroid(bbox: list) -> Tuple[float, float]:
        """Get centroid of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
