"""
Alert Engine — Multi-channel incident alerting with cooldown.

Processes detection events and dispatches alerts through multiple
channels: snapshots, CSV logging, Telegram, siren, and REST API.
Runs alert processing in a background thread with per-type cooldown.
"""

import os
import csv
import json
import time
import logging
import threading
import subprocess
from queue import Queue, Empty
from datetime import datetime
from typing import Dict, Optional, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Multi-channel alert dispatcher with cooldown and queued processing.

    Alerts are pushed to a queue and processed by a background thread
    to avoid blocking the inference pipeline. Each alert type has a
    configurable cooldown period to prevent notification spam.

    Supported channels:
        - Snapshot: Save annotated frame as JPEG
        - CSV: Append incident record to CSV file
        - Telegram: Send photo + caption via Bot API
        - Siren: Play local WAV siren sound
        - REST API: POST alert to an external endpoint

    Usage:
        alert = AlertEngine(config["alerts"])
        alert.start()
        alert.trigger({
            "type": "ppe_violation",
            "class_name": "no_helmet",
            "confidence": 0.92,
            "message": "Worker without helmet detected",
            "frame": annotated_frame,
            "camera": "Main Floor",
        })
        ...
        alert.stop()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Alert configuration dict from settings.yaml.
        """
        self.enabled = config.get("enabled", True)
        self.cooldown = config.get("cooldown_seconds", 30)

        # Channel configs
        self.snapshot_config = config.get("snapshot", {})
        self.csv_config = config.get("csv", {})
        self.telegram_config = config.get("telegram", {})
        self.siren_config = config.get("siren", {})
        self.rest_config = config.get("rest_api", {})

        # Cooldown tracking: {alert_type: last_alert_time}
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # Alert queue and worker
        self._queue: Queue = Queue(maxsize=100)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Stats
        self._alert_count = 0
        self._alerts_suppressed = 0

        # Initialize CSV if enabled
        if self.csv_config.get("enabled"):
            self._init_csv()

        # Create snapshot directory
        if self.snapshot_config.get("enabled"):
            os.makedirs(self.snapshot_config.get("directory", "logs/snapshots"), exist_ok=True)

    def start(self):
        """Start the background alert processing thread."""
        if not self.enabled:
            logger.info("Alert engine disabled in config.")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="AlertEngine",
        )
        self._worker_thread.start()
        logger.info(f"🔔 Alert Engine started (cooldown={self.cooldown}s)")

    def trigger(self, event: Dict[str, Any]):
        """
        Queue an alert event for processing.

        Args:
            event: Alert event dict with keys:
                - type: str (ppe_violation, hazard, zone_intrusion, proximity_alert, etc.)
                - class_name: str
                - confidence: float
                - message: str
                - frame: np.ndarray (optional, for snapshot)
                - camera: str (optional)
                - bbox: list (optional)
                - zone: str (optional)
        """
        if not self.enabled:
            return

        # Check cooldown
        alert_key = f"{event.get('type', 'unknown')}_{event.get('class_name', '')}"
        with self._cooldown_lock:
            last_time = self._cooldowns.get(alert_key, 0)
            if time.time() - last_time < self.cooldown:
                self._alerts_suppressed += 1
                return
            self._cooldowns[alert_key] = time.time()

        # Add timestamp
        event["timestamp"] = datetime.now().isoformat()

        # Queue the alert
        try:
            self._queue.put_nowait(event)
        except Exception:
            logger.warning("Alert queue full. Dropping alert.")

    def _process_loop(self):
        """Background alert processing loop."""
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._process_alert(event)
                self._alert_count += 1
            except Exception as e:
                logger.error(f"Alert processing error: {e}")

    def _process_alert(self, event: Dict):
        """Process a single alert through all enabled channels."""
        logger.info(
            f"🚨 ALERT [{event.get('type', 'unknown').upper()}]: "
            f"{event.get('message', 'No message')}"
        )

        snapshot_path = None

        # 1. Save snapshot
        if self.snapshot_config.get("enabled") and event.get("frame") is not None:
            snapshot_path = self._save_snapshot(event)
            event["snapshot_path"] = snapshot_path

        # 2. Log to CSV
        if self.csv_config.get("enabled"):
            self._log_csv(event)

        # 3. Send Telegram alert
        if self.telegram_config.get("enabled"):
            self._send_telegram(event, snapshot_path)

        # 4. Play siren
        if self.siren_config.get("enabled"):
            self._play_siren()

        # 5. POST to REST API
        if self.rest_config.get("enabled"):
            self._post_rest(event)

    def _save_snapshot(self, event: Dict) -> Optional[str]:
        """Save annotated frame as JPEG snapshot."""
        try:
            directory = self.snapshot_config.get("directory", "logs/snapshots")
            os.makedirs(directory, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            alert_type = event.get("type", "alert").replace(" ", "_")
            filename = f"{alert_type}_{timestamp}.jpg"
            filepath = os.path.join(directory, filename)

            cv2.imwrite(filepath, event["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug(f"📸 Snapshot saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Snapshot save failed: {e}")
            return None

    def _init_csv(self):
        """Initialize CSV file with headers."""
        filepath = self.csv_config.get("filepath", "logs/incidents.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not os.path.exists(filepath):
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "type", "class_name", "confidence",
                    "camera", "zone", "message", "snapshot_path"
                ])
            logger.info(f"📋 CSV incident log created: {filepath}")

    def _log_csv(self, event: Dict):
        """Append incident record to CSV."""
        try:
            filepath = self.csv_config.get("filepath", "logs/incidents.csv")
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.get("timestamp", ""),
                    event.get("type", ""),
                    event.get("class_name", ""),
                    event.get("confidence", ""),
                    event.get("camera", ""),
                    event.get("zone", ""),
                    event.get("message", ""),
                    event.get("snapshot_path", ""),
                ])
        except Exception as e:
            logger.error(f"CSV logging failed: {e}")

    def _send_telegram(self, event: Dict, photo_path: Optional[str] = None):
        """Send alert via Telegram Bot API."""
        try:
            import requests

            token = self.telegram_config.get("bot_token", "")
            chat_id = self.telegram_config.get("chat_id", "")

            if not token or not chat_id or token == "YOUR_BOT_TOKEN":
                logger.debug("Telegram not configured. Skipping.")
                return

            caption = (
                f"🚨 *Factory AI Alert*\n\n"
                f"*Type:* {event.get('type', 'unknown')}\n"
                f"*Class:* {event.get('class_name', 'N/A')}\n"
                f"*Confidence:* {event.get('confidence', 0):.0%}\n"
                f"*Camera:* {event.get('camera', 'N/A')}\n"
                f"*Time:* {event.get('timestamp', 'N/A')}\n"
                f"*Message:* {event.get('message', '')}"
            )

            if photo_path and os.path.exists(photo_path):
                # Send photo with caption
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(photo_path, "rb") as photo:
                    resp = requests.post(
                        url,
                        data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
                        files={"photo": photo},
                        timeout=10,
                    )
            else:
                # Send text-only message
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                resp = requests.post(
                    url,
                    data={"chat_id": chat_id, "text": caption, "parse_mode": "Markdown"},
                    timeout=10,
                )

            if resp.status_code == 200:
                logger.debug("📱 Telegram alert sent successfully")
            else:
                logger.warning(f"Telegram API error: {resp.status_code} — {resp.text}")

        except ImportError:
            logger.warning("requests package not installed. Telegram alerts disabled.")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def _play_siren(self):
        """Play siren WAV file using system audio."""
        try:
            wav_file = self.siren_config.get("wav_file", "")
            if not wav_file or not os.path.exists(wav_file):
                logger.debug("Siren WAV not found. Skipping.")
                return

            # Use subprocess to play audio (non-blocking)
            subprocess.Popen(
                ["aplay", wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug("🔊 Siren played")

        except Exception as e:
            logger.error(f"Siren playback failed: {e}")

    def _post_rest(self, event: Dict):
        """POST alert to REST API endpoint."""
        try:
            import requests

            endpoint = self.rest_config.get("endpoint", "")
            if not endpoint:
                return

            # Remove frame from payload (not JSON serializable)
            payload = {k: v for k, v in event.items() if k != "frame"}

            resp = requests.post(
                endpoint,
                json=payload,
                timeout=5,
            )
            logger.debug(f"REST API response: {resp.status_code}")

        except Exception as e:
            logger.error(f"REST API post failed: {e}")

    def stop(self):
        """Stop the alert processing thread."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info(
            f"🔔 Alert Engine stopped. "
            f"Total alerts: {self._alert_count}, Suppressed: {self._alerts_suppressed}"
        )

    @property
    def stats(self) -> Dict:
        """Get alert statistics."""
        return {
            "total_alerts": self._alert_count,
            "suppressed": self._alerts_suppressed,
            "queue_size": self._queue.qsize(),
        }
