"""
Video Stream Handler — Threaded RTSP/Webcam capture with reconnection.

Provides non-blocking frame capture from RTSP cameras, HTTP streams,
video files, or local webcams. Drops stale frames for real-time performance.
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Threaded video stream reader with auto-reconnection.

    Captures frames in a background thread so the main inference loop
    never blocks on I/O. Always returns the latest frame (drops stale ones).

    Usage:
        stream = VideoStream("rtsp://...", name="Floor Cam 1")
        stream.start()
        while True:
            ret, frame = stream.read()
            if ret:
                process(frame)
        stream.stop()
    """

    def __init__(
        self,
        source: Union[str, int] = 0,
        name: str = "Camera",
        resolution: Optional[Tuple[int, int]] = None,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 50,
    ):
        """
        Args:
            source: RTSP URL, video file path, or webcam index (0).
            name: Human-readable camera name for logging.
            resolution: Optional (width, height) to set capture resolution.
            reconnect_delay: Seconds to wait before reconnecting on failure.
            max_reconnect_attempts: Max reconnect tries before giving up.
        """
        self.source = source
        self.name = name
        self.resolution = resolution
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._connected = False

    def start(self) -> "VideoStream":
        """Start the capture thread."""
        self._stop_event.clear()
        self._connect()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"VideoStream-{self.name}"
        )
        self._thread.start()
        logger.info(f"📹 [{self.name}] Stream started: {self.source}")
        return self

    def _connect(self) -> bool:
        """Open the video capture connection."""
        try:
            if self._cap is not None:
                self._cap.release()

            # Use CAP_FFMPEG for RTSP streams for better compatibility
            if isinstance(self.source, str) and self.source.startswith("rtsp"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                # Reduce RTSP latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if self.resolution:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            if self._cap.isOpened():
                self._connected = True
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
                logger.info(f"✅ [{self.name}] Connected: {w}x{h} @ {fps:.0f}fps")
                return True
            else:
                self._connected = False
                logger.warning(f"⚠️  [{self.name}] Failed to open: {self.source}")
                return False

        except Exception as e:
            self._connected = False
            logger.error(f"❌ [{self.name}] Connection error: {e}")
            return False

    def _capture_loop(self):
        """Background frame capture loop with reconnection logic."""
        reconnect_count = 0

        while not self._stop_event.is_set():
            if not self._connected:
                # Attempt reconnect
                if reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"❌ [{self.name}] Max reconnect attempts reached. Stopping.")
                    break
                logger.info(f"🔄 [{self.name}] Reconnecting ({reconnect_count + 1}/{self.max_reconnect_attempts})...")
                time.sleep(self.reconnect_delay)
                if self._connect():
                    reconnect_count = 0
                else:
                    reconnect_count += 1
                continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning(f"⚠️  [{self.name}] Frame read failed. Reconnecting...")
                    self._connected = False
                    continue

                # Update the latest frame (thread-safe)
                with self._lock:
                    self._ret = True
                    self._frame = frame
                    self._frame_count += 1

                # Calculate FPS every second
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_time = now

            except Exception as e:
                logger.error(f"❌ [{self.name}] Capture error: {e}")
                self._connected = False

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        Get the latest captured frame.

        Returns:
            Tuple of (success_bool, frame_or_None).
        """
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    @property
    def fps(self) -> float:
        """Current capture FPS."""
        return round(self._fps, 1)

    @property
    def is_connected(self) -> bool:
        """Whether the stream is currently connected."""
        return self._connected

    def stop(self):
        """Stop capture thread and release resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        self._connected = False
        logger.info(f"🛑 [{self.name}] Stream stopped.")

    def __del__(self):
        self.stop()


class MultiStreamManager:
    """
    Manages multiple VideoStream instances for multi-camera setups.

    Usage:
        manager = MultiStreamManager(sources_config)
        manager.start_all()
        frames = manager.read_all()
        manager.stop_all()
    """

    def __init__(self, sources: list):
        """
        Args:
            sources: List of dicts with keys: name, url, enabled
        """
        self.streams: dict[str, VideoStream] = {}
        for src in sources:
            if src.get("enabled", True):
                self.streams[src["name"]] = VideoStream(
                    source=src["url"],
                    name=src["name"],
                )

    def start_all(self):
        """Start all enabled streams."""
        for stream in self.streams.values():
            stream.start()
            time.sleep(0.5)  # Stagger connections

    def read_all(self) -> dict:
        """
        Read latest frames from all streams.

        Returns:
            Dict of {name: (ret, frame)} for connected streams.
        """
        frames = {}
        for name, stream in self.streams.items():
            if stream.is_connected:
                ret, frame = stream.read()
                frames[name] = (ret, frame)
        return frames

    def stop_all(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()

    def get_stream(self, name: str) -> Optional[VideoStream]:
        """Get a specific stream by name."""
        return self.streams.get(name)
