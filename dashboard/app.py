"""
Factory AI Dashboard — Flask-based real-time monitoring UI.

Serves a live annotated video feed via MJPEG, incident log API,
GPU statistics, and a dark-themed monitoring interface.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from functools import wraps
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    Flask-based monitoring dashboard.

    Features:
        - Live MJPEG video stream
        - REST API for incidents, GPU stats, system status
        - Basic auth for access control
        - Thread-safe frame updates

    Usage:
        dashboard = DashboardServer(config)
        dashboard.start()
        # In inference loop:
        dashboard.update_frame(annotated_frame)
        dashboard.update_stats({...})
    """

    def __init__(self, config: dict):
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 5000)
        self.auth_config = config.get("auth", {})
        self._frame = None
        self._frame_lock = threading.Lock()
        self._stats = {}
        self._stats_lock = threading.Lock()
        self._incidents = []
        self._incidents_lock = threading.Lock()
        self._app: Optional[Flask] = None
        self._thread: Optional[threading.Thread] = None

    def _create_app(self) -> Flask:
        """Create and configure the Flask application."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        static_dir = os.path.join(os.path.dirname(__file__), "static")

        app = Flask(
            __name__,
            template_folder=template_dir,
            static_folder=static_dir,
        )
        app.config["SECRET_KEY"] = os.urandom(24).hex()

        # Auth decorator
        def require_auth(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                if not self.auth_config.get("enabled", False):
                    return f(*args, **kwargs)
                auth = request.authorization
                if not auth:
                    return Response(
                        "Authentication required",
                        401,
                        {"WWW-Authenticate": 'Basic realm="Factory AI"'}
                    )
                if (auth.username != self.auth_config.get("username") or
                        auth.password != self.auth_config.get("password")):
                    return Response("Invalid credentials", 403)
                return f(*args, **kwargs)
            return decorated

        # --- Routes ---

        @app.route("/")
        @require_auth
        def index():
            return render_template("index.html")

        @app.route("/video_feed")
        @require_auth
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @app.route("/api/stats")
        @require_auth
        def api_stats():
            with self._stats_lock:
                return jsonify(self._stats)

        @app.route("/api/incidents")
        @require_auth
        def api_incidents():
            with self._incidents_lock:
                limit = request.args.get("limit", 50, type=int)
                return jsonify(self._incidents[-limit:])

        @app.route("/api/system")
        @require_auth
        def api_system():
            return jsonify({
                "status": "running",
                "uptime": time.time(),
                "version": "1.0.0",
            })

        return app

    def _generate_frames(self):
        """Generate MJPEG frames for the video stream."""
        while True:
            with self._frame_lock:
                frame = self._frame

            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() +
                    b"\r\n"
                )
            else:
                # Send a blank frame while waiting
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera...", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                _, buffer = cv2.imencode(".jpg", blank)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() +
                    b"\r\n"
                )

            time.sleep(0.033)  # ~30 FPS

    def update_frame(self, frame: np.ndarray):
        """Thread-safe frame update for the video feed."""
        with self._frame_lock:
            self._frame = frame

    def update_stats(self, stats: dict):
        """Thread-safe stats update."""
        with self._stats_lock:
            self._stats = stats

    def add_incident(self, incident: dict):
        """Thread-safe incident log append."""
        with self._incidents_lock:
            self._incidents.append(incident)
            # Keep last 500 incidents in memory
            if len(self._incidents) > 500:
                self._incidents = self._incidents[-500:]

    def start(self):
        """Start the dashboard server in a background thread."""
        self._app = self._create_app()
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
            name="Dashboard",
        )
        self._thread.start()
        logger.info(f"🖥️  Dashboard running at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the dashboard."""
        logger.info("Dashboard stopped.")
