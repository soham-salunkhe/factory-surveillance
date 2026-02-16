"""
Device Utilities — GPU/MPS/CPU setup, monitoring, and optimization.

Handles device initialization, cuDNN benchmarking, VRAM monitoring,
and provides GPU statistics for the dashboard and logging.
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
"""

import os
import sys
import time
import platform
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """Container for GPU statistics."""
    device_name: str = "N/A"
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_percent: float = 0.0
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    power_draw_w: float = 0.0
    cuda_available: bool = False
    fp16_supported: bool = False


class CUDAManager:
    """
    Manages CUDA device setup, optimization flags, and GPU monitoring.

    Usage:
        manager = CUDAManager(gpu_id=0, fp16=True, cudnn_benchmark=True)
        device = manager.setup()
        manager.start_monitoring(interval=5)
        ...
        stats = manager.get_stats()
        manager.stop_monitoring()
    """

    def __init__(self, gpu_id: int = 0, fp16: bool = True, cudnn_benchmark: bool = True):
        self.gpu_id = gpu_id
        self.fp16 = fp16
        self.cudnn_benchmark = cudnn_benchmark
        self.device: Optional[torch.device] = None
        self._stats = GPUStats()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False

    def setup(self) -> torch.device:
        """
        Initialize compute device with all optimizations.

        Returns:
            torch.device: Configured CUDA, MPS, or CPU device.
        """
        logger.info("=" * 60)
        logger.info("Device Setup")
        logger.info("=" * 60)

        # Check CUDA availability
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
            torch.cuda.set_device(self.device)
            self._stats.cuda_available = True

            device_name = torch.cuda.get_device_name(self.gpu_id)
            self._stats.device_name = device_name
            logger.info(f"✅ CUDA Device: {device_name}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            logger.info(f"   PyTorch Version: {torch.__version__}")

            capability = torch.cuda.get_device_capability(self.gpu_id)
            self._stats.fp16_supported = capability[0] >= 7
            logger.info(f"   Compute Capability: {capability[0]}.{capability[1]}")
            logger.info(f"   FP16 Supported: {self._stats.fp16_supported}")

            if self.cudnn_benchmark:
                cudnn.benchmark = True
                cudnn.deterministic = False
                logger.info("✅ cuDNN Benchmark: ENABLED")

            vram_total = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 ** 2)
            vram_used = torch.cuda.memory_allocated(self.gpu_id) / (1024 ** 2)
            self._stats.vram_total_mb = vram_total
            logger.info(f"   VRAM Total: {vram_total:.0f} MB")
            logger.info(f"   VRAM Available: {vram_total - vram_used:.0f} MB")

            if self.fp16 and self._stats.fp16_supported:
                logger.info("✅ FP16 Mixed Precision: ENABLED")
            elif self.fp16:
                logger.warning("⚠️  FP16 requested but not supported by GPU. Using FP32.")
                self.fp16 = False

            self._init_nvml()

        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self._stats.cuda_available = False
            self._stats.device_name = f"Apple Silicon ({platform.processor() or 'M-series'})"
            self._stats.fp16_supported = True
            logger.info(f"🍎 MPS Device: {self._stats.device_name}")
            logger.info(f"   PyTorch Version: {torch.__version__}")
            logger.info("   Using Metal Performance Shaders backend")
            self.fp16 = False  # MPS handles precision internally

        else:
            logger.warning("⚠️  No GPU available. Falling back to CPU.")
            self.device = torch.device("cpu")
            self._stats.cuda_available = False
            self.fp16 = False

        logger.info("=" * 60)
        return self.device

    def _init_nvml(self):
        """Initialize NVIDIA Management Library for GPU monitoring."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("✅ NVML Monitoring: ENABLED")
        except ImportError:
            logger.warning("⚠️  pynvml not installed. GPU monitoring limited.")
            logger.warning("   Install with: pip install pynvml")
        except Exception as e:
            logger.warning(f"⚠️  NVML init failed: {e}")

    def get_stats(self) -> GPUStats:
        """
        Get current GPU statistics.

        Returns:
            GPUStats: Current VRAM, utilization, temperature, power stats.
        """
        if self.device is not None and self.device.type == "mps":
            # MPS doesn't expose detailed stats; return basic info
            return self._stats

        if not self._stats.cuda_available:
            return self._stats

        # PyTorch memory stats (always available)
        self._stats.vram_used_mb = torch.cuda.memory_allocated(self.gpu_id) / (1024 ** 2)
        self._stats.vram_total_mb = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 ** 2)
        self._stats.vram_percent = (self._stats.vram_used_mb / self._stats.vram_total_mb) * 100

        # NVML detailed stats (if available)
        if self._nvml_initialized:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._stats.gpu_utilization = util.gpu

                self._stats.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                self._stats.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._stats.vram_used_mb = mem_info.used / (1024 ** 2)
                self._stats.vram_total_mb = mem_info.total / (1024 ** 2)
                self._stats.vram_percent = (mem_info.used / mem_info.total) * 100

            except Exception as e:
                logger.debug(f"NVML stats read error: {e}")

        return self._stats

    def start_monitoring(self, interval: float = 5.0):
        """
        Start background GPU monitoring thread.

        Args:
            interval: Seconds between stat updates.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("GPU monitoring already running.")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="GPUMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"📊 GPU Monitor started (interval={interval}s)")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            stats = self.get_stats()
            logger.debug(
                f"GPU: {stats.vram_used_mb:.0f}/{stats.vram_total_mb:.0f}MB "
                f"({stats.vram_percent:.1f}%) | Util: {stats.gpu_utilization}% | "
                f"Temp: {stats.temperature}°C | Power: {stats.power_draw_w:.1f}W"
            )
            self._stop_event.wait(interval)

    def stop_monitoring(self):
        """Stop the GPU monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("GPU Monitor stopped.")

    def optimize_memory(self):
        """Free cached GPU memory."""
        if self._stats.cuda_available:
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared.")

    def get_device_string(self) -> str:
        """Get device string for Ultralytics model loading."""
        if self._stats.cuda_available:
            return str(self.gpu_id)
        if self.device is not None and self.device.type == "mps":
            return "mps"
        return "cpu"

    def __del__(self):
        """Cleanup NVML on destruction."""
        self.stop_monitoring()
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
