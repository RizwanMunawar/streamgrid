"""Core utilities for StreamGrid - logging, configuration, and optimization."""

import logging
import math
import time
import csv
from pathlib import Path
from collections import deque


def setup_logger(name="streamgrid", level=logging.INFO):
    """Setup optimized logger for StreamGrid.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with clean format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_optimal_grid_layout(source_count):
    """Calculate optimal grid layout for given number of sources.

    Args:
        source_count: Number of video sources

    Returns:
        tuple: (columns, rows, cell_width, cell_height)
    """
    # Get screen dimensions
    screen_width, screen_height = get_screen_size()

    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(source_count)))
    rows = int(math.ceil(source_count / cols))

    # Calculate cell dimensions with margins
    available_width = int(screen_width * 0.95)  # 5% margin
    available_height = int(screen_height * 0.90)  # 10% margin

    cell_width = available_width // cols
    cell_height = available_height // rows

    # Maintain 16:9 aspect ratio
    target_ratio = 16 / 9
    current_ratio = cell_width / cell_height

    if current_ratio > target_ratio:
        cell_width = int(cell_height * target_ratio)
    else:
        cell_height = int(cell_width / target_ratio)

    # Ensure minimum dimensions and even numbers (for video encoding)
    cell_width = max(320, cell_width - (cell_width % 2))
    cell_height = max(180, cell_height - (cell_height % 2))

    return cols, rows, cell_width, cell_height


def get_screen_size():
    """Get screen dimensions with fallback.

    Returns:
        tuple: (width, height) in pixels
    """
    try:
        from screeninfo import get_monitors
        monitor = get_monitors()[0]
        return monitor.width, monitor.height
    except (ImportError, IndexError):
        # Fallback to common resolutions
        return 1920, 1080


def optimize_for_performance(source_count, device="cpu"):
    """Get optimized settings based on source count and device.

    Args:
        source_count: Number of video sources
        device: Processing device ("cpu" or "cuda")

    Returns:
        dict: Optimized configuration settings
    """
    # Base configuration
    config = {
        "target_fps": 15,
        "batch_size": 4,
        "buffer_size": 2,
        "detection_conf": 0.25,
        "max_det": 100,
    }

    # Adjust based on source count
    if source_count <= 2:
        config.update({
            "target_fps": 20,
            "batch_size": 2,
        })
    elif source_count <= 4:
        config.update({
            "target_fps": 15,
            "batch_size": 4,
        })
    elif source_count <= 8:
        config.update({
            "target_fps": 10,
            "batch_size": 8,
            "detection_conf": 0.3,  # Higher confidence for performance
        })
    else:
        config.update({
            "target_fps": 5,
            "batch_size": min(16, source_count),
            "detection_conf": 0.4,
            "max_det": 50,  # Limit detections for performance
        })

    # GPU optimizations
    if device.startswith("cuda"):
        config["target_fps"] = int(config["target_fps"] * 1.5)
        config["batch_size"] = min(32, config["batch_size"] * 2)

    return config


class PerformanceTracker:
    """Track and calculate performance metrics efficiently."""

    def __init__(self, window_size=30):
        """Initialize performance tracker.

        Args:
            window_size: Number of samples to keep for averaging
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.frame_counts = deque(maxlen=window_size)
        self.last_update = time.time()

    def update(self, frame_count=1):
        """Update performance metrics.

        Args:
            frame_count: Number of frames processed
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        self.frame_counts.append(frame_count)
        self.last_update = current_time

    def get_fps(self):
        """Calculate current FPS.

        Returns:
            float: Current frames per second
        """
        if len(self.timestamps) < 2:
            return 0.0

        time_span = self.timestamps[-1] - self.timestamps[0]
        total_frames = sum(self.frame_counts)

        return total_frames / time_span if time_span > 0 else 0.0

    def get_avg_fps(self):
        """Get average FPS over the tracking window.

        Returns:
            float: Average FPS
        """
        return self.get_fps()  # Same calculation for now

    def reset(self):
        """Reset performance tracking."""
        self.timestamps.clear()
        self.frame_counts.clear()


class AnalyticsLogger:
    """Lightweight analytics logger for StreamGrid."""

    def __init__(self, output_file="streamgrid_analytics.csv", enabled=True):
        """Initialize analytics logger.

        Args:
            output_file: Output CSV file path
            enabled: Whether to enable logging
        """
        self.enabled = enabled
        self.output_file = Path(output_file)
        self.start_time = time.time()

        if self.enabled:
            self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "stream_id", "detections",
                    "fps", "processing_time"
                ])

            print(f"📊 Analytics enabled: {self.output_file}")
        except Exception as e:
            print(f"⚠️ Analytics initialization failed: {e}")
            self.enabled = False

    def log(self, stream_id, detections=0, fps=0.0, processing_time=0.0):
        """Log analytics data.

        Args:
            stream_id: ID of the stream
            detections: Number of detections
            fps: Current FPS
            processing_time: Processing time in seconds
        """
        if not self.enabled:
            return

        try:
            with open(self.output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%H:%M:%S"),
                    stream_id,
                    detections,
                    round(fps, 2),
                    round(processing_time, 4)
                ])
        except Exception as e:
            print(f"⚠️ Analytics logging failed: {e}")

    def get_summary(self):
        """Get analytics summary.

        Returns:
            dict: Summary statistics
        """
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "output_file": str(self.output_file),
            "enabled": self.enabled
        }


class ConfigManager:
    """Manage StreamGrid configuration efficiently."""

    DEFAULT_CONFIG = {
        "target_fps": 15,
        "save_output": False,
        "show_stats": True,
        "analytics": False,
        "device": "cpu",
        "confidence": 0.25,
        "max_detections": 100,
        "auto_optimize": True,
    }

    def __init__(self, **kwargs):
        """Initialize configuration manager.

        Args:
            **kwargs: Configuration overrides
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        # Auto-optimize if enabled
        if self.config["auto_optimize"]:
            self._auto_optimize()

    def get(self, key, default=None):
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def update(self, updates):
        """Update multiple configuration values.

        Args:
            updates: Dictionary of updates
        """
        self.config.update(updates)

    def _auto_optimize(self):
        """Apply automatic optimizations based on system capabilities."""
        try:
            import torch
            if torch.cuda.is_available() and self.config["device"] == "auto":
                self.config["device"] = "cuda"
                self.config["target_fps"] = int(self.config["target_fps"] * 1.5)
        except ImportError:
            pass


# Create global logger instance
LOGGER = setup_logger()
