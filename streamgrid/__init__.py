"""
StreamGrid - Ultra-fast multi-stream video display with YOLO detection.

A professional, optimized solution for displaying multiple video streams
in a grid layout with optional real-time object detection.
"""

__version__ = "1.0.0"
__author__ = "Muhammad Rizwan Munawar"
__email__ = "rizwanmunawar@ultralytics.com"
__license__ = "MIT"

# Main imports
from .grid import StreamGrid
from .stream import VideoStream

# Expose main class for easy import
__all__ = ["StreamGrid", "VideoStream"]

# Package metadata
__package_info__ = {
    "name": "streamgrid",
    "version": __version__,
    "description": "Ultra-fast multi-stream video display",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.8",
    "dependencies": ["opencv-python>=4.5.0", "numpy>=1.21.0"],
}