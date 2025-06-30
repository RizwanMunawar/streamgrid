"""Ultra-optimized StreamGrid - Single file implementation"""

import math
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class VideoStream:
    """Single video stream."""

    def __init__(self, source: Union[str, int], fps: int, cell_size: Tuple[int, int]):
        self.source = source
        self.fps = fps
        self.cell_size = cell_size
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.frame_interval = 1.0 / fps
        self.last_time = 0

    def start(self):
        """Start video capture."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        return True

    def _capture(self):
        """Capture loop."""
        while self.running and self.cap.isOpened():
            current_time = time.time()

            if current_time - self.last_time < self.frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # Stream ended - set frame to black
                self.frame = np.zeros((self.cell_size[1], self.cell_size[0], 3), dtype=np.uint8)
                break

            self.frame = cv2.resize(frame, self.cell_size)
            self.last_time = current_time

    def get_frame(self):
        """Get current frame."""
        return self.frame

    def stop(self):
        """Stop stream."""
        self.running = False
        if self.cap:
            self.cap.release()

    def is_active(self):
        """Check if stream is still active."""
        return self.running and self.thread and self.thread.is_alive()