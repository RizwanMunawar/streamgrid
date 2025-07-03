import threading
import time
import cv2
import numpy as np


class VideoStream:
    """Simplified VideoStream for backward compatibility."""

    def __init__(self, source, fps=10, size=(640, 360), stream_id=0):
        self.source = source
        self.fps = fps
        self.size = size
        self.stream_id = stream_id
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.frame_interval = 1.0 / fps
        self.last_time = 0

    def start(self):
        """Start video capture."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except:
            return False

    def _capture_loop(self):
        """Capture loop."""
        while self.running and self.cap and self.cap.isOpened():
            current_time = time.time()

            if current_time - self.last_time < self.frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame = cv2.resize(frame, self.size)
            self.last_time = current_time

    def get_frame(self):
        """Get current frame."""
        return self.frame

    def stop(self):
        """Stop stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
