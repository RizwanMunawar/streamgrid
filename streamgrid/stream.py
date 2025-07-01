"""Ultra-optimized StreamGrid - Single file implementation"""

import math
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class VideoStream:
    """Single video stream with optional YOLO."""

    def __init__(self, source, fps, cell_size, stream_id=0, yolo_processor=None):
        self.source = source
        self.fps = fps
        self.cell_size = cell_size
        self.stream_id = stream_id
        self.yolo_processor = yolo_processor

        self.cap = None
        self.current_frame = None  # Always current frame for display
        self.yolo_frame = None     # Latest YOLO processed frame
        self.running = False
        self.thread = None
        self.frame_interval = 1.0 / fps
        self.last_time = 0

        # YOLO processing
        self.detection_count = 0
        self.last_yolo_time = 0
        self.yolo_interval = 0.2  # Process for YOLO every 200ms (5fps max)
        # FPS tracking
        self.frame_times = []
        self.fps_window_size = 30  # Calculate FPS over last 30 frames

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
        """Capture loop with separate YOLO processing."""
        while self.running and self.cap.isOpened():
            current_time = time.time()

            # Maintain stable frame rate for display
            if current_time - self.last_time < self.frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # Stream ended - set frame to black
                self.current_frame = np.zeros((self.cell_size[1], self.cell_size[0], 3), dtype=np.uint8)
                break

            # Always update current frame for stable display
            resized_frame = cv2.resize(frame, self.cell_size)
            self.current_frame = resized_frame
            self.last_time = current_time

            # Track FPS
            self.frame_times.append(current_time)
            if len(self.frame_times) > self.fps_window_size:
                self.frame_times.pop(0)

            # Send to YOLO processing (throttled and non-blocking)
            if (self.yolo_processor and
                current_time - self.last_yolo_time >= self.yolo_interval):

                if self.yolo_processor.add_frame(self.stream_id, resized_frame.copy()):
                    self.last_yolo_time = current_time

            # Check for latest YOLO results (non-blocking)
            if self.yolo_processor:
                result = self.yolo_processor.get_result(self.stream_id)
                if result:
                    # Only update if result is recent (within 1 second)
                    if current_time - result.get('timestamp', 0) < 1.0:
                        self.yolo_frame = result['frame']
                        self.detection_count = result['detections']

    def get_frame(self):
        """Get current frame for display - stable and natural."""
        if self.yolo_processor and self.yolo_frame is not None:
            return self.yolo_frame
        return self.current_frame

    def get_detection_count(self) -> int:
        """Get number of detections in current frame."""
        return self.detection_count

    def stop(self):
        """Stop stream."""
        self.running = False
        if self.cap:
            self.cap.release()

    def is_active(self):
        """Check if stream is still active."""
        return self.running and self.thread and self.thread.is_alive()

    def get_actual_fps(self) -> float:
        """Get actual FPS of this stream."""
        if len(self.frame_times) < 2:
            return 0.0
        time_diff = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0.0