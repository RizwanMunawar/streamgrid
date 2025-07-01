"""Ultra-optimized StreamGrid - Single file implementation"""

import math
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from .stream import VideoStream
from .yolo import stop_yolo


class StreamGrid:
    """Ultra-fast multi-stream video display with optional YOLO."""

    def __init__(self, sources: List[Union[str, int]], fps: int = 10, model=None, confidence: float = 0.25):
        """Initialize StreamGrid.

        Args:
            sources: List of video sources
            fps: Target FPS
            model: YOLO model instance (optional)
            confidence: YOLO confidence threshold
        """
        self.sources = sources
        self.num_streams = len(sources)
        self.fps = fps
        self.model = model
        self.confidence = confidence

        # Calculate grid layout
        cols = int(math.ceil(math.sqrt(self.num_streams)))
        rows = int(math.ceil(self.num_streams / cols))

        # Auto cell size based on stream count
        if self.num_streams <= 4:
            cell_size = (1280, 720)
        elif self.num_streams <= 9:
            cell_size = (640, 360)
        else:
            cell_size = (480, 270)

        self.cell_size = cell_size
        self.grid_shape = (rows, cols)

        # Initialize YOLO processor
        self.yolo_processor = None
        if self.model:
            from .yolo import get_yolo_processor
            self.yolo_processor = get_yolo_processor(self.model, self.confidence)

        # Initialize
        self.streams = []
        self.grid_image = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)
        self.running = False

        # Stats
        self.show_stats = True
        self.start_time = None
        self.frame_count = 0

        yolo_info = f" + YOLO({model.__class__.__name__ if model else 'None'})" if model else ""
        print(f"StreamGrid: {self.num_streams} streams @ {fps}fps, {cell_size}{yolo_info}")

    def run(self):
        """Start and run StreamGrid."""
        self.running = True
        self.start_time = time.time()

        # Start streams
        for i, source in enumerate(self.sources):
            stream = VideoStream(source, self.fps, self.cell_size,
                               stream_id=i, yolo_processor=self.yolo_processor)
            self.streams.append(stream)
            stream.start()

        controls = "Press ESC to exit"
        if self.model:
            controls += ", 's' for stats"
        print(f"Started {len(self.streams)} streams. {controls}")

        # Main loop
        try:
            while self.running:
                self._update_display()

                # Check if all streams are finished
                active_streams = sum(1 for stream in self.streams if stream.is_active())
                if active_streams == 0:
                    print("All streams finished. Exiting...")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s') and self.model:  # Toggle stats
                    self.show_stats = not self.show_stats

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _update_display(self):
        """Update display with current frames."""
        rows, cols = self.grid_shape
        h, w = self.cell_size[1], self.cell_size[0]

        for i, stream in enumerate(self.streams):
            row = i // cols
            col = i % cols

            frame = stream.get_frame()
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SIGNAL", (w//4, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            # Ensure frame is the right size (safety check)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))

            # Add detection count if YOLO enabled
            if self.model and self.show_stats:
                detection_count = stream.get_detection_count()
                info_text = f"S{i}: {detection_count} objs"
                cv2.putText(frame, info_text, (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            self.grid_image[row*h:(row+1)*h, col*w:(col+1)*w] = frame

        # Add global stats
        if self.model and self.show_stats:
            self._add_stats_overlay()

        cv2.imshow("StreamGrid", self.grid_image)
        self.frame_count += 1

    def _add_stats_overlay(self):
        """Add stats overlay."""
        if not self.start_time:
            return

        elapsed = time.time() - self.start_time
        display_fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Stats background
        cv2.rectangle(self.grid_image, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(self.grid_image, (10, 10), (250, 80), (255, 255, 255), 1)

        # Stats text
        stats = [
            f"Display FPS: {display_fps:.1f}",
            f"Streams: {self.num_streams}",
            f"YOLO: {'ON' if self.model else 'OFF'}"
        ]

        for i, text in enumerate(stats):
            cv2.putText(self.grid_image, text, (15, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def stop(self):
        """Stop all streams."""
        self.running = False
        for stream in self.streams:
            stream.stop()

        # Stop global YOLO detector
        if self.enable_yolo:
            stop_yolo()

        cv2.destroyAllWindows()
        print("StreamGrid stopped")