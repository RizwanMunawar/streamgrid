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

    def __init__(self, sources, fps= 10, model=None, confidence=0.25, batch=4):
        """Initialize StreamGrid."""
        self.sources = sources
        self.num_streams = len(sources)
        self.fps = fps / self.num_streams
        self.model = model
        self.confidence = confidence
        self.batch_size = batch_size = batch

        # Calculate grid layout
        cols = int(math.ceil(math.sqrt(self.num_streams)))
        rows = int(math.ceil(self.num_streams / cols))

        # Auto cell size based on stream count
        if self.num_streams <= 4:
            cell_size = (640, 360)
        elif self.num_streams <= 9:
            cell_size = (480, 270)
        else:
            cell_size = (360, 180)

        self.cell_size = cell_size
        self.grid_shape = (rows, cols)

        # Initialize YOLO processor
        self.yolo_processor = None
        if self.model:
            from .yolo import get_yolo_processor
            self.yolo_processor = get_yolo_processor(self.model, self.confidence, self.batch_size)

        # Initialize
        self.streams = []
        self.grid_image = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)
        self.running = False

        # Stats
        self.show_stats = True
        self.start_time = None
        self.frame_count = 0

        yolo_info = f" + YOLO(batch={batch_size})" if model else ""
        print(f"StreamGrid: {self.num_streams} streams @ {self.fps}fps, {cell_size}{yolo_info}")

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
                # detection_count = stream.get_detection_count()
                info_text = f"STREAM #{i}"
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

        # Calculate total system throughput (all active streams combined)
        active_streams = [s for s in self.streams if s.is_active()]
        total_fps = sum(s.get_actual_fps() for s in active_streams)
        avg_per_stream = total_fps / len(active_streams) if active_streams else 0.0

        # Stats text
        stats = [
            f"Average FPS: {avg_per_stream:.1f}",
            f"Streams: {self.num_streams}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 15
        line_height = 20

        # Calculate the max text width and total height
        text_sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in stats]
        max_width = max(w for w, h in text_sizes)
        total_height = len(stats) * line_height

        # Top-left corner of the rectangle
        x, y = 10, 10
        rect_width = max_width + 2 * padding
        rect_height = total_height + 2 * padding

        # Draw background rectangle
        cv2.rectangle(self.grid_image, (x, y), (x + rect_width, y + rect_height), (0, 0, 0), -1)
        # cv2.rectangle(self.grid_image, (x, y), (x + rect_width, y + rect_height), (255, 255, 255), 1)

        # Draw text
        for i, text in enumerate(stats):
            text_x = x + padding
            text_y = y + padding + (i + 1) * line_height - 5
            cv2.putText(self.grid_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    def stop(self):
        """Stop all streams."""
        self.running = False
        for stream in self.streams:
            stream.stop()

        cv2.destroyAllWindows()
        print("StreamGrid stopped")