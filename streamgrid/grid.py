"""Ultra-optimized StreamGrid - Single file implementation"""

import math
import threading
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from .stream import VideoStream


class StreamGrid:
    """Ultra-fast multi-stream video display."""

    def __init__(self, sources: List[Union[str, int]], fps: int = 10):
        """Initialize StreamGrid."""
        self.sources = sources
        self.num_streams = len(sources)
        self.fps = fps

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

        # Initialize
        self.streams = []
        self.grid_image = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)
        self.running = False

        print(f"StreamGrid: {self.num_streams} streams @ {fps}fps, {cell_size}")

    def run(self):
        """Start and run StreamGrid."""
        self.running = True

        # Start streams
        for i, source in enumerate(self.sources):
            stream = VideoStream(source, self.fps, self.cell_size)
            self.streams.append(stream)
            stream.start()

        print(f"Started {len(self.streams)} streams. Press ESC to exit.")

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

            self.grid_image[row*h:(row+1)*h, col*w:(col+1)*w] = frame

        cv2.imshow("StreamGrid", self.grid_image)

    def stop(self):
        """Stop all streams."""
        self.running = False
        for stream in self.streams:
            stream.stop()
        cv2.destroyAllWindows()
        print("StreamGrid stopped")