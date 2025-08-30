# StreamGrid - Display and Processing

import math
import time
import threading
import cv2
import numpy as np
from collections import deque
from .stream import StreamManager
from .plotting import StreamAnnotator
from .utils import LOGGER, get_optimal_grid_size
from .analytics import StreamAnalytics


class StreamGrid:
    """Ultra-fast multi-stream video display with object detection."""

    def __init__(
        self, sources=None, model=None, save=True, device="cpu", analytics=False
    ):
        # Initialize components
        self.stream_manager = StreamManager(sources)
        self.model = model
        self.device = device
        self.analytics = StreamAnalytics() if analytics else None

        # Grid layout
        self.max_sources = len(self.stream_manager.sources)
        self.cols = int(math.ceil(math.sqrt(self.max_sources)))
        self.rows = int(math.ceil(self.max_sources / self.cols))
        self.cell_w, self.cell_h = get_optimal_grid_size(self.max_sources, self.cols)

        # Initialize annotator
        self.plotter = StreamAnnotator(self.cell_w, self.cell_h, self.max_sources)

        # Display state
        self.grid = np.zeros(
            (self.rows * self.cell_h, self.cols * self.cell_w, 3), dtype=np.uint8
        )
        self.frames = {}
        self.show_stats = True
        self.running = False
        self.lock = threading.Lock()

        # Performance tracking
        self.batch_times = deque(maxlen=10)
        self.prediction_fps = 0.0

        # Video saving
        self.video_writer = self.setup_video_writer() if save else None

        self.run()

    def setup_video_writer(self):
        """Setup video writer for saving output."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(
            f"streamgrid_output_{self.max_sources}_streams.mp4",
            fourcc,
            30,
            (self.cols * self.cell_w, self.rows * self.cell_h),
        )

    def process_batch(self):
        """Process frames in batches with consistent timing."""
        batch_interval = 0.033  # ~30 FPS processing
        last_batch_time = time.time()

        while self.running:
            current_time = time.time()

            # Maintain consistent batch processing rate
            if current_time - last_batch_time < batch_interval:
                time.sleep(0.001)
                continue

            frame_data = self.stream_manager.get_frames(self.max_sources)
            if not frame_data:
                time.sleep(0.001)  # Small sleep if no frames
                continue

            batch_start = time.time()
            frames = [data[1] for data in frame_data]
            ids = [data[0] for data in frame_data]

            # Run inference if model available
            if self.model:
                results = self.model.predict(
                    frames,
                    conf=0.25,
                    verbose=False,
                    device=self.device,
                    batch=16,
                )
                for source_id, frame, result in zip(ids, frames, results):
                    self.update_source(source_id, frame, result)
            else:
                for source_id, frame in zip(ids, frames):
                    self.update_source(source_id, frame)

            # Update performance metrics
            self.update_fps(len(frames), time.time() - batch_start)
            last_batch_time = current_time

    def update_fps(self, frame_count, batch_time):
        """Update FPS calculations."""
        self.batch_times.append(batch_time)
        if self.batch_times:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            self.prediction_fps = frame_count / avg_time if avg_time > 0 else 0

    def update_source(self, source_id, frame, results=None):
        """Update display with processed frame."""
        if source_id >= self.max_sources:
            return

        with self.lock:
            # Resize frame to cell dimensions
            resized = cv2.resize(frame, (self.cell_w, self.cell_h))

            # Draw detections if available
            detections = 0
            if results and results.boxes is not None:
                detections = len(results.boxes)
                resized = self.plotter.draw_detections(
                    resized, results, frame.shape[:2]
                )

            # Add source label
            resized = self.plotter.draw_source_label(
                resized, source_id, self.show_stats
            )

            # Store processed frame
            self.frames[source_id] = resized

            # Log analytics
            if self.analytics:
                self.analytics.log(source_id, detections, self.prediction_fps)

    def update_display(self):
        """Update the main grid display."""
        self.grid.fill(0)

        with self.lock:
            # Place each frame in grid
            for i in range(self.max_sources):
                row, col = divmod(i, self.cols)
                y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
                x1, x2 = col * self.cell_w, (col + 1) * self.cell_w

                # Use processed frame or placeholder
                frame = self.frames.get(i, self.plotter.create_placeholder())
                self.grid[y1:y2, x1:x2] = frame

        # Add FPS overlay
        if self.show_stats:
            self.grid = self.plotter.draw_fps_overlay(
                self.grid,
                self.prediction_fps,
                self.cols * self.cell_w,
                self.rows * self.cell_h,
            )

        # Display and save
        cv2.imshow("StreamGrid", self.grid)
        if self.video_writer:
            self.video_writer.write(self.grid)

    def run(self):
        """Main execution loop."""
        self.running = True
        self.stream_manager.start()

        # Start batch processing
        threading.Thread(target=self.process_batch, daemon=True).start()

        cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
        LOGGER.info("ℹ️ Running. Press ESC to exit, 's' to toggle stats")

        try:
            while self.running:
                has_frame = self.update_display()
                if not has_frame:
                    LOGGER.info("ℹ️ All videos ended. Exiting.")
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord("s"):
                    self.show_stats = not self.show_stats
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self):
        """Clean shutdown."""
        if not self.running:
            return

        LOGGER.info("ℹ️ Shutting down...")
        self.running = False
        self.stream_manager.stop()

        if self.analytics:
            self.analytics.summary()
        if self.video_writer:
            self.video_writer.release()
            LOGGER.info(
                f"✅ Video saved: streamgrid_output_{self.max_sources}_streams.mp4"
            )

        with self.lock:
            self.frames.clear()
