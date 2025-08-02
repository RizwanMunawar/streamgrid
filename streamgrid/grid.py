"""Main StreamGrid controller - clean, optimized, and scalable."""

import time
import threading
import cv2
import numpy as np

from .stream import StreamManager
from .plotting import StreamAnnotator, FPSOverlay
from .utils import (
    LOGGER, get_optimal_grid_layout, optimize_for_performance,
    PerformanceTracker, AnalyticsLogger, ConfigManager
)


class StreamGrid:
    """Optimized StreamGrid for multi-stream video display with object detection.

    Clean architecture with separated concerns:
    - Stream handling in stream.py
    - Visualization in plotting.py  
    - Utilities in utils.py
    """

    def __init__(self, sources=None, model=None, **kwargs):
        """Initialize StreamGrid with clean configuration.

        Args:
            sources: List of video sources (files, cameras, URLs)
            model: YOLO model for object detection (optional)
            **kwargs: Configuration options
        """
        # Handle default sources
        self.sources = self._validate_sources(sources)
        self.model = model

        # Initialize configuration
        self.config = ConfigManager(**kwargs)

        # Get optimized settings
        performance_config = optimize_for_performance(
            len(self.sources),
            self.config.get("device", "cpu")
        )
        self.config.update(performance_config)

        # Initialize grid layout
        self.cols, self.rows, self.cell_w, self.cell_h = get_optimal_grid_layout(len(self.sources))
        self.grid_width = self.cols * self.cell_w
        self.grid_height = self.rows * self.cell_h

        # Initialize components
        self.stream_manager = StreamManager(self.sources, self.config.get("target_fps"))
        self.annotator = StreamAnnotator(self.cell_w, self.cell_h)
        self.fps_overlay = FPSOverlay(self.grid_width, self.grid_height)
        self.performance_tracker = PerformanceTracker()

        # Initialize analytics
        self.analytics = None
        if self.config.get("analytics"):
            self.analytics = AnalyticsLogger()

        # Initialize video writer
        self.video_writer = None
        if self.config.get("save_output"):
            self._setup_video_writer()

        # Control variables
        self.running = False
        self.processing_thread = None
        self.display_thread = None

        # Display grid
        self.grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        self.current_frames = {}
        self.lock = threading.Lock()

        LOGGER.info(f"🚀 StreamGrid initialized: {len(self.sources)} sources, {self.cols}x{self.rows} grid")

        # Start the system
        self.start()

    def _validate_sources(self, sources):
        """Validate and prepare sources list."""
        if sources is None:
            LOGGER.warning("⚠️ No sources provided. Using default webcam.")
            return [0]  # Default to webcam

        if not isinstance(sources, list):
            sources = [sources]

        if len(sources) == 0:
            raise ValueError("At least one source must be provided")

        if len(sources) > 16:
            LOGGER.warning("⚠️ Too many sources. Limiting to 16 for performance.")
            sources = sources[:16]

        return sources

    def _setup_video_writer(self):
        """Setup video writer for output recording."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = f"streamgrid_output_{len(self.sources)}_streams.mp4"

            self.video_writer = cv2.VideoWriter(
                output_file,
                fourcc,
                self.config.get("target_fps", 15),
                (self.grid_width, self.grid_height)
            )

            LOGGER.info(f"📹 Video recording enabled: {output_file}")
        except Exception as e:
            LOGGER.error(f"❌ Failed to setup video writer: {e}")
            self.config.set("save_output", False)

    def start(self):
        """Start the StreamGrid processing and display."""
        if self.running:
            return

        self.running = True

        # Start stream manager
        self.stream_manager.start_all()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # Start display thread  
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

        # Run main control loop
        self._main_loop()

    def _processing_loop(self):
        """Main processing loop - handles inference and frame updates."""
        while self.running:
            start_time = time.time()

            # Get frames from all streams
            frames = self.stream_manager.get_frames()

            if not frames:
                time.sleep(0.01)
                continue

            # Process frames with model if available
            if self.model and frames:
                self._process_with_model(frames, start_time)
            else:
                self._process_without_model(frames, start_time)

            # Update performance tracking
            processing_time = time.time() - start_time
            self.performance_tracker.update(len(frames))

            # Small delay to prevent CPU overload
            time.sleep(0.001)

    def _process_with_model(self, frames, start_time):
        """Process frames with YOLO model."""
        try:
            # Prepare batch
            frame_list = list(frames.values())
            stream_ids = list(frames.keys())

            # Run inference
            results = self.model.predict(
                frame_list,
                conf=self.config.get("confidence", 0.25),
                max_det=self.config.get("max_detections", 100),
                verbose=False,
                device=self.config.get("device", "cpu")
            )

            # Update frames with results
            processing_time = time.time() - start_time

            with self.lock:
                for stream_id, frame, result in zip(stream_ids, frame_list, results):
                    # Resize frame
                    resized_frame = cv2.resize(frame, (self.cell_w, self.cell_h))

                    # Annotate detections
                    annotated_frame = self.annotator.annotate_detections(
                        resized_frame, result, frame.shape[:2]
                    )

                    # Add source label
                    final_frame = self.annotator.add_source_label(
                        annotated_frame, stream_id, self.config.get("show_stats", True)
                    )

                    self.current_frames[stream_id] = final_frame

                    # Log analytics
                    if self.analytics:
                        detections = len(result.boxes) if result.boxes else 0
                        self.analytics.log(
                            stream_id, detections,
                            self.performance_tracker.get_fps(), processing_time
                        )

        except Exception as e:
            LOGGER.error(f"❌ Model processing error: {e}")
            self._process_without_model(frames, start_time)

    def _process_without_model(self, frames, start_time):
        """Process frames without model (display only)."""
        processing_time = time.time() - start_time

        with self.lock:
            for stream_id, frame in frames.items():
                # Resize frame
                resized_frame = cv2.resize(frame, (self.cell_w, self.cell_h))

                # Add source label
                final_frame = self.annotator.add_source_label(
                    resized_frame, stream_id, self.config.get("show_stats", True)
                )

                self.current_frames[stream_id] = final_frame

                # Log analytics
                if self.analytics:
                    self.analytics.log(
                        stream_id, 0,
                        self.performance_tracker.get_fps(), processing_time
                    )

    def _display_loop(self):
        """Display loop - updates the grid and shows it."""
        while self.running:
            self._update_grid()

            # Control display rate
            time.sleep(1 / 30)  # 30 FPS display

    def _update_grid(self):
        """Update the display grid with current frames."""
        # Clear grid
        self.grid.fill(0)

        with self.lock:
            # Place frames in grid
            for i in range(len(self.sources)):
                row, col = divmod(i, self.cols)
                y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
                x1, x2 = col * self.cell_w, (col + 1) * self.cell_w

                if i in self.current_frames:
                    self.grid[y1:y2, x1:x2] = self.current_frames[i]
                else:
                    # Create placeholder
                    placeholder = self.annotator.create_placeholder(i)
                    self.grid[y1:y2, x1:x2] = placeholder

        # Add FPS overlay
        if self.config.get("show_stats", True):
            current_fps = self.performance_tracker.get_fps()
            self.grid = self.fps_overlay.draw_fps(self.grid, current_fps)

            # Add stream stats
            stream_stats = self.stream_manager.get_stream_stats()
            self.grid = self.fps_overlay.draw_stream_stats(self.grid, stream_stats)

        # Show grid
        cv2.imshow("StreamGrid", self.grid)

        # Save to video if enabled
        if self.video_writer:
            self.video_writer.write(self.grid)

    def _main_loop(self):
        """Main control loop - handles user input and system control."""
        cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
        LOGGER.info("ℹ️ StreamGrid running. Press 'q' to quit, 's' to toggle stats")

        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Toggle stats
                    current = self.config.get("show_stats", True)
                    self.config.set("show_stats", not current)
                    LOGGER.info(f"ℹ️ Stats display: {'ON' if not current else 'OFF'}")
                elif key == ord('r'):  # Reset performance tracking
                    self.performance_tracker.reset()
                    LOGGER.info("ℹ️ Performance tracking reset")

                # Check if all streams are done
                if self.stream_manager.get_active_count() == 0:
                    LOGGER.info("ℹ️ All streams finished")
                    break

        except KeyboardInterrupt:
            LOGGER.info("ℹ️ Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop StreamGrid and cleanup resources."""
        if not self.running:
            return

        LOGGER.info("🛑 Stopping StreamGrid...")
        self.running = False

        # Stop stream manager
        self.stream_manager.stop_all()

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)

        # Cleanup video writer
        if self.video_writer:
            self.video_writer.release()
            LOGGER.info("✅ Video saved successfully")

        # Analytics summary
        if self.analytics:
            summary = self.analytics.get_summary()
            LOGGER.info(f"📊 Session: {summary['uptime_seconds']:.1f}s, Data: {summary['output_file']}")

        # Cleanup OpenCV
        cv2.destroyAllWindows()

        LOGGER.info("✅ StreamGrid stopped successfully")
