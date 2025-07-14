import math
import time
import threading
import queue
import cv2
import numpy as np
from collections import deque
from pathlib import Path
import requests
from tqdm import tqdm
from streamgrid.utils import LOGGER, get_optimal_grid_size
from ultralytics.utils.plotting import Annotator, colors


class StreamGrid:
    """StreamGrid for multi-stream video display with batch processing.

    A real-time video grid display system that can handle multiple video sources
    simultaneously with optional YOLO object detection and video recording capabilities.
    Features adaptive UI scaling, batch processing for efficiency, and comprehensive
    FPS monitoring.

    Attributes:
        sources (list): List of video sources (file paths, camera indices, or URLs).
        max_sources (int): Maximum number of sources to display.
        batch_size (int): Number of frames to process in each batch.
        cols (int): Number of columns in the grid layout.
        rows (int): Number of rows in the grid layout.
        cell_w (int): Width of each cell in pixels.
        cell_h (int): Height of each cell in pixels.
        grid (np.ndarray): The main display grid array.
        frames (dict): Dictionary storing current frames for each source.
        stats (dict): Dictionary storing statistics for each source.
        show_stats (bool): Whether to display statistics overlay.
        running (bool): Flag controlling the main processing loop.
        model: YOLO model for object detection (optional).
        prediction_fps (float): Current prediction frames per second.
        save (bool): Whether to save output video.
        video_writer: OpenCV video writer object.
    """

    def __init__(self, sources=None, model=None, save=True):
        """Initialize StreamGrid with video sources and configuration.

        Args:
            sources (list): List of video sources. Can be:
                - File paths (str): "video.mp4", "stream.avi"
                - Camera indices (int): 0, 1, 2
                - Stream URLs (str): "rtsp://camera_url"
            model (optional): YOLO model instance for object detection.
                If None, only displays video without detection.
            save (bool, optional): Whether to save output video. Defaults to True.
                Output will be saved as "streamgrid_output_{N}_streams.mp4".
        """
        # GitHub repository URLs for default videos
        self.GITHUB_ASSETS_BASE = "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/"
        self.DEFAULT_VIDEOS = [
            "grid_1.mp4",
            "grid_2.mp4",
            "grid_3.mp4",
            "grid_4.mp4"
        ]
        # Handle default sources
        if sources is None:
            LOGGER.warning("⚠️ No sources provided. Downloading default demo videos.")
            sources = self.get_default_videos()

        self.sources = sources
        self.max_sources = self.batch_size = self.active_streams = len(sources)
        self.cols = int(math.ceil(math.sqrt(self.max_sources)))
        self.rows = int(math.ceil(self.max_sources / self.cols))

        self.cell_w, self.cell_h = get_optimal_grid_size(self.max_sources, self.cols)  # Auto cell size based on sources
        self.grid = np.zeros((self.rows * self.cell_h, self.cols * self.cell_w, 3), dtype=np.uint8)
        self.frames = {}
        self.stats = {}
        self.show_stats = True
        self.running = False
        self.lock = threading.Lock()

        # Pre-generate colors for classes
        self.colors = {}
        self.color_idx = 0

        # Batch processing
        self.model = model
        self.frame_queue = queue.Queue(maxsize=max(50, self.max_sources * 4))  # Adaptive queue based on source count

        # FPS tracking - Use deque for better performance
        self.batch_times = deque(maxlen=10)
        self.prediction_fps = 0.0

        # Colors for source labels and detection boxes
        self.colors = [(255, 0, 0), (104, 31, 17), (0, 0, 255),
                       (128, 0, 255), (255, 0, 255), (0, 255, 255),
                       (255, 128, 0), (128, 0, 255)]

        # Video writer support
        self.save = save
        self.video_writer = None
        self.target_fps = 30

        if self.save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                f"streamgrid_output_{self.batch_size}_streams.mp4",
                fourcc, self.target_fps,
                (self.cols * self.cell_w, self.rows * self.cell_h)
            )

        # Add these new attributes
        self.stream_threads = []
        self.auto_shutdown = True  # Control auto-shutdown behavior
        self.shutdown_delay = 3.0  # Seconds to wait after streams end before shutdown

        self.run()

    def get_default_videos(self):
        """Downloads demo videos from GitHub releases if they don't exist locally."""
        demo_dir = Path("assets")
        demo_dir.mkdir(exist_ok=True)

        local_paths = []

        for video_name in self.DEFAULT_VIDEOS:
            local_path = demo_dir / video_name

            if not local_path.exists():
                LOGGER.info(f"ℹ️ Downloading {video_name}...")
                try:
                    url = f"{self.GITHUB_ASSETS_BASE}{video_name}"
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    # Get total file size for progress bar
                    total_size = int(response.headers.get('content-length', 0))

                    with open(local_path, 'wb') as f:
                        with tqdm(
                                desc=video_name,
                                total=total_size,
                                unit='B',
                                unit_scale=True,
                                unit_divisor=1024,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                pbar.update(size)

                except Exception as e:
                    LOGGER.error(f"❌ Failed to download {video_name}: {e}")
                    # Clean up partial download
                    if local_path.exists():
                        local_path.unlink()
                    continue
            else:
                LOGGER.info(f"ℹ️ Found existing {video_name}")

            local_paths.append(str(local_path))

        if not local_paths:
            raise RuntimeError("Unable to download or find any demo videos")

        return local_paths

    def get_color(self, class_idx):
        """Get color for a given class index.

        Args:
            class_idx (int): Index of the class/source.

        Returns:
            tuple: RGB color tuple (r, g, b) with values 0-255.
        """
        return tuple(map(int, self.colors[class_idx % len(self.colors)]))

    def delayed_shutdown(self):
        """Shutdown after a delay to allow for cleanup."""
        self.running = False

    def capture_video(self, source, source_id):
        """Capture video frames from a source in a separate thread.

        Continuously captures frames from the specified source and adds them to
        the processing queue. Handles different source types (files, cameras, streams)
        with appropriate error handling and reconnection logic.

        Args:
            source (str or int): Video source (file path, camera index, or stream URL).
            source_id (int): Unique identifier for this source (0-based index).

        Note:
            This method runs in a separate thread and will continue until
            self.running is set to False or the source is exhausted.
        """
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                LOGGER.error(f"❌ Failed to open source: {source}")
                with self.lock:
                    self.active_streams -= 1
                return

            # Optimize capture settings for CPU processing
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

            frame_count = 0
            no_frame_count = 0
            max_retries = 5  # Number of retries before considering stream dead

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Handle end of video file vs. stream disconnection
                    if isinstance(source, str) and not source.isdigit():
                        no_frame_count += 1
                        if no_frame_count > max_retries:
                            with self.lock:
                                self.active_streams -= 1
                                LOGGER.info(f"ℹ️ Source {source} ended. Active streams: {self.active_streams}")
                            break
                        time.sleep(0.1)
                        continue
                    else:
                        no_frame_count += 1  # Camera/stream disconnected, try to reconnect
                        if no_frame_count > max_retries:
                            with self.lock:
                                self.active_streams -= 1
                                LOGGER.warning(f"Source {source_id} disconnected. Active streams: {self.active_streams}")
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.1)
                        continue

                no_frame_count = 0   # Reset retry counter on successful frame read
                frame_count += 1
                try:
                    # Non-blocking queue insertion
                    self.frame_queue.put((source_id, frame), timeout=0.01)
                except queue.Full:
                    # Drop frame if queue is full to prevent memory buildup
                    pass
                time.sleep(0.05)  # Throttle for CPU efficiency

            cap.release()

            if self.auto_shutdown and self.active_streams <= 0:  # Check if this was the last active stream
                LOGGER.info(f"ℹ️ All streams finished. Initiating shutdown...")
                shutdown_thread = threading.Thread(target=self.delayed_shutdown, daemon=True)
                shutdown_thread.start()

        except Exception as e:
            LOGGER.error(f"❌ Error in capture_video for source {source}: {e}")
            with self.lock:
                self.active_streams -= 1

    def _batch_worker(self):
        """Process frames in batches for efficient YOLO inference.

        Collects frames from multiple sources into batches and processes them
        together for better GPU/CPU utilization. Calculates and maintains
        prediction FPS statistics.

        Note:
            This method runs in a separate thread and handles all ML inference
            and FPS calculation logic.
        """
        batch_frames, batch_ids = [], []

        while self.running:
            # Collect frames up to batch_size
            while len(batch_frames) < self.batch_size:
                try:
                    source_id, frame = self.frame_queue.get(timeout=0.01)
                    batch_frames.append(frame)
                    batch_ids.append(source_id)
                except queue.Empty:
                    break

            if batch_frames:
                try:
                    batch_start = time.time()

                    if self.model:
                        # Run YOLO inference on the batch
                        results = self.model.predict(
                            batch_frames,
                            conf=0.25,
                            verbose=False,
                            device='cpu',
                        )

                        # Update each source with its results
                        for source_id, frame, result in zip(batch_ids, batch_frames, results):
                            self.update_source(source_id, frame, result)
                    else:
                        # No model, just display frames
                        for source_id, frame in zip(batch_ids, batch_frames):
                            self.update_source(source_id, frame)

                    # Calculate prediction FPS
                    batch_time = time.time() - batch_start
                    self.batch_times.append(batch_time)

                    if self.batch_times:
                        avg_batch_time = sum(self.batch_times) / len(self.batch_times)
                        self.prediction_fps = len(batch_frames) / avg_batch_time if avg_batch_time > 0 else 0

                except Exception as e:
                    LOGGER.error(f"❌ Batch processing error: {e}")

                batch_frames.clear()
                batch_ids.clear()

    def update_source(self, source_id, frame, yolo_results=None):
        """Update a source with new frame and detection results.

        Args:
            source_id (int): Index of the source to update.
            frame (np.ndarray): Raw frame from the source.
            yolo_results (optional): YOLO detection results for the frame.
                If None, frame is displayed without detections.
        """
        if source_id >= self.max_sources:
            return

        with self.lock:
            # Resize frame to fit cell dimensions
            resized = cv2.resize(frame, (self.cell_w, self.cell_h))

            # Draw detections if available
            detections = 0
            if yolo_results and yolo_results.boxes is not None:
                detections = len(yolo_results.boxes)
                resized = self.draw_boxes(resized, yolo_results, frame.shape[:2])

            # Store processed frame and statistics
            self.frames[source_id] = resized
            self.stats[source_id] = {'detections': detections, 'time': time.time()}

    def draw_boxes(self, frame, results, orig_shape):
        """Draw YOLO detection boxes and labels on frame.

        Uses Ultralytics' Annotator for consistent styling and proper scaling
        of detection boxes from original frame dimensions to cell dimensions.

        Args:
            frame (np.ndarray): Resized frame to draw on.
            results: YOLO detection results object.
            orig_shape (tuple): Original frame shape (height, width).

        Returns:
            np.ndarray: Frame with detection boxes and labels drawn.
        """
        if not results.boxes:
            return frame

        ann = Annotator(frame)

        # Calculate scaling factors from original to cell dimensions
        scale_x = self.cell_w / orig_shape[1]
        scale_y = self.cell_h / orig_shape[0]

        # Extract detection data
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # Draw each detection
        for box, conf, cls in zip(boxes, confs, classes):
            # Scale coordinates to cell dimensions
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            ann.box_label([x1, y1, x2, y2], label=label, color=colors(int(cls), True))

        return frame

    def run(self):
        """Start the StreamGrid display and processing loop.

        Initializes all worker threads, sets up the display window, and runs
        the main event loop. Handles keyboard input for user interaction.

        Keyboard Controls:
            ESC: Exit the application
            's': Toggle statistics display

        Note:
            This method blocks until the user exits. All cleanup is handled
            automatically through the finally block.
        """
        self.running = True

        # Start capture threads for each source
        for i, source in enumerate(self.sources):
            thread = threading.Thread(target=self.capture_video, args=(source, i), daemon=True)
            thread.start()

        # Start batch processing thread
        batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        batch_thread.start()

        # Initialize display window
        cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
        LOGGER.info("ℹ️ Application running. Press ESC to exit, 's' to toggle stats")

        try:
            while self.running:
                self.update_display()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('s'):  # 's' key
                    self.show_stats = not self.show_stats
                    LOGGER.info(f"ℹ️ Stats display: {'ON' if self.show_stats else 'OFF'}")

        finally:
            self.stop()
            cv2.destroyAllWindows()

    def update_display(self):
        """Update the main grid display with current frames.

        Composites all source frames into a single grid layout, adds source
        labels with adaptive sizing and contrasting colors, and optionally
        displays FPS statistics. Handles placeholder display for inactive sources.
        """
        self.grid.fill(0)

        with self.lock:
            for i in range(self.max_sources):
                # Calculate cell position in grid
                row, col = divmod(i, self.cols)
                y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
                x1, x2 = col * self.cell_w, (col + 1) * self.cell_w

                if i in self.frames:
                    frame = self.frames[i].copy()
                else:
                    # Create placeholder for inactive sources
                    frame = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)

                    # Draw checkerboard pattern
                    for y in range(0, self.cell_h, 20):
                        for x in range(0, self.cell_w, 20):
                            if (x // 20 + y // 20) % 2:
                                frame[y:y + 20, x:x + 20] = 20

                    # Add "WAITING" text
                    text = "WAITING"
                    wait_scale = max(0.4, min(1.0, self.cell_w / 400))
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, wait_scale, 2)
                    cv2.putText(frame, text, ((self.cell_w - w) // 2, (self.cell_h - h) // 2 + h),
                                cv2.FONT_HERSHEY_SIMPLEX, wait_scale, (100, 100, 100), 2)

                if self.show_stats:
                    # Add source label with adaptive sizing
                    info = f"Source #{i}"
                    text_scale = max(1.6, min(0.8, self.cell_w / 400))
                    thickness = max(4, int(text_scale * 3))
                    padding = max(15, int(self.cell_w / 100))

                    # Calculate text dimensions
                    (text_width, text_height), baseline = cv2.getTextSize(
                        info, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness
                    )

                    # Draw colored background for each source
                    bg_color = self.get_color(i)
                    cv2.rectangle(frame, (2, 2),
                                  (2 + text_width + padding * 2, 2 + text_height + baseline + padding * 2),
                                  bg_color, -1)

                    # Use luminance-based contrast for text color
                    r, g, b = bg_color
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)

                    cv2.putText(frame, info, (2 + padding, 4 + padding + text_height),
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, thickness)

                # Place frame in grid
                self.grid[y1:y2, x1:x2] = frame

        # Display FPS statistics if enabled
        if self.show_stats and self.prediction_fps > 0:

            self._draw_fps_overlay()

        # Show the grid
        cv2.imshow("StreamGrid", self.grid)

        # Write frame to video if recording
        if self.save and self.video_writer:
            self.video_writer.write(self.grid)

    def _draw_fps_overlay(self):
        """Draw FPS overlay at the bottom of the grid.

        Creates a centered FPS display with adaptive text sizing and
        high-contrast background for optimal visibility.
        """
        fps_text = f"Prediction FPS: {self.prediction_fps:.1f}"

        # Scale text size based on grid dimensions
        text_scale = max(1.6, min(1.5, (self.cols * self.cell_w) / 800))
        thickness = max(4, int(text_scale * 2))

        # Calculate text dimensions
        (text_w, text_h), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness
        )

        # Position at bottom center
        center_x = (self.cols * self.cell_w - text_w) // 2
        bottom_y = self.rows * self.cell_h - max(10, int(text_scale * 10))

        # Draw background rectangle
        padding = max(20, int(text_scale * 8))
        cv2.rectangle(self.grid,
                      (center_x - padding, bottom_y - text_h - padding),
                      (center_x + text_w + padding, bottom_y + padding),
                      (255, 255, 255), -1)

        # Draw text
        cv2.putText(self.grid, fps_text, (center_x, bottom_y),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (104, 31, 17), thickness)

    def stop(self):
        """Stop all processing and release resources.

        Cleanly shuts down all threads, releases video capture and writer
        resources, and prints completion message if video was saved.
        """
        if not self.running:
            return  # Already stopped

        LOGGER.info("ℹ️ Shutting down StreamGrid...")
        self.running = False

        try:  # Clear the frame queue to prevent blocking
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass

        for i, thread in enumerate(self.stream_threads):  # Wait for threads to finish (with timeout)
            thread.join(timeout=0.5)


        if self.save and self.video_writer:
            self.video_writer.release()
            LOGGER.info(f"✅ Video saved as: streamgrid_output_{self.batch_size}_streams.mp4")

        with self.lock:  # Clear cached frames to free memory
            self.frames.clear()
            self.stats.clear()
