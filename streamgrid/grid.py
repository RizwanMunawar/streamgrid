"""Multi-stream inference and rendering with independently paced stages."""

import math
import threading
import time
from collections import deque

import cv2
import numpy as np

from .analytics import StreamAnalytics
from .plotting import StreamAnnotator
from .stream import StreamManager
from .utils import LOGGER, get_optimal_grid_size


class StreamGrid:
    """Display multiple independent video streams and optionally run YOLO inference."""

    def __init__(
        self, sources=None, model=None, save=False, device="cpu", analytics=False,
        buffer_mode="latest", queue_size=8, capture_fps=None, inference_fps=30,
        display_fps=30, batch_size=0, conf=0.25, imgsz=640, show=True,
    ):
        if inference_fps <= 0 or display_fps <= 0:
            raise ValueError("inference_fps and display_fps must be positive")
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        self.stream_manager = StreamManager(
            sources, buffer_mode=buffer_mode, queue_size=queue_size,
            capture_fps=capture_fps
        )
        self.model = model
        self.device = device
        self.analytics = StreamAnalytics() if analytics else None
        self.inference_fps = inference_fps
        self.display_fps = display_fps
        self.batch_size = batch_size
        self.conf = conf
        self.imgsz = imgsz
        self.show = show
        self.max_sources = len(self.stream_manager.sources)
        self.cols = math.ceil(math.sqrt(self.max_sources))
        self.rows = math.ceil(self.max_sources / self.cols)
        self.cell_w, self.cell_h = get_optimal_grid_size(self.max_sources, self.cols)
        self.plotter = StreamAnnotator(self.cell_w, self.cell_h, self.max_sources)
        self.placeholder = self.plotter.create_placeholder()
        self.grid = np.zeros(
            (self.rows * self.cell_h, self.cols * self.cell_w, 3), dtype=np.uint8
        )
        self.frames = {}
        self.show_stats = True
        self.running = False
        self.lock = threading.Lock()
        self.batch_times = deque(maxlen=30)
        self.prediction_fps = 0.0
        self.processed_frames = 0
        self.latency_ms = deque(maxlen=200)
        self.video_writer = None
        self.save = save
        self.worker = None
        self.error = None
        self.run()

    def setup_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            f"streamgrid_output_{self.max_sources}_streams.mp4", fourcc,
            float(self.display_fps), (self.grid.shape[1], self.grid.shape[0])
        )
        if not writer.isOpened():
            raise RuntimeError("Unable to open output video writer")
        return writer

    def process_batch(self):
        """Infer on distinct source frames; do not block the display thread."""
        interval = 1.0 / self.inference_fps
        next_run = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                if now < next_run:
                    time.sleep(min(next_run - now, 0.01))
                    continue
                next_run = max(next_run + interval, now)
                limit = self.batch_size or self.max_sources
                frame_data = self.stream_manager.get_frames(limit)
                if not frame_data:
                    if self.stream_manager.finished:
                        break
                    continue
                started = time.monotonic()
                ids = [entry[0] for entry in frame_data]
                frames = [entry[1] for entry in frame_data]
                if self.model is not None:
                    results = self.model.predict(
                        frames, conf=self.conf, verbose=False, device=self.device,
                        imgsz=self.imgsz
                    )
                    if len(results) != len(frames):
                        raise RuntimeError("Model returned unexpected result count")
                    for (source_id, frame, captured), result in zip(frame_data, results):
                        self.update_source(source_id, frame, result, captured)
                else:
                    for source_id, frame, captured in frame_data:
                        self.update_source(source_id, frame, None, captured)
                elapsed = time.monotonic() - started
                self.update_fps(len(frames), elapsed)
        except Exception as exc:
            self.error = exc
            LOGGER.exception("StreamGrid processing failed")
        finally:
            self.running = False

    def update_fps(self, frame_count, batch_time):
        self.processed_frames += frame_count
        self.batch_times.append((frame_count, batch_time))
        total_time = sum(duration for _, duration in self.batch_times)
        self.prediction_fps = (
            sum(count for count, _ in self.batch_times) / total_time
            if total_time > 0 else 0.0
        )

    def update_source(self, source_id, frame, results=None, captured=None):
        if source_id >= self.max_sources:
            return
        resized = cv2.resize(frame, (self.cell_w, self.cell_h))
        detections = 0
        if results is not None and results.boxes is not None:
            detections = len(results.boxes)
            resized = self.plotter.draw_detections(resized, results, frame.shape[:2])
        resized = self.plotter.draw_source_label(resized, source_id, self.show_stats)
        latency = 1000 * (time.monotonic() - captured) if captured else 0.0
        with self.lock:
            self.frames[source_id] = resized
            if captured:
                self.latency_ms.append(latency)
        if self.analytics:
            self.analytics.log(source_id, detections, self.prediction_fps)

    def update_display(self):
        """Draw latest processed images, return whether any images are available."""
        with self.lock:
            snapshot = dict(self.frames)
            fps = self.prediction_fps
        self.grid.fill(0)
        for i in range(self.max_sources):
            row, col = divmod(i, self.cols)
            y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
            x1, x2 = col * self.cell_w, (col + 1) * self.cell_w
            self.grid[y1:y2, x1:x2] = snapshot.get(i, self.placeholder)
        if self.show_stats:
            self.plotter.draw_fps_overlay(
                self.grid, fps, self.grid.shape[1], self.grid.shape[0]
            )
        if self.show:
            cv2.imshow("StreamGrid", self.grid)
        if self.video_writer is not None:
            self.video_writer.write(self.grid)
        return bool(snapshot)

    def run(self):
        self.running = True
        self.stream_manager.start()
        self.worker = threading.Thread(
            target=self.process_batch, name="streamgrid-inference", daemon=True
        )
        self.worker.start()
        try:
            if self.show:
                cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
                LOGGER.info("Running. Press ESC to exit, 's' to toggle stats")
            if self.save:
                self.video_writer = self.setup_video_writer()
            next_display = time.monotonic()
            while self.running or (self.worker.is_alive() and not self.error):
                now = time.monotonic()
                if now >= next_display:
                    self.update_display()
                    next_display = max(next_display + 1.0 / self.display_fps, now)
                if self.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                    if key == ord("s"):
                        self.show_stats = not self.show_stats
                else:
                    time.sleep(0.002)
            if self.error:
                raise RuntimeError("StreamGrid inference worker failed") from self.error
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            if self.show:
                cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.stream_manager.stop()
        if self.worker and self.worker.is_alive() and self.worker is not threading.current_thread():
            self.worker.join(timeout=5.0)
        if self.analytics:
            self.analytics.summary()
            self.analytics = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        with self.lock:
            self.frames.clear()
