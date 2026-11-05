"""Per-source, bounded frame capture for live and file inputs."""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

from .utils import LOGGER


class StreamManager:
    """Decode each source independently without accumulating stale live frames."""

    def __init__(self, sources=None, buffer_mode="latest", queue_size=8, capture_fps=None):
        self.sources = sources if sources is not None else self.get_default_videos()
        if not self.sources:
            raise ValueError("At least one video source is required")
        if buffer_mode not in ("latest", "fifo"):
            raise ValueError("buffer_mode must be 'latest' or 'fifo'")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if capture_fps is not None and capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        self.buffer_mode = buffer_mode
        self.queue_size = queue_size
        self.capture_fps = capture_fps
        self.buffers = [deque(maxlen=1 if buffer_mode == "latest" else queue_size)
                        for _ in self.sources]
        self.ended = [False] * len(self.sources)
        self.dropped = [0] * len(self.sources)
        self.decoded = [0] * len(self.sources)
        self.running = False
        self.condition = threading.Condition()
        self.threads = []
        self._cursor = 0

    def get_default_videos(self):
        """Download demo inputs on demand."""
        LOGGER.warning("No sources provided; downloading default demo videos")
        base = "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/"
        folder = Path("assets")
        folder.mkdir(exist_ok=True)
        paths = []
        for name in ("grid_1.mp4", "grid_2.mp4", "grid_3.mp4", "grid_4.mp4"):
            dest = folder / name
            if not dest.exists():
                try:
                    response = requests.get(base + name, stream=True, timeout=30)
                    response.raise_for_status()
                    with open(dest, "wb") as output, tqdm(
                        total=int(response.headers.get("content-length", 0)),
                        unit="B", unit_scale=True, desc=name
                    ) as bar:
                        for chunk in response.iter_content(8192):
                            output.write(chunk)
                            bar.update(len(chunk))
                except (OSError, requests.RequestException) as exc:
                    LOGGER.warning("Cannot download %s: %s", name, exc)
                    continue
            paths.append(str(dest))
        return paths

    def capture_stream(self, source, source_id):
        """Read frames; a bounded buffer prevents unbounded latency and memory use."""
        cap = None
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                LOGGER.error("Unable to open source %s", source)
                return
            next_read = time.monotonic()
            while self.running and cap.isOpened():
                if self.capture_fps:
                    remaining = next_read - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(remaining, 0.05))
                        continue
                    next_read = time.monotonic() + 1 / self.capture_fps
                ok, frame = cap.read()
                if not ok:
                    # EOF for files; on live sources, let the caller restart the app.
                    break
                timestamp = time.monotonic()
                with self.condition:
                    buffer = self.buffers[source_id]
                    if self.buffer_mode == "fifo":
                        while self.running and len(buffer) >= buffer.maxlen:
                            self.condition.wait(timeout=0.05)
                        if not self.running:
                            break
                    elif len(buffer) == buffer.maxlen:
                        self.dropped[source_id] += 1
                    buffer.append((source_id, frame, timestamp))
                    self.decoded[source_id] += 1
                    self.condition.notify_all()
        except Exception:
            LOGGER.exception("Capture failed for source %s", source)
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.ended[source_id] = True
                self.condition.notify_all()

    def start(self):
        self.running = True
        for i, source in enumerate(self.sources):
            worker = threading.Thread(
                target=self.capture_stream, args=(source, i), daemon=True,
                name=f"streamgrid-capture-{i}"
            )
            self.threads.append(worker)
            worker.start()

    def get_frames(self, max_frames=None, timeout=0.02):
        """Return at most one frame per stream, round-robin, with capture timestamps."""
        limit = max_frames or len(self.sources)
        with self.condition:
            if not any(self.buffers) and self.running and not all(self.ended):
                self.condition.wait(timeout=timeout)
            frames = []
            count = len(self.sources)
            for step in range(count):
                index = (self._cursor + step) % count
                if self.buffers[index] and len(frames) < limit:
                    frames.append(self.buffers[index].popleft())
            self._cursor = (self._cursor + 1) % count
            self.condition.notify_all()
            return frames

    @property
    def finished(self):
        with self.condition:
            return all(self.ended) and not any(self.buffers)

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
