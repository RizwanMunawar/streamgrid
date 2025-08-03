# StreamGrid - Stream Management

import threading
import time
import queue
import cv2
import requests
from pathlib import Path
from tqdm import tqdm
from .utils import LOGGER


class StreamManager:
    """Manages multiple video streams with threading and queue processing."""

    def __init__(self, sources=None):
        self.sources = sources or self.get_default_videos()
        self.frame_queue = queue.Queue(maxsize=max(50, len(self.sources) * 4))
        self.active_streams = len(self.sources)
        self.running = False
        self.lock = threading.Lock()

    def get_default_videos(self):
        """Download demo videos if no sources provided."""
        LOGGER.warning("⚠️ No sources provided. Downloading default demo videos.")

        base_url = (
            "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/"
        )
        videos = ["grid_1.mp4", "grid_2.mp4", "grid_3.mp4", "grid_4.mp4"]
        demo_dir = Path("assets")
        demo_dir.mkdir(exist_ok=True)

        paths = []
        for video in videos:
            local_path = demo_dir / video
            if not local_path.exists():
                LOGGER.info(f"ℹ️ Downloading {video}...")
                try:
                    response = requests.get(f"{base_url}{video}", stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))

                    with open(local_path, "wb") as f, tqdm(
                        desc=video, total=total_size, unit="B", unit_scale=True
                    ) as pbar:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                except Exception as e:
                    LOGGER.error(f"❌ Failed to download {video}: {e}")
                    continue
            paths.append(str(local_path))
        return paths

    def capture_stream(self, source, source_id):
        """Capture frames from a single stream."""
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                LOGGER.error(f"❌ Failed to open source: {source}")
                with self.lock:
                    self.active_streams -= 1
                return

            # Optimize capture settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            no_frame_count = 0
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    no_frame_count += 1
                    if no_frame_count > 5:
                        with self.lock:
                            self.active_streams -= 1
                        break
                    time.sleep(0.1)
                    continue

                no_frame_count = 0
                try:
                    self.frame_queue.put((source_id, frame), timeout=0.01)
                except queue.Full:
                    pass  # Drop frame if queue full
                time.sleep(0.05)

        except Exception as e:
            LOGGER.error(f"❌ Stream error {source}: {e}")
            with self.lock:
                self.active_streams -= 1
        finally:
            if "cap" in locals():
                cap.release()

    def start(self):
        """Start all stream capture threads."""
        self.running = True
        for i, source in enumerate(self.sources):
            thread = threading.Thread(
                target=self.capture_stream, args=(source, i), daemon=True
            )
            thread.start()

    def get_frames(self, max_frames=None):
        """Get available frames from queue."""
        frames = []
        max_frames = max_frames or len(self.sources)

        while len(frames) < max_frames:
            try:
                frames.append(self.frame_queue.get(timeout=0.01))
            except queue.Empty:
                break
        return frames

    def stop(self):
        """Stop all streams."""
        self.running = False
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:  # noqa
            pass
