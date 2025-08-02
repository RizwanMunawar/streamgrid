"""Stream handling for StreamGrid - optimized for performance and stability."""

import threading
import time
import queue
import cv2
from collections import deque
from .utils import LOGGER


class VideoStream:
    """Optimized video stream handler with automatic reconnection and frame dropping."""

    def __init__(self, source, stream_id, target_fps=15, buffer_size=2):
        """Initialize video stream.

        Args:
            source: Video source (file path, camera index, or URL)
            stream_id: Unique identifier for this stream
            target_fps: Target frames per second
            buffer_size: Internal buffer size (smaller = lower latency)
        """
        self.source = source
        self.stream_id = stream_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # Threading and control
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)

        # Performance tracking
        self.fps_tracker = deque(maxlen=30)
        self.last_frame_time = 0
        self.frame_count = 0

        # Connection management
        self.cap = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0

    def start(self):
        """Start the video stream capture."""
        if self.running:
            return True

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop the video stream capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._release_capture()

    def get_frame(self, timeout=0.01):
        """Get the latest frame from the stream.

        Args:
            timeout: Maximum time to wait for a frame

        Returns:
            Frame data or None if no frame available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None

    def get_fps(self):
        """Get current FPS of the stream."""
        if len(self.fps_tracker) < 2:
            return 0.0
        return len(self.fps_tracker) / (self.fps_tracker[-1] - self.fps_tracker[0]) if self.fps_tracker else 0.0

    def is_active(self):
        """Check if stream is actively capturing frames."""
        return self.running and self.thread and self.thread.is_alive()

    def _capture_loop(self):
        """Main capture loop - runs in separate thread."""
        while self.running:
            if not self._ensure_connection():
                time.sleep(self.reconnect_delay)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._handle_read_failure()
                continue

            # Frame rate control
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                continue

            # Update performance tracking
            self.fps_tracker.append(current_time)
            self.last_frame_time = current_time
            self.frame_count += 1

            # Add frame to queue (drop old frames if queue is full)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self.frame_queue.get(block=False)
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass

    def _ensure_connection(self):
        """Ensure video capture is connected."""
        if self.cap and self.cap.isOpened():
            return True

        return self._connect()

    def _connect(self):
        """Connect to video source."""
        try:
            self._release_capture()
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise Exception(f"Failed to open source: {self.source}")

            # Optimize capture settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.reconnect_attempts = 0
            LOGGER.info(f"✅ Stream {self.stream_id} connected: {self.source}")
            return True

        except Exception as e:
            self.reconnect_attempts += 1
            if self.reconnect_attempts <= self.max_reconnect_attempts:
                LOGGER.warning(f"⚠️ Stream {self.stream_id} connection failed (attempt {self.reconnect_attempts}): {e}")
            return False

    def _handle_read_failure(self):
        """Handle frame read failure."""
        if isinstance(self.source, str) and not self.source.isdigit():
            # For file sources, end of file is normal
            if self.cap and self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                LOGGER.info(f"📹 Stream {self.stream_id} reached end of file")
                self.running = False
                return

        # For cameras/streams, try to reconnect
        self._release_capture()
        time.sleep(0.1)

    def _release_capture(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


class StreamManager:
    """Manages multiple video streams efficiently."""

    def __init__(self, sources, target_fps=15):
        """Initialize stream manager.

        Args:
            sources: List of video sources
            target_fps: Target FPS for all streams
        """
        self.sources = sources if isinstance(sources, list) else [sources]
        self.streams = {}
        self.target_fps = target_fps

        # Create streams
        for i, source in enumerate(self.sources):
            self.streams[i] = VideoStream(source, i, target_fps)

    def start_all(self):
        """Start all video streams."""
        for stream in self.streams.values():
            stream.start()

        # Wait a moment for streams to initialize
        time.sleep(0.5)

        active_count = sum(1 for stream in self.streams.values() if stream.is_active())
        LOGGER.info(f"🚀 Started {active_count}/{len(self.streams)} streams")

    def stop_all(self):
        """Stop all video streams."""
        for stream in self.streams.values():
            stream.stop()
        LOGGER.info("🛑 All streams stopped")

    def get_frames(self):
        """Get current frames from all active streams.

        Returns:
            dict: {stream_id: frame} for streams with available frames
        """
        frames = {}
        for stream_id, stream in self.streams.items():
            if stream.is_active():
                frame = stream.get_frame()
                if frame is not None:
                    frames[stream_id] = frame
        return frames

    def get_stream_stats(self):
        """Get statistics for all streams.

        Returns:
            dict: {stream_id: {'fps': float, 'active': bool, 'frames': int}}
        """
        stats = {}
        for stream_id, stream in self.streams.items():
            stats[stream_id] = {
                'fps': stream.get_fps(),
                'active': stream.is_active(),
                'frames': stream.frame_count
            }
        return stats

    def get_active_count(self):
        """Get number of active streams."""
        return sum(1 for stream in self.streams.values() if stream.is_active())
