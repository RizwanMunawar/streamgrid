"""
Core utilities: YOLO processing, color management, and text rendering.
Optimized for performance and minimal code footprint.
"""

import colorsys
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class ColorManager:
    """Ultra-fast color management with consistent class colors."""

    def __init__(self):
        self._colors = {}
        self._counter = 0

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent BGR color for class using golden ratio distribution."""
        if class_name not in self._colors:
            hue = (self._counter * 137.508) % 360
            rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
            self._colors[class_name] = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self._counter += 1
        return self._colors[class_name]


class YOLOProcessor:
    """Ultra-optimized YOLO processor with batch processing and integrated rendering."""

    def __init__(self, model, confidence: float = 0.25, batch_size: int = 4):
        self.model = model
        self.confidence = confidence
        self.batch_size = batch_size
        self.color_manager = ColorManager()

        # Optimized buffers
        self.frame_buffer = {}
        self.result_cache = {}
        self.batch_frames = []
        self.batch_ids = []

        # Threading
        self.running = False
        self.thread = None

        # Stats
        self.stats = {'fps': 0.0, 'detections': 0}
        self._last_update = time.time()
        self._frame_count = 0

    def start(self) -> bool:
        """Start processing thread."""
        if not self.model:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        time.sleep(0.1)  # Wait for thread readiness
        return True

    def _process_loop(self):
        """Optimized processing loop with minimal overhead."""
        while self.running:
            try:
                # Collect batch
                self._collect_frames()

                # Process if batch ready
                if len(self.batch_frames) >= self.batch_size:
                    self._process_batch()
                elif self.batch_frames:
                    time.sleep(0.02)  # Short wait for more frames
                    if self.batch_frames:
                        self._process_batch()
                else:
                    time.sleep(0.005)

            except Exception as e:
                print(f"YOLO error: {e}")
                time.sleep(0.1)

    def _collect_frames(self):
        """Collect frames for batch processing."""
        for stream_id, frame_data in list(self.frame_buffer.items()):
            if frame_data and len(self.batch_frames) < self.batch_size * 2:
                if stream_id not in self.batch_ids:
                    self.batch_frames.append(frame_data['frame'])
                    self.batch_ids.append(stream_id)
                    self.frame_buffer[stream_id] = None

    def _process_batch(self):
        """Process batch with integrated detection drawing."""
        if not self.batch_frames:
            return

        try:
            # Get batch to process
            frames = self.batch_frames.copy()  # Process all available frames
            self.batch_frames.clear()
            ids = self.batch_ids[:self.batch_size]

            # Clear processed items
            self.batch_frames = self.batch_frames[self.batch_size:]
            self.batch_ids = self.batch_ids[self.batch_size:]

            # YOLO inference
            results = self.model.predict(frames, conf=self.confidence, verbose=False, device="cpu")
            if not isinstance(results, list):
                results = [results]

            # Process and cache results
            for i, (frame, result, stream_id) in enumerate(zip(frames, results, ids)):
                if i >= len(results):
                    break

                # Draw detections directly
                annotated = self._draw_detections(frame, result)
                detection_count = len(result.boxes) if result.boxes else 0

                # Cache result
                self.result_cache[stream_id] = {
                    'frame': annotated,
                    'detections': detection_count,
                    'timestamp': time.time()
                }

                # Update stats
                self.stats['detections'] += detection_count

            # Update performance stats
            self._update_stats(len(frames))

        except Exception as e:
            print(f"Batch error: {e}")
            self.batch_frames.clear()
            self.batch_ids.clear()

    def _draw_detections(self, frame: np.ndarray, result) -> np.ndarray:
        """Optimized detection drawing with minimal function calls."""
        if not result.boxes:
            return frame

        # Get detection data
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # Draw all detections in one loop
        for box, conf, cls in zip(boxes, confs, classes):
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = box.astype(int)
            class_name = self.model.names[int(cls)]
            color = self.color_manager.get_color(class_name)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with optimized text rendering
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = y1 - 5 if y1 > 30 else y1 + 25

            # Background and text in one operation
            cv2.rectangle(frame, (x1, y_text - h - 5), (x1 + w + 6, y_text + 5), color, -1)
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return frame

    def _update_stats(self, batch_size: int):
        """Update performance statistics."""
        self._frame_count += batch_size
        current_time = time.time()

        if current_time - self._last_update >= 1.0:
            self.stats['fps'] = self._frame_count / (current_time - self._last_update)
            self._frame_count = 0
            self._last_update = current_time

    def add_frame(self, stream_id: int, frame: np.ndarray) -> bool:
        """Add frame for processing."""
        if not self.running:
            return False

        self.frame_buffer[stream_id] = {'frame': frame, 'timestamp': time.time()}
        return True

    def get_result(self, stream_id: int) -> Optional[Dict]:
        """Get latest result for stream."""
        return self.result_cache.get(stream_id)

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        return self.stats.copy()

    def stop(self):
        """Stop processing and cleanup."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        # Clear all buffers
        self.frame_buffer.clear()
        self.result_cache.clear()
        self.batch_frames.clear()
        self.batch_ids.clear()


# Text utilities - optimized for performance
def draw_text_bg(img, text, pos, scale=0.5, color=(255, 255, 255), bg=(0, 0, 0), thickness=1):
    """Ultra-fast text with background rendering."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos

    if bg:
        cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 2), bg, -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img


def draw_stats_panel(img, stats, pos=(10, 10)):
    """Optimized stats panel rendering."""
    if not stats:
        return img

    # Calculate panel size
    max_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in stats)
    panel_h = len(stats) * 20 + 10

    # Draw background
    cv2.rectangle(img, pos, (pos[0] + max_w + 16, pos[1] + panel_h), (0, 0, 0), -1)

    # Draw stats
    for i, stat in enumerate(stats):
        cv2.putText(img, stat, (pos[0] + 8, pos[1] + 20 + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


# Global YOLO processor for memory efficiency
_global_yolo = None


def get_yolo_processor(model=None, confidence=0.25, batch_size=4):
    """Get or create global YOLO processor."""
    global _global_yolo

    if _global_yolo is None and model:
        _global_yolo = YOLOProcessor(model, confidence, batch_size)
        _global_yolo.start()

    return _global_yolo


def cleanup_yolo():
    """Cleanup global YOLO processor."""
    global _global_yolo
    if _global_yolo:
        _global_yolo.stop()
        _global_yolo = None