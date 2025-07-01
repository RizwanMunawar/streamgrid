"""Simple YOLO integration for StreamGrid"""

import threading
import time
from queue import Queue, Empty
from typing import Optional, List, Dict, Any
import numpy as np
import cv2


class YOLOProcessor:
    """YOLO processor that uses user's model instance with CPU batch processing"""

    def __init__(self, model, confidence: float = 0.25, batch_size: int = 4):
        """Initialize with user's YOLO model."""
        self.model = model
        self.confidence = confidence
        self.batch_size = batch_size

        # Batch processing for CPU efficiency
        self.frame_buffer = {}      # stream_id -> latest frame
        self.result_cache = {}      # stream_id -> latest result
        self.batch_queue = []       # Frames ready for batch processing
        self.batch_metadata = []    # Corresponding stream IDs

        # Processing thread
        self.running = False
        self.thread = None

        print(f"YOLO processor initialized with CPU batch processing (batch_size={batch_size})")
        print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'Custom'}")

    def start(self):
        """Start YOLO processing."""
        if not self.model:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        return True

    def _process_loop(self):
        """CPU-optimized batch processing loop."""
        while self.running:
            try:
                # Collect frames for batch processing
                self.collect_batch()

                # Process batch if we have enough frames
                if len(self.batch_queue) >= self.batch_size:
                    self.process_batch()
                elif len(self.batch_queue) > 0:
                    # Process remaining frames after a timeout
                    time.sleep(0.05)  # 50ms timeout
                    if len(self.batch_queue) > 0:  # Still have frames
                        self.process_batch()
                else:
                    time.sleep(0.01)  # Wait for frames

            except Exception as e:
                if self.running:
                    print(f"YOLO batch processing error: {e}")
                time.sleep(0.1)

    def collect_batch(self):
        """Collect frames from buffer into batch queue."""
        # Get latest frame from each stream that has new data
        for stream_id, frame_data in list(self.frame_buffer.items()):
            if frame_data and len(self.batch_queue) < self.batch_size * 2:  # Limit queue size
                # Check if this frame is newer than what we already have in batch
                already_in_batch = any(meta['stream_id'] == stream_id for meta in self.batch_metadata)
                if not already_in_batch:
                    self.batch_queue.append(frame_data['frame'])
                    self.batch_metadata.append({
                        'stream_id': stream_id,
                        'timestamp': frame_data['timestamp']
                    })
                    # Clear from buffer to avoid reprocessing
                    self.frame_buffer[stream_id] = None

    def process_batch(self):
        """Process collected batch of frames."""
        if not self.batch_queue:
            return

        try:
            batch_frames = self.batch_queue[:self.batch_size]
            batch_meta = self.batch_metadata[:self.batch_size]

            # Clear processed items from queues
            self.batch_queue = self.batch_queue[self.batch_size:]
            self.batch_metadata = self.batch_metadata[self.batch_size:]

            # CPU batch processing - handle different sizes
            if len(batch_frames) == 1:
                results = self.model.predict(batch_frames[0], conf=self.confidence, verbose=False)
            else:
                # Process batch individually (YOLO handles batching internally)
                results = self.model.predict(batch_frames, conf=self.confidence, verbose=False)

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            # Process results and update cache
            for i, (result, meta) in enumerate(zip(results, batch_meta)):
                if i >= len(results):
                    break

                annotated_frame = self.draw_detections(batch_frames[i], result)

                self.result_cache[meta['stream_id']] = {
                    'frame': annotated_frame,
                    'detections': len(result.boxes) if result.boxes is not None else 0,
                    'timestamp': time.time()
                }

            # Print batch processing stats occasionally
            if hasattr(self, '_batch_count'):
                self._batch_count += 1
            else:
                self._batch_count = 1

            if self._batch_count % 20 == 0:  # Every 20 batches
                print(f"Processed batch #{self._batch_count}, size: {len(batch_frames)}")

        except Exception as e:
            print(f"Batch processing error: {e}")
            # Clear problematic batch
            self.batch_queue.clear()
            self.batch_metadata.clear()

    def draw_detections(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw YOLO detections on frame."""
        annotated = frame.copy()

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                class_name = self.model.names[int(cls)]
                label = f"{class_name}: {conf:.2f}"

                # Draw box and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 5),
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated

    def add_frame(self, stream_id: int, frame: np.ndarray) -> bool:
        """Add frame to buffer for batch processing."""
        if not self.running:
            return False

        # Store latest frame from each stream
        self.frame_buffer[stream_id] = {
            'frame': frame,
            'timestamp': time.time()
        }
        return True

    def get_result(self, stream_id: int) -> Optional[Dict]:
        """Get latest result for specific stream."""
        return self.result_cache.get(stream_id, None)

    def stop(self):
        """Stop YOLO processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        # Clear all buffers
        self.frame_buffer.clear()
        self.result_cache.clear()
        self.batch_queue.clear()
        self.batch_metadata.clear()


# Global YOLO processor instance
_yolo_processor = None


def get_yolo_processor(model=None, confidence: float = 0.25, batch_size: int = 4) -> Optional[YOLOProcessor]:
    """Get global YOLO processor instance."""
    global _yolo_processor

    if _yolo_processor is None and model is not None:
        _yolo_processor = YOLOProcessor(model, confidence, batch_size)
        if _yolo_processor.start():
            return _yolo_processor
        else:
            _yolo_processor = None

    return _yolo_processor


def stop_yolo():
    """Stop global YOLO processor."""
    global _yolo_processor
    if _yolo_processor:
        _yolo_processor.stop()
        _yolo_processor = None