"""Simple YOLO integration for StreamGrid"""

import threading
import time
from queue import Queue, Empty
from typing import Optional, List, Dict, Any
import numpy as np
import cv2


class YOLOProcessor:
    """YOLO processor that uses user's model instance"""

    def __init__(self, model, confidence: float = 0.25):
        """Initialize with user's YOLO model."""
        self.model = model
        self.confidence = confidence

        # Separate queues for each stream to avoid mixing results
        self.frame_queues = {}  # stream_id -> queue
        self.result_cache = {}  # stream_id -> latest result

        # Processing thread
        self.running = False
        self.thread = None

        print(f"YOLO processor initialized with model: {model.model_name if hasattr(model, 'model_name') else 'Custom'}")

    def start(self):
        """Start YOLO processing."""
        if not self.model:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        return True

    def _process_loop(self):
        """Main processing loop - handles all streams independently."""
        while self.running:
            try:
                # Process frames from all streams
                for stream_id, queue in list(self.frame_queues.items()):
                    try:
                        # Get the latest frame, skip old ones
                        frame = None
                        frame_count = 0

                        # Drain queue to get latest frame
                        while not queue.empty() and frame_count < 5:
                            try:
                                frame = queue.get_nowait()
                                frame_count += 1
                            except Empty:
                                break

                        if frame is not None:
                            # Use user's model with predict method
                            results = self.model.predict(frame, conf=self.confidence, verbose=False)

                            # Draw detections
                            annotated_frame = self._draw_detections(frame, results[0])

                            # Update result cache
                            self.result_cache[stream_id] = {
                                'frame': annotated_frame,
                                'detections': len(results[0].boxes) if results[0].boxes is not None else 0,
                                'timestamp': time.time()
                            }

                    except Exception as e:
                        if self.running:
                            print(f"YOLO processing error for stream {stream_id}: {e}")

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

            except Exception as e:
                if self.running:
                    print(f"YOLO processing error: {e}")
                time.sleep(0.1)

    def _draw_detections(self, frame: np.ndarray, result) -> np.ndarray:
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
        """Add frame for processing."""
        if not self.running:
            return False

        # Create queue for new stream
        if stream_id not in self.frame_queues:
            self.frame_queues[stream_id] = Queue(maxsize=2)

        try:
            # If queue is full, remove old frame first
            if self.frame_queues[stream_id].full():
                try:
                    self.frame_queues[stream_id].get_nowait()
                except Empty:
                    pass

            self.frame_queues[stream_id].put_nowait(frame)
            return True
        except:
            return False

    def get_result(self, stream_id: int) -> Optional[Dict]:
        """Get latest result for specific stream."""
        return self.result_cache.get(stream_id, None)

    def stop(self):
        """Stop YOLO processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        # Clear caches
        self.frame_queues.clear()
        self.result_cache.clear()


# Global YOLO processor instance
_yolo_processor = None


def get_yolo_processor(model=None, confidence: float = 0.25) -> Optional[YOLOProcessor]:
    """Get global YOLO processor instance."""
    global _yolo_processor

    if _yolo_processor is None and model is not None:
        _yolo_processor = YOLOProcessor(model, confidence)
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