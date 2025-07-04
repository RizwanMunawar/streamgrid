import math
import time
import threading
import queue
import cv2
import numpy as np


class StreamGrid:
    """StreamGrid for multi-stream video display with batch processing."""

    def __init__(self, sources, model=None, batch_size=4):
        self.sources = sources
        self.max_sources = len(sources)
        self.cols = int(math.ceil(math.sqrt(self.max_sources)))
        self.rows = int(math.ceil(self.max_sources / self.cols))

        # Auto cell size based on source count
        sizes = {1: (1280, 720), 4: (640, 360), 9: (480, 270), 16: (320, 180)}
        self.cell_w, self.cell_h = next((s for n, s in sizes.items() if self.max_sources <= n), (240, 135))

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
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=50)

        # Colors
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 0, 255), (0, 255, 255),
                       (255, 128, 0), (128, 0, 255)]

    def get_color(self, class_idx):
        return self.colors[class_idx % len(self.colors)]

    def capture_video(self, source, source_id):
        """Capture video frames with CPU optimizations."""
        cap = cv2.VideoCapture(source)

        # Reduce capture resolution for CPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer

        frame_count = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            try:
                self.frame_queue.put((source_id, frame), timeout=0.01)
            except queue.Full:
                pass
            time.sleep(0.05)  # Slower for CPU
        cap.release()

    def _batch_worker(self):
        """Batch processing worker with CPU optimizations."""
        batch_frames, batch_ids = [], []

        while self.running:
            # Collect frames
            while len(batch_frames) < self.batch_size:
                try:
                    source_id, frame = self.frame_queue.get(timeout=0.01)
                    batch_frames.append(frame)
                    batch_ids.append(source_id)
                except queue.Empty:
                    break

            if batch_frames:
                try:
                    if self.model:
                        results = self.model.predict(
                            batch_frames,
                            conf=0.25,
                            verbose=False,
                            device='cpu',  # Force CPU
                            half=False,  # No half precision on CPU
                        )

                        for source_id, frame, result in zip(batch_ids, batch_frames, results):
                            self.update_source(source_id, frame, result)
                    else:
                        for source_id, frame in zip(batch_ids, batch_frames):
                            self.update_source(source_id, frame)
                except Exception as e:
                    print(f"Batch error: {e}")

                batch_frames.clear()
                batch_ids.clear()

    def update_source(self, source_id, frame, yolo_results=None):
        """Update frame and results for a source."""
        if source_id >= self.max_sources:
            return

        with self.lock:
            # Resize frame
            resized = cv2.resize(frame, (self.cell_w, self.cell_h))

            # Draw detections
            detections = 0
            if yolo_results and yolo_results.boxes is not None:
                detections = len(yolo_results.boxes)
                resized = self.draw_boxes(resized, yolo_results, frame.shape[:2])

            self.frames[source_id] = resized
            self.stats[source_id] = {'detections': detections, 'time': time.time()}

    def draw_boxes(self, frame, results, orig_shape):
        """Draw YOLO detections with proper scaling."""
        if not results.boxes:
            return frame

        # Scale factors
        scale_x = self.cell_w / orig_shape[1]  # orig_shape from YOLO input
        scale_y = self.cell_h / orig_shape[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            # Scale coordinates
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)

            # Draw box
            class_name = results.names[int(cls)]
            color = self.get_color(int(cls))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y_text = max(y1 - 5, 15)

            cv2.rectangle(frame, (x1, y_text - h - 3), (x1 + w + 4, y_text + 3), color, -1)
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        return frame

    def run(self):
        """Run display loop."""
        self.running = True

        # Start capture threads
        for i, source in enumerate(self.sources):
            threading.Thread(target=self.capture_video, args=(source, i), daemon=True).start()

        # Start batch processing
        threading.Thread(target=self._batch_worker, daemon=True).start()

        cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
        print("StreamGrid running. Press ESC to exit, 's' to toggle stats")

        while self.running:
            self.update_display()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats

        cv2.destroyAllWindows()

    def update_display(self):
        """Update grid display."""
        self.grid.fill(0)

        with self.lock:
            for i in range(self.max_sources):
                row, col = divmod(i, self.cols)
                y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
                x1, x2 = col * self.cell_w, (col + 1) * self.cell_w

                if i in self.frames:
                    frame = self.frames[i].copy()
                else:
                    # Create placeholder
                    frame = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)

                    # Checkerboard pattern
                    for y in range(0, self.cell_h, 20):
                        for x in range(0, self.cell_w, 20):
                            if (x // 20 + y // 20) % 2:
                                frame[y:y + 20, x:x + 20] = 20

                    # Center "WAITING" text
                    text = "WAITING"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.putText(frame, text, ((self.cell_w - w) // 2, (self.cell_h - h) // 2 + h),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

                info = f"Source #{i}"
                (text_width, text_height), baseline = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                cv2.rectangle(frame, (2, 2), (2 + text_width + 8, 2 + text_height + baseline + 8), (0, 0, 0),
                              -1)
                cv2.putText(frame, info, (6, 6 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                            1)

                self.grid[y1:y2, x1:x2] = frame

        cv2.imshow("StreamGrid", self.grid)

    def stop(self):
        """Stop display."""
        self.running = False