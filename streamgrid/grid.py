import math
import time
import threading
import cv2
import numpy as np


class StreamGrid:
    """Ultra-optimized StreamGrid for external frame/result input."""

    def __init__(self, max_sources=4):
        self.max_sources = max_sources
        self.cols = int(math.ceil(math.sqrt(max_sources)))
        self.rows = int(math.ceil(max_sources / self.cols))

        # Auto cell size based on source count
        sizes = {1: (1280, 720), 4: (640, 360), 9: (480, 270), 16: (320, 180)}
        self.cell_w, self.cell_h = next((s for n, s in sizes.items() if max_sources <= n), (240, 135))

        self.grid = np.zeros((self.rows * self.cell_h, self.cols * self.cell_w, 3), dtype=np.uint8)
        self.frames = {}
        self.stats = {}
        self.show_stats = True
        self.running = False
        self.lock = threading.Lock()

        # Pre-generate colors for classes
        self.colors = {}
        self.color_idx = 0

    def _get_color(self, class_name):
        """Get consistent color for class."""
        if class_name not in self.colors:
            hue = int((self.color_idx * 137.5) % 180)  # OpenCV hue is 0-179
            rgb = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            self.colors[class_name] = tuple(map(int, rgb))
            self.color_idx += 1
        return self.colors[class_name]

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
                resized = self._draw_boxes(resized, yolo_results, frame.shape[:2])

            self.frames[source_id] = resized
            self.stats[source_id] = {'detections': detections, 'time': time.time()}

    def _draw_boxes(self, frame, results, orig_shape):
        """Draw YOLO detections with proper scaling."""
        if not results.boxes:
            return frame

        # Scale factors
        scale_x = self.cell_w / orig_shape[1]
        scale_y = self.cell_h / orig_shape[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            # Scale coordinates
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)

            # Draw box
            class_name = results.names[int(cls)]
            color = self._get_color(class_name)
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
        cv2.namedWindow("StreamGrid", cv2.WINDOW_AUTOSIZE)
        print("StreamGrid running. Press ESC to exit, 's' to toggle stats")

        while self.running:
            self._update_display()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats

        cv2.destroyAllWindows()

    def _update_display(self):
        """Update grid display."""
        self.grid.fill(0)

        with self.lock:
            for i in range(self.max_sources):
                row, col = divmod(i, self.cols)
                y1, y2 = row * self.cell_h, (row + 1) * self.cell_h
                x1, x2 = col * self.cell_w, (col + 1) * self.cell_w

                if i in self.frames:
                    frame = self.frames[i].copy()

                    # Add stats
                    if self.show_stats:
                        info = f"Source #{i}"
                        if i in self.stats and self.stats[i]['detections'] > 0:
                            info += f" - {self.stats[i]['detections']} objects"

                        cv2.rectangle(frame, (2, 2), (len(info) * 6 + 6, 18), (0, 0, 0), -1)
                        cv2.putText(frame, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    # Create placeholder
                    frame = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)

                    # Checkerboard pattern
                    for y in range(0, self.cell_h, 20):
                        for x in range(0, self.cell_w, 20):
                            if (x // 20 + y // 20) % 2:
                                frame[y:y + 20, x:x + 20] = 20

                    cv2.putText(frame, "WAITING", (self.cell_w // 4, self.cell_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.putText(frame, f"Source #{i}", (self.cell_w // 4, self.cell_h // 2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                self.grid[y1:y2, x1:x2] = frame

        # # Global stats
        # if self.show_stats:
        #     active = len(self.frames)
        #     total_detections = sum(s.get('detections', 0) for s in self.stats.values())
        #
        #     stats_text = [
        #         f"Active: {active}/{self.max_sources}",
        #         f"Objects: {total_detections}"
        #     ]
        #
        #     # Draw stats panel
        #     max_w = max(len(s) * 6 for s in stats_text)
        #     cv2.rectangle(self.grid, (10, 10), (20 + max_w, 50), (0, 0, 0), -1)
        #
        #     for i, stat in enumerate(stats_text):
        #         cv2.putText(self.grid, stat, (15, 28 + i * 15),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("StreamGrid", self.grid)

    def stop(self):
        """Stop display."""
        self.running = False
