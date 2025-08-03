# StreamGrid - Plotting and Drawing

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


class StreamAnnotator:
    """Handles all drawing and plotting operations for StreamGrid."""

    def __init__(self, cell_w, cell_h, colors_palette=None):
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.colors = [
            (255, 0, 0),
            (104, 31, 17),
            (0, 0, 255),
            (128, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

    def draw_detections(self, frame, results, orig_shape):
        """Draw YOLO detections on frame."""
        if not results or not results.boxes:
            return frame

        ann = Annotator(frame)
        scale_x = self.cell_w / orig_shape[1]
        scale_y = self.cell_h / orig_shape[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            ann.box_label([x1, y1, x2, y2], label=label, color=colors(int(cls), True))

        return frame

    def draw_source_label(self, frame, source_id, show_stats=True):
        """Draw source identifier label on frame with dynamic sizing."""
        if not show_stats:
            return frame

        info = f"{source_id}"
        bg_color = self.colors[source_id % len(self.colors)]

        # Calculate text dimensions dynamically
        font_scale = 1.2
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            info, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Dynamic rectangle size based on text
        padding = 10
        rect_w = text_w + padding
        rect_h = text_h + baseline + padding

        # Background rectangle
        cv2.rectangle(frame, (2, 2), (2 + rect_w, 2 + rect_h), bg_color, -1)

        # Text color based on background brightness
        text_color = (0, 0, 0) if sum(bg_color) > 384 else (255, 255, 255)
        cv2.putText(
            frame,
            info,
            (padding, padding + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )

        return frame

    def create_placeholder(self):
        """Create placeholder frame for inactive sources."""
        frame = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)

        # Checkerboard pattern
        for y in range(0, self.cell_h, 20):
            for x in range(0, self.cell_w, 20):
                if (x // 20 + y // 20) % 2:
                    frame[y : y + 20, x : x + 20] = 20

        # "WAITING" text
        text = "WAITING"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame,
            text,
            ((self.cell_w - w) // 2, (self.cell_h + h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 100),
            2,
        )

        return frame

    def draw_fps_overlay(self, grid, fps, grid_width, grid_height):
        """Draw FPS overlay in a fixed circle at the center of the entire grid."""
        if fps <= 0:
            return grid

        fps_text = f"{fps:.1f}"

        # Center position
        center_x = grid_width // 2
        center_y = grid_height // 2

        # Fixed circle radius
        radius = 35

        # Draw circle background (black)
        cv2.circle(grid, (center_x, center_y), radius, (255, 255, 255), -1)

        # White border circle for better contrast
        cv2.circle(grid, (center_x, center_y), radius, (186, 0, 221), 3)

        # Calculate font scale to fit text inside circle
        # Start with a reasonable font scale and adjust if needed
        font_scale = 1.1
        thickness = 3

        # Get text dimensions and adjust font scale if text is too big
        (text_w, text_h), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Ensure text fits within circle (leave some margin)
        max_text_width = radius * 1.6  # Allow text to be 70% of circle diameter
        if text_w > max_text_width:
            font_scale = font_scale * (max_text_width / text_w)
            thickness = max(2, int(font_scale * 2))
            (text_w, text_h), baseline = cv2.getTextSize(
                fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

        # Center text in circle
        text_x = center_x - text_w // 2
        text_y = center_y + text_h // 2

        cv2.putText(
            grid,
            fps_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (104, 31, 17),
            thickness,
        )

        return grid
