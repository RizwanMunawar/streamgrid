"""Plotting and annotation utilities for StreamGrid."""

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


class StreamAnnotator:
    """Optimized annotator for StreamGrid with consistent styling."""

    def __init__(self, cell_width, cell_height):
        """Initialize annotator with cell dimensions.

        Args:
            cell_width: Width of each grid cell
            cell_height: Height of each grid cell
        """
        self.cell_w = cell_width
        self.cell_h = cell_height

        # Pre-calculate text scaling based on cell size
        self.text_scale = max(0.4, min(0.8, cell_width / 400))
        self.text_thickness = max(1, int(self.text_scale * 3))
        self.padding = max(10, int(cell_width / 80))

        # Source label colors
        self.source_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]

    def annotate_detections(self, frame, results, orig_shape):
        """Draw YOLO detection boxes and labels on frame.

        Args:
            frame: Resized frame to draw on
            results: YOLO detection results
            orig_shape: Original frame shape (height, width)

        Returns:
            Annotated frame
        """
        if not results or not results.boxes:
            return frame

        # Use Ultralytics annotator for consistency
        ann = Annotator(frame, line_width=max(1, int(self.cell_w / 200)))

        # Calculate scaling factors
        scale_x = self.cell_w / orig_shape[1]
        scale_y = self.cell_h / orig_shape[0]

        # Extract detection data
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        # Draw detections
        for box, conf, cls in zip(boxes, confs, classes):
            # Scale coordinates to cell dimensions
            x1, y1, x2, y2 = (box * [scale_x, scale_y, scale_x, scale_y]).astype(int)

            # Create label
            class_name = results.names[int(cls)]
            label = f"{class_name}: {conf:.2f}"

            # Draw box and label
            ann.box_label([x1, y1, x2, y2], label=label, color=colors(int(cls), True))

        return frame

    def add_source_label(self, frame, source_id, show_stats=True):
        """Add source label to frame.

        Args:
            frame: Frame to annotate
            source_id: ID of the source
            show_stats: Whether to show the label

        Returns:
            Annotated frame
        """
        if not show_stats:
            return frame

        # Create label
        label = f"Source #{source_id}"

        # Get text dimensions
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        # Select color
        bg_color = self.source_colors[source_id % len(self.source_colors)]

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (5, 5),
            (5 + text_w + self.padding * 2, 5 + text_h + baseline + self.padding * 2),
            bg_color,
            -1
        )

        # Calculate text color based on background luminance
        r, g, b = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)

        # Draw text
        cv2.putText(
            frame,
            label,
            (5 + self.padding, 5 + self.padding + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            text_color,
            self.text_thickness
        )

        return frame

    def create_placeholder(self, stream_id):
        """Create placeholder frame for inactive streams.

        Args:
            stream_id: ID of the stream

        Returns:
            Placeholder frame
        """
        frame = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)

        # Create checkerboard pattern
        checker_size = 20
        for y in range(0, self.cell_h, checker_size):
            for x in range(0, self.cell_w, checker_size):
                if (x // checker_size + y // checker_size) % 2:
                    frame[y:y + checker_size, x:x + checker_size] = 30

        # Add "WAITING" text
        text = "WAITING"
        text_scale = max(0.5, min(1.0, self.cell_w / 300))
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)

        cv2.putText(
            frame,
            text,
            ((self.cell_w - w) // 2, (self.cell_h + h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (100, 100, 100),
            2
        )

        return frame


class FPSOverlay:
    """Optimized FPS overlay for the grid display."""

    def __init__(self, grid_width, grid_height):
        """Initialize FPS overlay.

        Args:
            grid_width: Total width of the grid
            grid_height: Total height of the grid
        """
        self.grid_w = grid_width
        self.grid_h = grid_height

        # Calculate optimal text size
        self.text_scale = max(0.6, min(1.2, grid_width / 1000))
        self.text_thickness = max(2, int(self.text_scale * 2))
        self.padding = max(15, int(self.text_scale * 10))

    def draw_fps(self, grid, fps_value):
        """Draw FPS overlay on the grid.

        Args:
            grid: Grid image to draw on
            fps_value: Current FPS value

        Returns:
            Grid with FPS overlay
        """
        if fps_value <= 0:
            return grid

        fps_text = f"FPS: {fps_value:.1f}"

        # Calculate text dimensions
        (text_w, text_h), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        # Position at bottom center
        center_x = (self.grid_w - text_w) // 2
        bottom_y = self.grid_h - self.padding

        # Draw background
        cv2.rectangle(
            grid,
            (center_x - self.padding, bottom_y - text_h - self.padding),
            (center_x + text_w + self.padding, bottom_y + self.padding),
            (0, 0, 0),  # Black background
            -1
        )

        # Draw border
        cv2.rectangle(
            grid,
            (center_x - self.padding, bottom_y - text_h - self.padding),
            (center_x + text_w + self.padding, bottom_y + self.padding),
            (255, 255, 255),  # White border
            2
        )

        # Draw text
        cv2.putText(
            grid,
            fps_text,
            (center_x, bottom_y - self.padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            (0, 255, 0),  # Green text
            self.text_thickness
        )

        return grid

    def draw_stream_stats(self, grid, stream_stats):
        """Draw stream statistics overlay.

        Args:
            grid: Grid image to draw on
            stream_stats: Dictionary of stream statistics

        Returns:
            Grid with statistics overlay
        """
        if not stream_stats:
            return grid

        # Calculate position (top-right corner)
        y_pos = 30
        x_pos = self.grid_w - 200

        # Background for stats
        cv2.rectangle(
            grid,
            (x_pos - 10, 10),
            (self.grid_w - 10, y_pos + len(stream_stats) * 25),
            (0, 0, 0),
            -1
        )

        # Draw stats for each stream
        for stream_id, stats in stream_stats.items():
            status = "🟢" if stats['active'] else "🔴"
            text = f"S{stream_id}: {status} {stats['fps']:.1f}fps"

            cv2.putText(
                grid,
                text,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            y_pos += 25

        return grid
