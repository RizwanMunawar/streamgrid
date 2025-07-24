# StreamGrid - Run Object Detection Across Multiple Streams

import csv
import time
from datetime import datetime
from pathlib import Path


class StreamAnalytics:
    """Simple analytics logger for StreamGrid."""

    def __init__(self, output_file="streamgrid_analytics.csv"):
        """Initialization method."""
        self.output_file = Path(output_file)
        self.start_time = time.time()

        # Create CSV with headers
        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "stream_id", "detections", "fps"])

        print(f"📊 Analytics: {self.output_file}")

    def log(self, stream_id, detections=0, fps=0.0):
        """Log frame data."""
        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    stream_id,
                    detections,
                    round(fps, 1),
                ]
            )

    def summary(self):
        """Print summary."""
        uptime = time.time() - self.start_time
        print(f"📊 Runtime: {uptime:.1f}s | Data saved: {self.output_file}")
