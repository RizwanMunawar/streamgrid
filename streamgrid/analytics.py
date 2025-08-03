# StreamGrid - Analytics

import csv
import time
from datetime import datetime
from pathlib import Path


class StreamAnalytics:
    """Lightweight analytics logger for StreamGrid."""

    def __init__(self, output_file="streamgrid_analytics.csv"):
        self.output_file = Path(output_file)
        self.start_time = time.time()

        # Initialize CSV with headers
        with open(self.output_file, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "stream_id", "detections", "fps"])

        print(f"📊 Analytics: {self.output_file}")

    def log(self, stream_id, detections=0, fps=0.0):
        """Log frame data to CSV."""
        with open(self.output_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().strftime("%H:%M:%S"),
                    stream_id,
                    detections,
                    round(fps, 1),
                ]
            )

    def summary(self):
        """Print analytics summary."""
        uptime = time.time() - self.start_time
        print(f"📊 Runtime: {uptime:.1f}s | Data: {self.output_file}")
