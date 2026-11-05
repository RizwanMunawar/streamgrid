"""Buffered analytics CSV logging to avoid one file open per frame."""

import csv
import time
from datetime import datetime
from pathlib import Path


class StreamAnalytics:
    def __init__(self, output_file="streamgrid_analytics.csv", flush_every=100):
        self.output_file = Path(output_file)
        self.started = time.monotonic()
        self.flush_every = max(1, int(flush_every))
        self.file = self.output_file.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "stream_id", "detections", "fps"])
        self.pending = 0

    def log(self, stream_id, detections=0, fps=0.0):
        self.writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            stream_id, detections, round(fps, 1)
        ])
        self.pending += 1
        if self.pending >= self.flush_every:
            self.file.flush()
            self.pending = 0

    def summary(self):
        self.close()
        print(f"Analytics: {self.output_file} | Runtime: {time.monotonic() - self.started:.1f}s")

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()
