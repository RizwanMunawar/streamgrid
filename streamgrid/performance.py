"""Streaming throughput and latency counters with JSON-friendly snapshots."""

import json
import statistics
import threading
import time
from collections import defaultdict, deque
from pathlib import Path


class PerformanceMetrics:
    """Track wall-clock throughput and frame-age percentiles independently."""

    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.processed = defaultdict(int)
        self.latencies_ms = deque(maxlen=10000)
        self.batch_durations_ms = deque(maxlen=10000)
        self.batches = 0

    def record_batch(self, duration_seconds):
        with self.lock:
            self.batch_durations_ms.append(duration_seconds * 1000)
            self.batches += 1

    def record_frame(self, source_id, capture_timestamp=None):
        with self.lock:
            self.processed[source_id] += 1
            if capture_timestamp is not None:
                self.latencies_ms.append(
                    max(0, (time.monotonic() - capture_timestamp) * 1000)
                )

    def snapshot(self, decoded=None, dropped=None):
        with self.lock:
            elapsed = max(time.monotonic() - self.started, 1e-9)
            per_stream = {
                str(k): {"processed": v, "fps": round(v / elapsed, 2)}
                for k, v in sorted(self.processed.items())
            }
            latencies = sorted(self.latencies_ms)
            durations = list(self.batch_durations_ms)
            count = sum(self.processed.values())

        def percentile(p):
            if not latencies:
                return None
            position = (len(latencies) - 1) * p
            low = int(position)
            high = min(low + 1, len(latencies) - 1)
            return round(
                latencies[low] + (latencies[high] - latencies[low]) * (position - low),
                2,
            )

        if decoded is not None:
            for i, n in enumerate(decoded):
                per_stream.setdefault(str(i), {"processed": 0, "fps": 0.0})[
                    "decoded"
                ] = n
        if dropped is not None:
            for i, n in enumerate(dropped):
                per_stream.setdefault(str(i), {"processed": 0, "fps": 0.0})[
                    "dropped"
                ] = n
        return {
            "elapsed_seconds": round(elapsed, 3),
            "processed_frames": count,
            "total_fps": round(count / elapsed, 2),
            "latency_p50_ms": percentile(0.5),
            "latency_p95_ms": percentile(0.95),
            "mean_batch_ms": (
                round(statistics.mean(durations), 2) if durations else None
            ),
            "batches": self.batches,
            "streams": per_stream,
        }

    def export(self, path, decoded=None, dropped=None):
        report = self.snapshot(decoded, dropped)
        Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
