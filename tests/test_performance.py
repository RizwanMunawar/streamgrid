"""Tests for benchmarking statistics and lossless capture behavior."""

import json
import time

from streamgrid.performance import PerformanceMetrics
from streamgrid.stream import StreamManager


def test_metrics_json_and_percentiles(tmp_path):
    metrics = PerformanceMetrics()
    metrics.record_frame(0, time.monotonic() - 0.01)
    metrics.record_frame(1, time.monotonic() - 0.02)
    metrics.record_batch(0.03)
    path = tmp_path / "metrics.json"
    report = metrics.export(path, decoded=[2, 2], dropped=[1, 0])
    assert report["processed_frames"] == 2
    assert report["batches"] == 1
    assert report["latency_p95_ms"] >= report["latency_p50_ms"]
    assert report["streams"]["0"]["dropped"] == 1
    assert json.loads(path.read_text())["processed_frames"] == 2


def test_fifo_has_no_implicit_overwrite():
    manager = StreamManager(["a"], buffer_mode="fifo", queue_size=2)
    assert manager.buffers[0].maxlen == 2
    assert manager.buffer_mode == "fifo"


def test_configuration_validation():
    import pytest

    with pytest.raises(ValueError):
        StreamManager(["a"], buffer_mode="unknown")
    with pytest.raises(ValueError):
        StreamManager(["a"], queue_size=0)
    with pytest.raises(ValueError):
        StreamManager(["a"], capture_fps=0)
