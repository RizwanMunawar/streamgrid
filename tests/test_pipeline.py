"""CPU-only regression tests for per-stream buffering and lifecycle."""

from unittest.mock import patch

import numpy as np

from streamgrid.stream import StreamManager


def test_latest_frame_replaces_stale_frames():
    manager = StreamManager(["cam0", "cam1"], buffer_mode="latest")
    with manager.condition:
        for index in range(5):
            manager.buffers[0].append((0, np.full((2, 2, 3), index), float(index)))
        manager.buffers[1].append((1, np.zeros((2, 2, 3)), 1.0))
    result = manager.get_frames()
    assert [item[0] for item in result] == [0, 1]
    assert result[0][2] == 4.0


def test_fifo_capacity_and_round_robin():
    manager = StreamManager(["a", "b"], buffer_mode="fifo", queue_size=2)
    with manager.condition:
        for value in range(3):
            manager.buffers[0].append((0, value, float(value)))
        manager.buffers[1].append((1, "b", 0.0))
    assert [item[1] for item in manager.get_frames()] == [1, "b"]
    assert manager.get_frames()[0][1] == 2


def test_finished_requires_all_sources_end_and_queues_empty():
    manager = StreamManager(["a"])
    manager.ended[0] = True
    manager.buffers[0].append((0, "frame", 0.0))
    assert not manager.finished
    manager.get_frames()
    assert manager.finished


def test_capture_has_no_fixed_sleep_and_releases_camera():
    class FakeCapture:
        def __init__(self):
            self.count = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.count += 1
            return (
                (True, np.zeros((2, 2, 3), dtype=np.uint8))
                if self.count <= 3
                else (False, None)
            )

        def release(self):
            self.released = True

    camera = FakeCapture()
    manager = StreamManager(["cam"], buffer_mode="latest")
    manager.running = True
    with patch("streamgrid.stream.cv2.VideoCapture", return_value=camera):
        manager.capture_stream("cam", 0)
    assert camera.released
    assert manager.decoded == [3]
    assert manager.dropped == [2]
    assert manager.finished is False
    item = manager.get_frames()[0]
    assert item[0] == 0
    assert manager.finished
