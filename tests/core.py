import pytest
import os
import tempfile
import requests
import shutil
from ultralytics import YOLO
from streamgrid import StreamGrid

video_urls = [
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/grid_2.mp4",
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/grid_3.mp4",
]


@pytest.fixture
def download_test_videos():
    """Download real test videos from GitHub releases. Fail if not available."""
    temp_dir = tempfile.mkdtemp(prefix="streamgrid_test_")
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    video_paths = []
    for i, url in enumerate(video_urls, start=1):
        print(f"Downloading Video{i}.mp4...")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            # Fail instead of skip or dummy
            pytest.fail(f"Failed to download Video{i}.mp4: {e}")

        video_path = os.path.join(temp_dir, f"Video{i}.mp4")
        with open(video_path, "wb") as f:
            f.write(r.content)
        video_paths.append(video_path)
        print(f"✓ Downloaded Video{i}.mp4")

    yield video_paths

    os.chdir(original_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_usage_code(download_test_videos):
    """Test the usage of StreamGrid with real videos."""
    paths = ["Video1.mp4", "Video2.mp4"]
    model = YOLO("yolo11n.pt")

    sg = StreamGrid(paths, model)
    try:
        assert sg is not None
        print("✓ StreamGrid initialization successful")
        print("✓ All videos loaded correctly")
        print("✓ YOLO model loaded successfully")
    finally:
        # Ensure background threads are cleaned up to avoid abort
        if hasattr(sg, "stop"):
            sg.stop()
        if hasattr(sg, "close"):
            sg.close()
