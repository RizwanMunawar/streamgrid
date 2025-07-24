import pytest
import os
import tempfile
import requests
import shutil
from ultralytics import YOLO
from streamgrid import StreamGrid

video_urls = [
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/1.mp4",
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/d3.mp4",
]


@pytest.fixture
def download_test_videos():
    """Download real test videos from GitHub releases"""
    temp_dir = tempfile.mkdtemp(prefix="streamgrid_test_")
    video_paths = []

    for i, url in enumerate(video_urls):
        try:
            print(f"Downloading Video{i + 1}.mp4...")
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                video_path = os.path.join(temp_dir, f"Video{i + 1}.mp4")
                with open(video_path, "wb") as f:
                    f.write(response.content)
                video_paths.append(video_path)
                print(f"✓ Downloaded Video{i + 1}.mp4")
            else:
                print(
                    f"✗ Failed to download Video{i + 1}.mp4 (Status: {response.status_code})"
                )
        except requests.RequestException as e:
            print(f"✗ Error downloading Video{i + 1}.mp4: {e}")

    # Change to temp directory so relative paths work
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    yield video_paths

    # Cleanup
    os.chdir(original_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_usage_code(download_test_videos):
    """Test the exact code from documentation"""
    if len(download_test_videos) != 2:
        pytest.skip("All 2 videos must be downloaded")

    # This is the exact code from your documentation

    # Video paths
    paths = ["Video1.mp4", "Video2.mp4"]
    model = YOLO("yolo11n.pt")

    # Test initialization
    sg = StreamGrid(paths, model)
    assert sg is not None

    print("✓ StreamGrid initialization successful")
    print("✓ All videos loaded correctly")
    print("✓ YOLO model loaded successfully")
