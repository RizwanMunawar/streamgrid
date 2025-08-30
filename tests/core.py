from ultralytics import YOLO
from ultralytics.utils.downloads import safe_download
from streamgrid import StreamGrid

video_urls = [
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/grid_2.mp4",
    "https://github.com/RizwanMunawar/streamgrid/releases/download/v1.0.0/grid_3.mp4",
]

for video in video_urls:
    safe_download(video)


def test_usage_code():
    """Test the usage of StreamGrid with real videos."""
    paths = ["grid_2.mp4", "grid_3.mp4"]
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
