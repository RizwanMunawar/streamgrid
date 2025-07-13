# StreamGrid ⚡

Ultra-fast multi-stream video display, Run real-time object detection across multiple video feeds with real-time speed. Whether you're on CPU or GPU, StreamGrid handles the load like a champ.

From smart surveillance to AI-powered dashboards, StreamGrid makes it effortless to display and analyze multiple video streams side-by-side, with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO-based detection running on all of them in real time.

[![Run CI](https://github.com/RizwanMunawar/streamgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanMunawar/streamgrid/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/badge/streamgrid)](https://pepy.tech/projects/streamgrid)
[![PyPI version](https://img.shields.io/pypi/v/streamgrid.svg)](https://pypi.org/project/streamgrid/)

## Installation

```bash
pip install streamgrid
```

## Quick Start

### Python

```python
from ultralytics import YOLO
from streamgrid import StreamGrid

# Use assets videos for testing
model = YOLO("yolo11n.pt")
StreamGrid(model=model)  

# Use your own videos
sources = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
StreamGrid(sources=sources, model=model)

```

## Performance (Beta, final benchmarks will be released soon)

StreamGrid automatically optimizes performance based on the number of streams:

- **1-2 streams**: 640×360 resolution, up to 15 FPS per stream
- **3-4 streams**: 480×270 resolution, up to 10 FPS total (CPU processing)
- **5-9 streams**: 320×180 resolution, up to 5 FPS per stream
- **10+ streams**: 240×135 resolution, up to 3 FPS per stream

*Note: Performance benchmarks are based on CPU processing. GPU acceleration can significantly improve throughput.*

## Contributing

We welcome contributions! Please feel free to submit a Pull Request or open an issue for discussion.
