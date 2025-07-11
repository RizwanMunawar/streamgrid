# StreamGrid ⚡

**Ultra-fast multi-stream video display** - Display multiple video sources with object detection simultaneously using the CPU 
or GPU device.

[![Run tests](https://github.com/RizwanMunawar/streamgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanMunawar/streamgrid/actions/workflows/ci.yml)
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
