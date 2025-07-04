# StreamGrid ⚡

**Ultra-fast multi-stream video display** - Display multiple video sources with object detection simultaneously using the CPU 
or GPU device.

## Installation

```bash
pip install streamgrid
```

## Quick Start

### Python

```python
from ultralytics import YOLO
from streamgrid import StreamGrid

# Video paths
paths = ["Video1.mp4", "Video2.mp4", "Video3.mp4", "Video4.mp4"]

model = YOLO("yolo11n.pt")
StreamGrid(paths, model).run()

```

## Performance

StreamGrid automatically optimizes performance:

- **1-2 streams**: 640x360 cells, up to 15 FPS each
- **3-4 streams**: 480x270 cells, up to 7 FPS each  
- **5-9 streams**: 320x180 cells, up to 5 FPS each
- **10+ streams**: 240x135 cells, up to 3 FPS each

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
