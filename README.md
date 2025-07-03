# StreamGrid ⚡

**Ultra-fast multi-stream video display** - Display multiple video sources simultaneously with minimal CPU usage.

## Installation

```bash
pip install streamgrid
```

## Quick Start

### Command Line

```bash
# Display multiple video files
streamgrid video1.mp4 video2.mp4 video3.mp4

# Mix cameras and videos
streamgrid 0 1 video.mp4

# RTSP streams
streamgrid rtsp://192.168.1.100/stream rtsp://192.168.1.101/stream
```

### Python API

```python
from streamgrid import StreamGrid

# Simple usage
grid = StreamGrid(['video1.mp4', 'video2.mp4', 0])
grid.run()
```

## Performance

StreamGrid automatically optimizes performance:

- **1-2 streams**: 640x360 cells, up to 15 FPS each
- **3-4 streams**: 480x270 cells, up to 7 FPS each  
- **5-9 streams**: 320x180 cells, up to 5 FPS each
- **10+ streams**: 240x135 cells, up to 3 FPS each

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
