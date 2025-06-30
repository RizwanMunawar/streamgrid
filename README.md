# StreamGrid ⚡

**Ultra-fast multi-stream video display** - Display multiple video sources simultaneously with minimal CPU usage. No AI processing, just pure speed and efficiency.

## 🚀 Features

- **⚡ Ultra-Fast**: Optimized for minimal CPU usage
- **📺 Multi-Stream**: Display 2-20+ video sources simultaneously  
- **🎯 Adaptive FPS**: Automatically adjusts FPS based on stream count
- **📐 Smart Grid**: Auto-arranges streams in optimal grid layout
- **🎮 Interactive**: Pause, reset, and control playback
- **🔧 Zero Config**: Works out of the box with smart defaults
- **🌐 Universal**: Supports files, cameras, RTSP, HTTP streams

## 📦 Installation

```bash
pip install streamgrid
```

## 🎯 Quick Start

### Command Line

```bash
# Display multiple video files
streamgrid video1.mp4 video2.mp4 video3.mp4

# Mix cameras and videos
streamgrid 0 1 video.mp4

# RTSP streams
streamgrid rtsp://192.168.1.100/stream rtsp://192.168.1.101/stream

# Custom settings
streamgrid *.mp4 --fps 15 --cell-size 640x360
```

### Python API

```python
from streamgrid import StreamGrid

# Simple usage
grid = StreamGrid(['video1.mp4', 'video2.mp4', 0])
grid.run()

# Advanced usage
grid = StreamGrid(
    sources=['video1.mp4', 'video2.mp4', 0, 1],
    fps=12,
    cell_size=(480, 270),
    window_title="My Streams",
    verbose=True
)
grid.run()
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `SPACE` | Pause/Resume |
| `R` | Reset streams |

## ⚙️ Configuration

### CLI Arguments

```bash
streamgrid [sources] [options]

sources              Video sources (files, URLs, camera indices)
--fps INTEGER        Target FPS per stream (auto-calculated)
--cell-size WxH      Cell size like 640x360 (auto-calculated)  
--title TEXT         Window title (default: StreamGrid)
--verbose, -v        Enable verbose output
```

### Python Parameters

```python
StreamGrid(
    sources,                    # List of video sources
    fps=None,                   # Target FPS (auto if None)
    cell_size=None,             # Cell size tuple (auto if None)
    window_title="StreamGrid",  # Window title
    verbose=False               # Enable verbose logging
)
```

## 📊 Performance

StreamGrid automatically optimizes performance:

- **1-2 streams**: 640x360 cells, up to 15 FPS each
- **3-4 streams**: 480x270 cells, up to 7 FPS each  
- **5-9 streams**: 320x180 cells, up to 5 FPS each
- **10+ streams**: 240x135 cells, up to 3 FPS each

## 🎯 Use Cases

- **Security Monitoring**: Display multiple camera feeds
- **Video Walls**: Create impressive multi-screen displays
- **Surveillance**: Monitor multiple locations simultaneously
- **Broadcasting**: Preview multiple video sources
- **Testing**: Debug multiple video streams at once

## 🔧 Advanced Examples

### Security Camera Setup
```bash
# 4 IP cameras
streamgrid rtsp://cam1/stream rtsp://cam2/stream rtsp://cam3/stream rtsp://cam4/stream
```

### Mixed Sources
```bash
# Local cameras + video files + network streams
streamgrid 0 1 recording.mp4 rtsp://192.168.1.100/stream
```

### High-Performance Setup
```bash
# Optimized for many streams
streamgrid cam*.mp4 --fps 5 --cell-size 320x180
```

## 📈 Performance Tips

1. **Reduce FPS**: Use `--fps 5` for many streams
2. **Smaller Cells**: Use `--cell-size 320x180` for 10+ streams  
3. **Close Other Apps**: Free up CPU resources
4. **Use SSD**: Faster disk access for video files
5. **Wired Network**: Better for IP camera streams

## 🛠️ Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.21+
- psutil 5.8+

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built with ❤️ using OpenCV and NumPy.