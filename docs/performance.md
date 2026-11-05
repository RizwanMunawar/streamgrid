# StreamGrid Performance Guide

StreamGrid supports two distinct workloads:

- **Real-time monitoring** (default): `buffer_mode="latest"` keeps a bounded newest frame for each camera. Older images can be dropped deliberately to prioritize freshness.
- **Offline / lossless analysis**: `buffer_mode="fifo"` uses bounded per-source queues and applies backpressure so the input reader waits for the inference worker instead of overwriting frames.

## Running StreamGrid

```python
from ultralytics import YOLO
from streamgrid import StreamGrid

StreamGrid(
    sources=["cam1.mp4", "cam2.mp4"],
    model=YOLO("yolo11n.pt"),
    device="cpu",
    batch_size=2,
    inference_fps=30,
    display_fps=15,
    buffer_mode="latest",
    save=False,
)
```

Set `show=False` to disable the desktop window. Use `render=False` only when neither display nor recording is needed; it disables overlay and grid updates to isolate model throughput. Set `inference_fps=0` for an uncapped inference loop.

### Inference export backends

The package delegates inference to Ultralytics `YOLO(...).predict()`. You can provide an exported model that is supported by your installed Ultralytics version:

```python
# Native PyTorch
YOLO("yolo11n.pt")
# Exported ONNX
YOLO("yolo11n.onnx")
# OpenVINO (directory)
YOLO("yolo11n_openvino_model/")
# TensorRT (NVIDIA GPU / appropriate runtime)
YOLO("yolo11n.engine")
```

Export separately using `YOLO("yolo11n.pt").export(format="onnx")`, `format="openvino"`, or `format="engine"`. Hardware and dependency support vary. An exported model can have different preprocessing and numerical characteristics, so validate accuracy before comparing runtimes. `half=True` is intended only for compatible inference devices and backends.

## Reproducible benchmark CLI

```bash
python -m streamgrid.benchmark --sources cam1.mp4 cam2.mp4 --model yolo11n.pt --device cpu --batch-size 2 --output pytorch.json
python -m streamgrid.benchmark --sources cam1.mp4 cam2.mp4 --model yolo11n.onnx --device cpu --batch-size 2 --output onnx.json
python -m streamgrid.benchmark --sources cam1.mp4 cam2.mp4 --model yolo11n_openvino_model --device cpu --batch-size 2 --output openvino.json
```

Each report contains elapsed wall time, processed images, overall FPS, per-source FPS, decoded and dropped frames, mean batch duration, and P50/P95 capture-to-processed frame age. JSON also includes model, resolution, batch size, runtime device and buffering mode. These metrics are not interchangeable:

- **Total FPS** = frames processed / wall-clock time, including startup/teardown.
- **Frame age** = monotonic time when processing of a frame is complete minus monotonic capture time. It is *not* camera sensor-to-screen latency.
- **Mean batch duration** includes postprocessing and optional rendering, but is not an isolated model-kernel timing.
- **Dropped** counts frames overwritten within latest-frame buffers. Use FIFO to ensure no intentional buffer overwrites.
- **Per-stream FPS** = processed frames from source / global runtime. It helps reveal starvation.

For meaningful comparisons, run at least three times after warming the hardware, use the same camera videos and `imgsz`, keep batch sizes fixed, report median FPS and P95 latency, and compare CPU/GPU utilization externally using OS tools or `nvidia-smi`. Do not report speedup claims unless the baseline and experimental runs are both reproducible.

## Tuning notes

- Increasing batch size may improve *throughput* but can increase per-stream latency and memory consumption.
- Live RTSP sources should generally use `latest`, not lossless FIFO; the latter can create growing latency upstream.
- Configure `capture_fps` if source decoding uses excessive CPU. Avoid treating this as an inference optimization.
- `display_fps` and video-output FPS are independently limited from inference. Saved videos record the latest processed tiles at the display cadence.
- Exported models may need backend-specific dependencies. TensorRT is only appropriate for supported NVIDIA devices and drivers.
- Hardware accelerated decoding, pinned-memory preprocessing, and automatic model-specific batch-size tuning require hardware-specific profiling and are **not** enabled by this change.
