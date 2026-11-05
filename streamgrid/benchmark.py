"""Run comparable headless multi-video inference benchmarks.

Example:
    python -m streamgrid.benchmark --sources a.mp4 b.mp4 --model yolo11n.pt \
        --batch-size 4 --buffer-mode fifo --output results.json
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from .grid import StreamGrid


def main(argv=None):
    parser = argparse.ArgumentParser(description="StreamGrid reproducible benchmark")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--model", default="none",
                        help="YOLO checkpoint or exported ONNX/OpenVINO/TensorRT model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--inference-fps", type=float, default=30)
    parser.add_argument("--buffer-mode", choices=["fifo", "latest"], default="fifo")
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default="streamgrid_benchmark.json")
    args = parser.parse_args(argv)

    model = YOLO(args.model) if args.model.lower() != "none" else None
    target = Path(args.output)
    started = time.monotonic()
    grid = StreamGrid(
        sources=args.sources, model=model, device=args.device,
        buffer_mode=args.buffer_mode, queue_size=args.queue_size,
        batch_size=args.batch_size, inference_fps=args.inference_fps,
        imgsz=args.imgsz, half=args.half,
        show=False, save=False, metrics_file=str(target)
    )
    if grid.error:
        raise RuntimeError("Inference failed") from grid.error
    report = json.loads(target.read_text(encoding="utf-8"))
    report["configuration"] = {
        "sources": args.sources,
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size or len(args.sources),
        "inference_fps_limit": args.inference_fps,
        "buffer_mode": args.buffer_mode,
        "half": args.half
    }
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
