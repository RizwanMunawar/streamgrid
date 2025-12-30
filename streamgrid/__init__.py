"""
StreamGrid - Ultra-fast multi-stream video display.

This module provides a command-line interface (CLI) for launching the
StreamGrid application, enabling real-time visualization of multiple
video streams in a grid layout.

Key Features:
- Supports multiple video sources (RTSP, files, webcams, URLs)
- Optional integration with Ultralytics YOLO models for inference
- Flexible configuration using key=value CLI arguments
- Robust parsing of lists, booleans, integers, and floats

Example:
    python -m streamgrid sources=cam1.mp4,cam2.mp4 model=yolo11n.pt imgsz=640
"""

__version__ = "1.0.10"
__all__ = ["StreamGrid"]

import argparse
import sys
import re
import ast
from ultralytics import YOLO
from .grid import StreamGrid


def parse_args(args):
    """
    Parse CLI arguments provided as key=value pairs into a dictionary.

    This function supports automatic type inference for:
    - Lists (e.g., [1, 2, 3])
    - Booleans (true / false)
    - Integers
    - Floats
    - Strings (default)

    Args:
        args (list[str]): List of command-line arguments in key=value format.

    Returns:
        dict: Parsed configuration dictionary with inferred value types.

    Example:
        Input:
            ["sources=[cam1.mp4,cam2.mp4]", "imgsz=640", "show=true"]

        Output:
            {
                "sources": ["cam1.mp4", "cam2.mp4"],
                "imgsz": 640,
                "show": True
            }
    """
    config = {}
    kv_pairs = re.findall(r"(\w+)=([^=]+?)(?=\s+\w+=|$)", " ".join(args))

    for k, v in kv_pairs:
        v = v.strip()

        # Handle lists
        if v.startswith("[") and v.endswith("]"):
            try:
                config[k] = ast.literal_eval(v)
                continue
            except Exception:
                pass

        # Handle booleans and numbers
        if v.lower() in ("true", "false"):
            config[k] = v.lower() == "true"
        elif v.isdigit():
            config[k] = int(v)
        elif v.replace(".", "").isdigit():
            config[k] = float(v)
        else:
            config[k] = v

    return config


def main():
    """
    StreamGrid CLI entry point.

    This function:
    1. Parses command-line arguments
    2. Processes video sources
    3. Loads an optional YOLO model
    4. Initializes and runs the StreamGrid application

    Exit Conditions:
    - Gracefully exits on keyboard interrupt (Ctrl+C)
    - Prints error messages and exits on runtime failures

    CLI Usage:
        streamgrid sources=cam1.mp4,cam2.mp4 model=yolo11n.pt

    Notes:
        - Use `model=none` to disable inference
        - Multiple sources can be separated by commas or semicolons
    """
    parser = argparse.ArgumentParser(description="StreamGrid")
    parser.add_argument("args", nargs="*", help="key=value pairs")
    config = parse_args(parser.parse_args().args)

    # Process sources
    sources = config.pop("sources", None)
    if sources and isinstance(sources, str):
        delimiter = ";" if ";" in sources else ","
        sources = [
            s.strip().strip("[]\"'") for s in sources.strip("[]").split(delimiter)
        ]

    # Load model
    model = None
    if "model" in config and config["model"] != "none":
        try:
            model = YOLO(config.pop("model", "yolo11n.pt"))
        except Exception as e:
            print(f"Model error: {e}")
            sys.exit(1)

    # Run StreamGrid
    try:
        StreamGrid(sources=sources, model=model, **config)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
