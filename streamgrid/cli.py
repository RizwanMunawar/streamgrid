# cli.py
"""Simplified CLI interface."""

import argparse
import sys
from pathlib import Path
from .grid import StreamGrid


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="StreamGrid - Ultra-fast multi-stream video display")
    parser.add_argument('sources', nargs='+', help='Video sources')
    parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    parser.add_argument('--confidence', type=float, default=0.25, help='YOLO confidence')
    parser.add_argument('--model', type=str, help='YOLO model path')

    args = parser.parse_args()

    # Validate sources
    sources = []
    for src in args.sources:
        try:
            # Try camera index
            cam_id = int(src)
            if 0 <= cam_id <= 10:
                sources.append(cam_id)
                continue
        except ValueError:
            pass

        # Check file/URL
        if Path(src).exists() or src.startswith(('http', 'rtsp', 'rtmp')):
            sources.append(src)
        else:
            print(f"Warning: Skipping invalid source '{src}'")

    if not sources:
        print("Error: No valid sources found")
        return 1

    # Load YOLO model
    model = None
    if args.model:
        try:
            from ultralytics import YOLO
            model = YOLO(args.model)
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")

    # Create and run StreamGrid
    try:
        grid = StreamGrid(max_sources=len(sources))

        # Here you would add your video processing threads
        # (as shown in your example code)

        return grid.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())