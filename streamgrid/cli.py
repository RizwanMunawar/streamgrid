"""
StreamGrid CLI - Ultra-optimized command line interface.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Union

from .grid import StreamGrid


def parse_sources(sources: List[str]) -> List[Union[str, int]]:
    """
    Parse and validate video sources.

    Args:
        sources: List of source strings

    Returns:
        List of validated sources (int for cameras, str for files/URLs)
    """
    valid_sources = []

    for src in sources:
        # Try camera index
        try:
            cam_id = int(src)
            if 0 <= cam_id <= 10:  # Reasonable camera range
                valid_sources.append(cam_id)
                continue
        except ValueError:
            pass

        # Check file/URL
        if (Path(src).exists() or
            src.startswith(('http://', 'https://', 'rtsp://', 'rtmp://', 'file://'))):
            valid_sources.append(src)
        else:
            print(f"Warning: Skipping invalid source '{src}'")

    return valid_sources


def main() -> int:
    """Main CLI entry point with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="StreamGrid - Ultra-fast multi-stream video display with YOLO detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  streamgrid 0 1                           # Two cameras
  streamgrid video1.mp4 video2.mp4         # Two video files
  streamgrid 0 video.mp4 rtsp://cam.ip     # Mixed sources
  streamgrid 0 1 --fps 15 --confidence 0.3 # Custom settings
        """
    )

    # Required arguments
    parser.add_argument(
        'sources',
        nargs='+',
        help='Video sources (camera indices, file paths, or URLs)'
    )

    # Optional arguments
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Target display FPS (default: 10)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='YOLO confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='YOLO batch size (default: 4)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to YOLO model file (optional)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate sources
    sources = parse_sources(args.sources)
    if not sources:
        print("Error: No valid sources found")
        return 1

    # Load YOLO model if specified
    model = None
    if args.model:
        try:
            # Try to import and load YOLO model
            from ultralytics import YOLO
            model = YOLO(args.model)
            print(f"Loaded YOLO model: {args.model}")
        except ImportError:
            print("Warning: ultralytics not installed, YOLO disabled")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")

    # Create and run StreamGrid
    try:
        grid = StreamGrid(
            sources=sources,
            fps=args.fps,
            model=model,
            confidence=args.confidence,
            batch_size=args.batch_size
        )

        return grid.run()

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())