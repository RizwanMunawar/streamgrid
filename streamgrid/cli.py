"""StreamGrid CLI"""

import argparse
import sys
from pathlib import Path

from .streamgrid import StreamGrid


def main():
    """Main CLI."""
    parser = argparse.ArgumentParser(description="StreamGrid - Multi-stream video display")
    parser.add_argument('sources', nargs='+', help='Video sources')
    parser.add_argument('--fps', type=int, default=10, help='FPS (default: 10)')

    args = parser.parse_args()

    # Parse sources
    sources = []
    for src in args.sources:
        try:
            sources.append(int(src))  # Camera
        except ValueError:
            if Path(src).exists() or src.startswith(('http', 'rtsp')):
                sources.append(src)  # File/URL

    if not sources:
        print("No valid sources found")
        return 1

    # Run StreamGrid
    grid = StreamGrid(sources, fps=args.fps)
    grid.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
