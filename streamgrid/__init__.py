"""StreamGrid - Ultra-fast multi-stream video display."""

__version__ = "1.0.5"

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from .grid import StreamGrid
from .utils import optimize

__all__ = ["StreamGrid"]

def parse_kv_args(args):
    """Parse key=value arguments into dict."""
    config = {}
    for arg in args:
        if '=' in arg:
            k, v = arg.split('=', 1)
            config[k] = {'true': True, 'false': False}.get(v.lower(), int(v) if v.isdigit() else float(v)
            if v.replace('.', '').isdigit() else v)
        else:
            config.setdefault('sources', []).append(int(arg) if arg.isdigit() else arg)
    return config


def main():
    parser = argparse.ArgumentParser(description="StreamGrid")
    parser.add_argument('args', nargs='*', help='key=value pairs i.e device="cuda"')
    config = parse_kv_args(parser.parse_args().args)
    sources = config.pop('sources', None)  # Process sources
    if sources and isinstance(sources, str):
        sources = [s.strip() for s in sources.split(',')]

    model = None  # Load model
    if 'model' in config and config['model'] != 'none':
        try:
            model = YOLO(config.pop('model', 'yolo11n.pt'))
        except Exception as e:
            print(f"Model error: {e}")
            sys.exit(1)

    try:  # Run StreamGrid
        StreamGrid(sources=sources, model=model, **config)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
