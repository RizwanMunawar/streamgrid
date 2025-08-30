"""StreamGrid - Ultra-fast multi-stream video display."""

__version__ = "1.0.9"
__all__ = ["StreamGrid"]

import argparse
import sys
import re
import ast
from ultralytics import YOLO
from .grid import StreamGrid


def parse_args(args):
    """Parse key=value arguments into dict."""
    config = {}
    kv_pairs = re.findall(r"(\w+)=([^=]+?)(?=\s+\w+=|$)", " ".join(args))

    for k, v in kv_pairs:
        v = v.strip()
        # Handle lists
        if v.startswith("[") and v.endswith("]"):
            try:
                config[k] = ast.literal_eval(v)
                continue
            except:  # noqa
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
    """StreamGrid CLI entry point."""
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
