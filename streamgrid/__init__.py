"""StreamGrid - Ultra-fast multi-stream video display."""

__version__ = "1.0.6"
__all__ = ["StreamGrid"]

import argparse
import sys
from ultralytics import YOLO
from .grid import StreamGrid


def parse_kv_args(args):
    """Parse key=value arguments into dict."""
    full_cmd = " ".join(args)
    import re

    config = {}
    kv_pairs = re.findall(r"(\w+)=([^=]+?)(?=\s+\w+=|$)", full_cmd)
    for k, v in kv_pairs:
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):  # Handle Python list literals
            import ast

            try:
                config[k] = ast.literal_eval(v)
                continue
            except:  # noqa: E722
                pass
        config[k] = {"true": True, "false": False}.get(
            v.lower(),  # Handle other types
            int(v) if v.isdigit() else float(v) if v.replace(".", "").isdigit() else v,
        )
    return config


def main():
    parser = argparse.ArgumentParser(description="StreamGrid")
    parser.add_argument("args", nargs="*", help="key=value pairs or source paths")
    config = parse_kv_args(parser.parse_args().args)
    sources = config.pop("sources", None)  # Process sources
    if sources and isinstance(sources, str):
        delimiter = (
            ";" if ";" in sources else ","
        )  # Support both comma and semicolon delimiters
        sources = [
            s.strip().strip("[]\"'") for s in sources.strip("[]").split(delimiter)
        ]

    model = None  # Load model
    if "model" in config and config["model"] != "none":
        try:
            model = YOLO(config.pop("model", "yolo11n.pt"))
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
