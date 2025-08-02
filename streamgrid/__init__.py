"""StreamGrid - Ultra-fast multi-stream video display."""

__version__ = "1.0.7"
__all__ = ["StreamGrid"]

import argparse
import sys
from ultralytics import YOLO
from .grid import StreamGrid


def parse_cli_args(args):
    """Parse command line arguments efficiently.

    Args:
        args: List of command line arguments

    Returns:
        dict: Parsed configuration
    """
    config = {}

    # Simple key=value parsing
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Handle different value types
            if value.lower() in ('true', 'false'):
                config[key] = value.lower() == 'true'
            elif value.isdigit():
                config[key] = int(value)
            elif value.replace('.', '').isdigit():
                config[key] = float(value)
            elif value.startswith('[') and value.endswith(']'):
                # Handle lists: [item1,item2,item3]
                import ast
                try:
                    config[key] = ast.literal_eval(value)
                except:
                    # Fallback: split by comma and clean
                    items = value[1:-1].split(',')
                    config[key] = [item.strip().strip('"\'') for item in items if item.strip()]
            else:
                config[key] = value.strip('"\'')

    return config


def main():
    parser = argparse.ArgumentParser(description="StreamGrid")
    parser.add_argument(
        "args",
        nargs="*",
        help="Configuration in key=value format"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    config = parse_cli_args(parsed_args.args)

    # Extract and process sources
    sources = config.pop("sources", None)

    # Load model if specified
    model = None
    model_path = config.pop("model", None)
    if model_path and model_path.lower() != "none":
        try:
            model = YOLO(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model '{model_path}': {e}")
            return 1

    # Validate sources
    if sources is None:
        print("⚠️ No sources specified. Using default webcam (source 0).")
        sources = [0]
    elif isinstance(sources, str):
        sources = [sources]

    try:
        # Run StreamGrid
        StreamGrid(sources=sources, model=model, **config)
        return 0

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
