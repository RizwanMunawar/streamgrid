# StreamGrid - Run Object Detection Across Multiple Streams

import torch
import logging


def setup_logger(name, log_file=None):
    """Create a simple logger with console and optional file output."""
    logger = logging.getLogger(name)  # Create logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )  # Create formatter

    # Add console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_optimal_grid_size(source_count, cols):
    """Get optimal cell size based on screen resolution and source count."""
    import math

    # Get screen resolution
    try:
        from screeninfo import get_monitors

        screen = get_monitors()[0]
        sw, sh = screen.width, screen.height
    except:  # noqa: E722
        sw, sh = 1920, 1080  # Default fallback

    cols, rows = int(math.ceil(math.sqrt(source_count))), int(
        math.ceil(source_count / cols)
    )  # Calculate grid dim
    cw, ch = (
        int(sw * 0.95) // cols,
        int(sh * 0.90) // rows,
    )  # Calculate cell size (with margins)

    # Maintain 16:9 aspect ratio
    if cw / ch > 16 / 9:
        cw = int(ch * 16 / 9)
    else:
        ch = int(cw * 9 / 16)

    return max(cw - (cw % 2), 320), max(
        ch - (ch % 2), 180
    )  # Ensure minimum size and even dimensions


LOGGER = setup_logger(name="streamgrid")
