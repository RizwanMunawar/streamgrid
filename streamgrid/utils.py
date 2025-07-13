import torch
import os
from ultralytics import YOLO
import logging

import logging


def setup_logger(name, log_file=None):
    """Create a simple logger with console and optional file output."""
    logger = logging.getLogger(name)  # Create logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Create formatter

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


def optimize(model, size=320):
    """Load YOLO11n with optimizations for slightly better FPS on CPU."""
    # Standard eval mode
    model.model.eval()
    torch.set_grad_enabled(False)

    # Warmup
    dummy = torch.zeros((1, 3, size, size), dtype=torch.float32, device='cpu')
    model.predict(dummy, device='cpu', verbose=False)

    LOGGER.info(f"🚀 Model loaded and optimized for performance")
    return model


def get_optimal_grid_size(source_count, cols):
    """Get optimal cell size based on screen resolution and source count."""
    import math

    # Get screen resolution
    try:
        from screeninfo import get_monitors
        screen = get_monitors()[0]
        sw, sh = screen.width, screen.height
    except:
        sw, sh = 1920, 1080  # Default fallback

    cols, rows = int(math.ceil(math.sqrt(source_count))), int(math.ceil(source_count / cols))  # Calculate grid dim
    cw, ch = int(sw * 0.95) // cols, int(sh * 0.90) // rows  # Calculate cell size (with margins)

    # Maintain 16:9 aspect ratio
    if cw / ch > 16 / 9:
        cw = int(ch * 16 / 9)
    else:
        ch = int(cw * 9 / 16)

    return max(cw - (cw % 2), 320), max(ch - (ch % 2), 180)   # Ensure minimum size and even dimensions

LOGGER = setup_logger(name="streamgrid")
