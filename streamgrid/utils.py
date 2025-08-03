# StreamGrid - Utilities

import math
import logging


def get_optimal_grid_size(source_count, cols):
    """Get optimal cell size based on screen resolution and source count."""
    # Get screen resolution
    try:
        from screeninfo import get_monitors

        sw, sh = get_monitors()[0].width, get_monitors()[0].height
    except:  # noqa
        sw, sh = 1920, 1080  # Default fallback

    rows = int(math.ceil(source_count / cols))
    cw, ch = int(sw * 0.95) // cols, int(sh * 0.90) // rows

    # Maintain 16:9 aspect ratio
    if cw / ch > 16 / 9:
        cw = int(ch * 16 / 9)
    else:
        ch = int(cw * 9 / 16)

    return max(cw - (cw % 2), 320), max(ch - (ch % 2), 180)


# Simple logger setup
LOGGER = logging.getLogger("streamgrid")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)
