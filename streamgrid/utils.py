import torch
import os
from ultralytics import YOLO
import logging

import logging


def setup_logger(name, log_file=None):
    """
    Create a simple logger with console and optional file output.

    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file, if None logs to console only

    Returns:
        logging.Logger: Configured logger instance

    Example:
        logger = setup_logger("my_app")
        logger = setup_logger("my_app", "app.log")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

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
    """Load YOLO11n with optimizations for 200 FPS on CPU.

    Args:
        model (YOLO): Ultralytics YOLO initialized model

    Returns:
        YOLO: Optimized YOLO model instance
    """
    # Standard eval mode
    model.model.eval()
    torch.set_grad_enabled(False)

    # Warmup
    dummy = torch.zeros((1, 3, size, size), dtype=torch.float32, device='cpu')
    model.predict(dummy, device='cpu', verbose=False)

    # # Try to enable Intel MKL optimizations if available
    # try:
    #     import intel_extension_for_pytorch as ipex
    #     LOGGER.info("✅ Intel extension for PyTorch (IPEX) enabled")
    #     ipex_available = True
    # except ImportError:
    #     ipex_available = False
    #
    #
    # model.model.eval()
    # model.model.training = False
    # torch.set_grad_enabled(False)  # Disable gradient calculation
    # torch.set_num_threads(min(8, torch.get_num_threads()))  # Optimal thread count
    #
    # # Set optimal CPU threads
    # physical_cores = os.cpu_count() // 2 if os.cpu_count() > 1 else 1
    # torch.set_num_threads(min(8, physical_cores))
    #
    # input_size = 320
    #
    # # Enable fast inplace operations
    # for m in model.model.modules():
    #     if hasattr(m, 'inplace'):
    #         m.inplace = True
    #
    # # ================= ADVANCED OPTIMIZATIONS =================
    # # Optimize model parameters precision
    # for param in model.model.parameters():
    #     # Store parameters in lower precision for faster computation
    #     if param.data.dtype == torch.float32 and not ipex_available:
    #         param.data = param.data.to(torch.float16).to('cpu').to(torch.float32)
    #
    # # Try Intel IPEX optimization if available
    # if ipex_available:
    #     try:
    #         # Optimize with Intel's extension
    #         model.model = ipex.optimize(model.model)
    #         LOGGER.info("✅ Model optimized with Intel IPEX")
    #     except Exception as e:
    #         LOGGER.warning(f"Intel IPEX optimization failed: {e}")
    #
    # # Try PyTorch 2.0+ compile
    # try:
    #     # Use most aggressive compilation mode
    #     model.model = torch.compile(
    #         model.model,
    #         mode='reduce-overhead',
    #         fullgraph=True,
    #         dynamic=False
    #     )
    #     LOGGER.info("✅ Model compiled with torch.compile")
    # except Exception as e:
    #     LOGGER.warning(f"Model compilation failed: {e}")
    #
    # # Warmup with CPU tensor at target size
    # try:
    #     # Create proper warmup tensor at target input size
    #     dummy = torch.zeros((1, 3, input_size, input_size),
    #                         dtype=torch.float32,
    #                         device='cpu')
    #
    #     # Force CPU prediction with optimized settings
    #     for _ in range(2):  # Multiple warmup runs for better cache priming
    #         results = model.predict(
    #             source=dummy,
    #             verbose=False,
    #             device='cpu',
    #             half=False,
    #             conf=0.35,
    #             iou=0.45,
    #             agnostic_nms=True,
    #             max_det=100
    #         )
    #     LOGGER.info(f"✅ Model warmup complete at {input_size}px input size")
    # except Exception as e:
    #     LOGGER.warning(f"Model warmup failed: {e}")

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

    cw, ch = max(cw - (cw % 2), 320), max(ch - (ch % 2), 180)  # Ensure minimum size and even dimensions
    return cw, ch

LOGGER = setup_logger(name="streamgrid")