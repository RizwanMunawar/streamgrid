import torch
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


LOGGER = setup_logger(name="streamgrid")


def optimize(model_path="yolo11n.pt"):
    """Load YOLO11n with optimizations for 200 FPS on CPU.

    Args:
        model_path (str): Path to YOLO11n model file

    Returns:
        YOLO: Optimized YOLO model instance
    """

    # Load model with optimizations
    model = YOLO(model_path)

    # CPU optimizations
    model.model.eval()  # Set to evaluation mode

    # Disable unnecessary features
    model.model.training = False

    # Compile model for faster inference (PyTorch 2.0+)
    try:
        model.model = torch.compile(model.model, mode='max-autotune')
        LOGGER.info("Model compiled with torch.compile")
    except Exception as e:
        LOGGER.error(f"Could not compile model: {e}")

    LOGGER.info(f"🚀 model loaded and optimized for CPU.")
    return model
