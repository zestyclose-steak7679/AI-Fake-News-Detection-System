import logging
import os
from src.config import LOGS_DIR, PROJECT_LOG

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger writing to logs/project.log.

    Args:
        name (str): Name of the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        os.makedirs(LOGS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(PROJECT_LOG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False
    return logger
