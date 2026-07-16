import logging
import os

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger writing to logs/project.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/project.log")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        # Avoid printing to stdout as per constraints
        logger.propagate = False
    return logger