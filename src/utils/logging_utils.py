# src/utils/logging_utils.py
import logging
import os

def setup_logging(log_dir="logs", filename="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Example usage:
# logger = setup_logging()
# logger.info("This is an info message.")
