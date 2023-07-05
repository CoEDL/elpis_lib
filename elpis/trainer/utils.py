import logging
import sys
from contextlib import contextmanager
from logging.handlers import WatchedFileHandler
from pathlib import Path

from loguru import logger
from transformers.utils import logging as transformer_logging


@contextmanager
def log_to_file(log_file: Path):
    """A context manager which logs its captured runtime to the provided file."""
    sink_id = logger.add(log_file)

    handler = WatchedFileHandler(str(log_file))
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)s | (HF) %(name)s |   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logging.root.addHandler(handler)

    # Propagate huggingface logs to logging root.
    transformer_logging.set_verbosity_info()
    transformer_logging.enable_propagation()
    transformer_logging.disable_default_handler()

    # Log stderr to log file to capture tqdm logs
    with open(log_file, "a") as stderr_hole:
        original_stderr = sys.stderr
        try:
            sys.stderr = stderr_hole
            yield log_file
        finally:
            sys.stderr = original_stderr
            # Flush and teardown logging handlers
            logger.remove(sink_id)
            handler.flush()
            logging.root.removeHandler(handler)
