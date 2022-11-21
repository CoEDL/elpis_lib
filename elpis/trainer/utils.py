import logging
from contextlib import contextmanager
from logging.handlers import WatchedFileHandler
from pathlib import Path

import transformers
from loguru import logger


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
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_propagation()

    try:
        yield log_file
    finally:
        # Flush and teardown logging handlers
        logger.remove(sink_id)
        handler.flush()
        logging.root.removeHandler(handler)
