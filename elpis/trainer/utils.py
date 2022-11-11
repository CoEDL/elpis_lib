import logging
from functools import wraps
from logging.handlers import WatchedFileHandler
from pathlib import Path
from typing import Callable, Optional

import transformers
from loguru import logger


def log_to_file(log_file: Optional[Path]) -> Callable[[Callable], Callable]:
    """Returns a decorator that will output the logs of a captured function
    to a log file, if it exists.

    Args:
        log_file: The optional file to log to.
    """

    def log_wrapper(func: Callable) -> Callable:
        @wraps(func)
        def function_wrapper(*args, **kwargs):
            if log_file is None:
                return func(*args, **kwargs)

            sink_id = logger.add(log_file)

            # Configure huggingface logging handler
            formatter = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d | %(levelname)s | (HF) %(name)s |   %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler = WatchedFileHandler(str(log_file))
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)

            logging.root.addHandler(handler)
            transformers.logging.set_verbosity_info()
            transformers.logging.enable_propagation()

            result = func(*args, **kwargs)
            # Flush and teardown logging handlers
            logger.remove(sink_id)
            handler.flush()
            logging.root.removeHandler(handler)
            return result

        return function_wrapper

    return log_wrapper
