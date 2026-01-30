import logging
import os
import warnings
from datetime import datetime


def setup_logger(logger=None, name="indra", log_dir="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    if logger is None:
        logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"{log_dir}/indra_{timestamp}.log"
    handler = logging.FileHandler(logfile)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    # Route warnings through logging to the same file.
    warnings.simplefilter("default")
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    if not warn_logger.handlers:
        warn_logger.addHandler(handler)
    warn_logger.setLevel(level)
    warn_logger.propagate = False

    return logger


def get_logger(logger, name="indra"):
    logger_out = logger if logger is not None else logging.getLogger(name)
    if not logger_out.handlers:
        setup_logger(logger_out, name=name)
    return logger_out
