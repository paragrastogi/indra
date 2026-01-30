import numpy as np
from .logging_utils import get_logger
"""
Calculate loss functions for incoming vectors of error.
Add any error functions below - just remember to explicitly
import them in the script you are calling them from.
"""


def rmseloss(x1, *args, logger=None):
    logger = get_logger(logger)

    if len(args) > 1:
        x2 = args[1]
        e = x1 - x2
    else:
        e = x1

    result = np.sqrt((e**2).mean())
    logger.info("rmseloss success")
    return result


def maeloss(e, logger=None):
    logger = get_logger(logger)
    result = np.abs(e.mean())
    logger.info("maeloss success")
    return result
