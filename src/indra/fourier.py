import numpy as np
from .logging_utils import get_logger


def fit(fstr, x, *args, logger=None):
    logger = get_logger(logger)
    if fstr == 'tdb':
        result = fit_tdb(x, args[0], args[1], args[2], args[3],
                         args[4], args[5], args[6])
    elif fstr == 'tdb_low':
        result = fit_tdb_low(x, args[0], args[1], args[2],
                             args[3], args[4])
    elif fstr == 'tdb_high':
        result = fit_tdb_high(x, args[0], args[1], args[2])
    elif fstr == 'rh':
        result = fit_rh(x, args[0], args[1], args[2], args[3], args[4])
    elif fstr == 'rh_low':
        result = fit_rh_low(x, args[0], args[1], args[2])
    elif fstr == 'rh_high':
        result = fit_rh_high(x, args[0], args[1], args[2])
    else:
        result = None
    logger.info("fit success")
    return result


def fit_tdb(x, a0, a1, b1, a2, b2, a3, b3, logger=None):
    result = (a0 +
              a1 * np.cos(2 * np.pi * x / 8760) +
              b1 * np.sin(2 * np.pi * x / 8760) +
              a2 * np.cos(2 * np.pi * x / 4380) +
              b2 * np.sin(2 * np.pi * x / 4380) +
              a3 * np.cos(2 * np.pi * x / 24) +
              b3 * np.sin(2 * np.pi * x / 24))
    return result


def fit_tdb_low(x, a0, a1, b1, a2, b2, logger=None):
    result = (a0 +
              a1 * np.cos(2 * np.pi * x / 8760) +
              b1 * np.sin(2 * np.pi * x / 8760) +
              a2 * np.cos(2 * np.pi * x / 4380) +
              b2 * np.sin(2 * np.pi * x / 4380))
    return result


def fit_tdb_high(x, a0, a3, b3, logger=None):
    result = (a0 +
              a3 * np.cos(2 * np.pi * x / 24) +
              b3 * np.sin(2 * np.pi * x / 24))
    return result


def fit_rh(x, a0, a1, b1, a3, b3, logger=None):
    result = (a0 +
              a1 * np.cos(2 * np.pi * x / 8760) +
              b1 * np.sin(2 * np.pi * x / 8760) +
              a3 * np.cos(2 * np.pi * x / 24) +
              b3 * np.sin(2 * np.pi * x / 24))
    return result


def fit_rh_low(x, a0, a1, b1, logger=None):
    result = (a0 +
              a1 * np.cos(2 * np.pi * x / 8760) +
              b1 * np.sin(2 * np.pi * x / 8760))
    return result


def fit_rh_high(x, a0, a3, b3, logger=None):
    result = (a0 +
              a3 * np.cos(2 * np.pi * x / 24) +
              b3 * np.sin(2 * np.pi * x / 24))
    return result
