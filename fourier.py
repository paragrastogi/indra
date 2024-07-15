import numpy as np


def fit(fstr, x,  * args):
    if fstr == 'tdb':
        return fit_tdb(x, args[0], args[1], args[2], args[3],
                       args[4], args[5], args[6])
    elif fstr == 'tdb_low':
        return fit_tdb_low(x, args[0], args[1], args[2],
                           args[3], args[4])
    elif fstr == 'tdb_high':
        return fit_tdb_high(x, args[0], args[1], args[2])
    elif fstr == 'rh':
        return fit_rh(x, args[0], args[1], args[2], args[3], args[4])
    elif fstr == 'rh_low':
        return fit_rh_low(x, args[0], args[1], args[2])
    elif fstr == 'rh_high':
        return fit_rh_high(x, args[0], args[1], args[2])


def fit_tdb(x, a0, a1, b1, a2, b2, a3, b3):
    return (a0 +
            a1 * np.cos(2 * np.pi * x / 8760) +
            b1 * np.sin(2 * np.pi * x / 8760) +
            a2 * np.cos(2 * np.pi * x / 4380) +
            b2 * np.sin(2 * np.pi * x / 4380) +
            a3 * np.cos(2 * np.pi * x / 24) +
            b3 * np.sin(2 * np.pi * x / 24))


def fit_tdb_low(x, a0, a1, b1, a2, b2):
    return (a0 +
            a1 * np.cos(2 * np.pi * x / 8760) +
            b1 * np.sin(2 * np.pi * x / 8760) +
            a2 * np.cos(2 * np.pi * x / 4380) +
            b2 * np.sin(2 * np.pi * x / 4380))


def fit_tdb_high(x, a0, a3, b3):
    return (a0 +
            a3 * np.cos(2 * np.pi * x / 24) +
            b3 * np.sin(2 * np.pi * x / 24))


def fit_rh(x, a0, a1, b1, a3, b3):
    return (a0 +
            a1 * np.cos(2 * np.pi * x / 8760) +
            b1 * np.sin(2 * np.pi * x / 8760) +
            a3 * np.cos(2 * np.pi * x / 24) +
            b3 * np.sin(2 * np.pi * x / 24))


def fit_rh_low(x, a0, a1, b1):
    return (a0 +
            a1 * np.cos(2 * np.pi * x / 8760) +
            b1 * np.sin(2 * np.pi * x / 8760))


def fit_rh_high(x, a0, a3, b3):
    return (a0 +
            a3 * np.cos(2 * np.pi * x / 24) +
            b3 * np.sin(2 * np.pi * x / 24))
