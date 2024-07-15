#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:34 2017

@author: parag rastogi
"""

import random

import numpy as np
# from scipy import interpolate
import pandas as pd

# Constants for Eq. 5, Temperature -200°C to 0°C.
FROZEN_CONST = [-5.6745359 * 10**3, 6.3925247, -9.6778430 * 10**-3,
                6.2215701 * 10**-7, 2.0747825 * 10**-9,
                -9.4840240 * 10**-13, 4.1635019]

# Constants for Eq. 6, Temperature 0°C to 200°C.
LIQUID_CONST = [-5.8002206 * 10**3, 1.3914993, -4.8640239 * 10**-2,
                4.1764768 * 10**-5, -1.4452093 * 10**-8, 6.5459673]


def setseed(randseed):
    '''Seed random number generators. Called as a function in main indra
    script once and only once.'''

    np.random.seed(randseed)
    random.seed = randseed

# ----------- END setseed function. -----------


def quantilecleaner(datain, xy_train, var, bounds=None):
    '''Generic cleaner based on quantiles. Needs a time series / dataset
       and cut-off quantiles. Also needs the name of the variable (var) in
       the incoming dataframe. This function will censor the data outside
       those quantiles and interpolate the missing values using linear
       interpolation.'''

    if bounds is None:
        bounds = [0.01, 99.9]

    dataout = pd.DataFrame(datain)

    for this_month in range(1, 13):

        idx_this_month_rec = xy_train.index.month == this_month
        idx_this_month_syn = datain.index.month == this_month

        rec_quantiles = np.percentile(
            xy_train[var].iloc[idx_this_month_rec], bounds)

        # import ipdb; ipdb.set_trace()

        dataout = dataout.mask(
            np.logical_and(
                idx_this_month_syn,
                np.squeeze(np.logical_or(dataout < rec_quantiles[0],
                                         dataout > rec_quantiles[1]))),
            other=np.NaN)

        dataout = dataout.interpolate(
            method='linear').fillna(method='bfill').fillna(method='ffill')

    # Pass back values with only one dimension.
    return np.squeeze(dataout.values)

# ----------- END quantilecleaner function. -----------


def solarcleaner(datain, master):

    '''Clean solar values by setting zeros at corresponding times in master
       to zero in the synthetic data. This is a proxy for sunrise, sunset,
       and twilight.'''

    # Using the source data - check to see if there
    # should be sunlight at a given hour. If not,
    # then set corresponding synthetic value to zero.
    # If there is a negative value (usually at sunrise
    # or sunset), set it to zero as well.

    datain = datain.mask(datain <= 0, other=0)

    return datain

    # A potential improvement would be to calculate sunrise and sunset
    # independently since that is an almost deterministic calculation.

# ----------- END solarcleaner function. -----------


def rhcleaner(rh):

    '''RH values cannot be more than 100 or less than 0.'''

    rhout = pd.DataFrame(rh)

    rhout = rhout.mask(rhout >= 99, other=np.NaN).mask(
        rhout <= 10, other=np.NaN).mask(
        np.isnan(rhout), other=np.NaN)

    rhout = rhout.interpolate(method='linear')
    rhout = rhout.fillna(method='bfill')

    return np.squeeze(rhout.values)

# ----------- END rhcleaner function. -----------


def tdpcleaner(tdp, tdb):

    if not isinstance(tdp, pd.DataFrame):
        tdpout = pd.DataFrame(tdp)

    else:
        tdpout = tdp

    tdpout = tdpout.mask(np.squeeze(tdp) >= np.squeeze(tdb),
                         other=np.NaN)
    tdpout = tdpout.mask(
        np.logical_or(np.squeeze(tdp) >= 50, np.squeeze(tdp) <= -50),
        other=np.NaN)

    if ((np.isnan(tdpout.values)).any()):
        tdpout = tdpout.interpolate(method='linear')
        tdpout = tdpout.fillna(method='bfill').fillna(method='ffill')

    return np.squeeze(tdpout.values)

# ----------- END rhcleaner function. -----------


def wstats(datain, key, stat):

    grouped_data = datain.groupby(key)

    if stat is 'mean':
        dataout = grouped_data.mean()
    elif stat is 'sum':
        dataout = grouped_data.sum()
    elif stat is 'max':
        dataout = grouped_data.max()
    elif stat is 'min':
        dataout = grouped_data.min()
    elif stat is 'std':
        dataout = grouped_data.std()
    elif stat is 'q1':
        dataout = grouped_data.quantile(0.25)
    elif stat is 'q3':
        dataout = grouped_data.quantile(0.75)
    elif stat is 'med':
        dataout = grouped_data.median()

    return dataout

# ----------- END wstats function. -----------


def calc_rh(tdb, tdp):

    rhout = 100 * (((112 - (0.1 * tdb) + tdp) / (112 + (0.9 * tdb))) ** 8)
    return rhcleaner(rhout)


def calc_tdp(tdb, rh):

    '''Calculate dew point temperature using dry bulb temperature
       and relative humidity.'''

    # Change relative humidity to fraction.
    phi = rh/100

    # Remove weird values.
    phi[phi > 1] = 1
    phi[phi < 0] = 0

    # Convert tdb to Kelvin.
    if any(tdb < 200):
        tdb_k = tdb + 273.15
    else:
        tdb_k = tdb

    # Equations for calculating the saturation pressure
    # of water vapour, taken from ASHRAE Fundamentals 2009.
    # (Eq. 5 and 6, Psychrometrics)

    # This is to distinguish between the two versions of equation 5.
    ice = tdb_k <= 273.15
    not_ice = np.logical_not(ice)

    lnp_ws = np.zeros(tdb_k.shape)

    # Eq. 5, pg 1.2
    lnp_ws[ice] = (
        FROZEN_CONST[0]/tdb_k[ice] + FROZEN_CONST[1] +
        FROZEN_CONST[2]*tdb_k[ice] + FROZEN_CONST[3]*tdb_k[ice]**2 +
        FROZEN_CONST[4]*tdb_k[ice]**3 + FROZEN_CONST[5]*tdb_k[ice]**4 +
        FROZEN_CONST[6]*np.log(tdb_k[ice]))

    # Eq. 6, pg 1.2
    lnp_ws[np.logical_not(ice)] = (
        LIQUID_CONST[0]/tdb_k[not_ice] + LIQUID_CONST[1] +
        LIQUID_CONST[2]*tdb_k[not_ice] +
        LIQUID_CONST[3]*tdb_k[not_ice]**2 +
        LIQUID_CONST[4]*tdb_k[not_ice]**3 +
        LIQUID_CONST[5]*np.log(tdb_k[not_ice]))

    # Temperature in the above formulae must be absolute,
    # i.e. in Kelvin

    # Continuing from eqs. 5 and 6
    p_ws = np.e**(lnp_ws)  # [Pa]

    # Eq. 24, pg 1.8
    p_w = (phi * p_ws) / 1000  # [kPa]

    # Constants for Eq. 39
    EQ39_CONST = [6.54, 14.526, 0.7389, 0.09486, 0.4569]

    p_w[p_w <= 0] = 1e-6
    alpha = pd.DataFrame(np.log(p_w))
    alpha = alpha.replace(
        [np.inf, -np.inf], np.NaN).interpolate(method='linear')

    # Eq. 39
    tdp = alpha.apply(
        lambda x: EQ39_CONST[0] + EQ39_CONST[1]*x + EQ39_CONST[2]*(x**2) +
        EQ39_CONST[3]*(x**3) + EQ39_CONST[4]*(p_w**0.1984))

    # Eq. 40, TDP less than 0°C and greater than -93°C
    tdp_ice = tdp < 0
    tdp[tdp_ice] = 6.09 + 12.608*alpha[tdp_ice] + 0.4959*(alpha[tdp_ice]**2)

    tdp = tdp.replace(
        [np.inf, -np.inf], np.NaN).interpolate(method='linear')

    tdp = tdp.fillna(method='bfill').fillna(method='ffill')

    # tdp = (tdp).rename('tdp')

    tdp = tdpcleaner(tdp, tdb)

    return tdp

# ----------- END tdb2tdp function. -----------


def w2rh(w, tdb, ps=101325):

    if any(tdb < 200):
        tdb_k = tdb + 273.15
    else:
        tdb_k = tdb

    # Humidity ratio W, [unitless fraction]
    # Equation (22), pg 1.8
    p_w = ((w / 0.621945) * ps) / (1 + (w / 0.621945))

    # This is to distinguish between the two versions of equation 5.
    ice = tdb_k <= 273.15
    not_ice = np.logical_not(ice)

    lnp_ws = np.zeros(tdb_k.shape)

    # Eq. 5, pg 1.2
    lnp_ws[ice] = (
        FROZEN_CONST[0] / tdb_k[ice] + FROZEN_CONST[1] +
        FROZEN_CONST[2] * tdb_k[ice] + FROZEN_CONST[3] * tdb_k[ice]**2 +
        FROZEN_CONST[4] * tdb_k[ice]**3 + FROZEN_CONST[5] * tdb_k[ice]**4 +
        FROZEN_CONST[6] * np.log(tdb_k[ice]))

    # Eq. 6, pg 1.2
    lnp_ws[np.logical_not(ice)] = (
        LIQUID_CONST[0] / tdb_k[not_ice] + LIQUID_CONST[1] +
        LIQUID_CONST[2] * tdb_k[not_ice] +
        LIQUID_CONST[3] * tdb_k[not_ice]**2 +
        LIQUID_CONST[4] * tdb_k[not_ice]**3 +
        LIQUID_CONST[5] * np.log(tdb_k[not_ice]))

    # Temperature in the above formulae must be absolute,
    # i.e. in Kelvin

    # Continuing from eqs. 5 and 6
    p_ws = np.e**(lnp_ws)  # [Pa]

    phi = p_w / p_ws  # [Pa] Formula(24), pg 1.8

    rh = phi * 100

    # Relative Humidity from fraction to percentage.
    return rhcleaner(rh)

# ----------- END w2rh function. -----------


def remove_leap_day(df):
    '''Removes leap day using time index.'''

    return df[~((df.index.month == 2) & (df.index.day == 29))]

# ----------- END remove_leap_day function. -----------


def euclidean(x, y):

    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# ----------- END euclidean function. -----------
