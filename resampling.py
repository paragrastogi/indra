#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:47:58 2017

@author: rasto

Translating, as faithfully as possible, the resampling method first proposed
in (Rastogi, 2016, EPFL).
"""

import pickle
import copy

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

import fourier
from ts_models import select_models
# Useful small functions like solarcleaner.
import petites as petite

# Number of variables resampled - TDB and RH.
NUM_VARS = 2

# "Standard" length of output year.
STD_LEN_OUT = 8760

# Number of nearest neighbours (of each synthetic day in array of recorded
# days using daily mean temperature) from which to choose solar radiation.
# NUM_NBOURS = 10

# This is the master tuple of column names, which should not
# be modified.
# COLUMNS = ('year', 'month', 'day', 'hour', 'tdb', 'tdp', 'rh',
#            'ghi', 'dni', 'dhi', 'wspd', 'wdr')

# Key for netcdf data.
# 'tas' = 'TDBdmean',
# 'tasmin' = 'TDBdmin',
# 'tasmax' = 'TDBdmax',
# 'ps' = 'ATMPRdmean',
# 'sfcWind' = 'Wspd_dmean'
# 'rsds' = 'GHIdmean',
# 'huss' = 'Wdmean'

cc_cols = [["tdb", "tas"], ["rh", None], ["atmpr", "ps"],
           ["wspd", "sfcWind"], ["ghi", "rsds"]]


def trainer(xy_train, n_samples, picklepath, arma_params, bounds, cc_data):
    """Train the model with this function."""

    # Save a copy of all data to calculate quantiles later.
    xy_train_all = xy_train

    # ARIMA models cannot be fit to overly long time series.
    # Select a random year from all the data available for ARIMA.

    # Get the unique years. The typical years are read as 2223.
    all_years = np.unique(xy_train_all.index.year)

    xy_train = pd.DataFrame()

    while xy_train.shape[0] < STD_LEN_OUT:

        select_year = all_years[np.random.randint(0, len(all_years), 1)[0]]

        # Keep only that one year of data.
        xy_train = xy_train_all[str(select_year) + '-01-01':
                                str(select_year) + '-12-31']

        xy_train = petite.remove_leap_day(xy_train)

        if xy_train.shape[0] > STD_LEN_OUT:
            xy_train = xy_train.iloc[0:STD_LEN_OUT, :]

    x_calc_params = np.arange(0, xy_train_all.shape[0])
    x_fit_models = np.arange(0, STD_LEN_OUT)

    # Fit fourier functions to the tdb and rh series.

    # The curve_fit function outputs two things: parameters of the fit and
    # the estimated covariance. We only use the first.
    # Inputs are the function to fit (fourier in this case),
    # xdata, and ydata.
    # Use all the data available to calculate these parameters.
    params = [curve_fit(fourier.fit_tdb, x_calc_params, xy_train_all['tdb']),
              curve_fit(fourier.fit_rh, x_calc_params, xy_train_all['rh'])]

    # Call the fourier fit function with the calculated
    # parameters to get the values of the fourier fit at each time step
    ffit = [fourier.fit('tdb', x_fit_models, *params[0][0]),
            fourier.fit('rh', x_fit_models, *params[1][0])]

    if cc_data is not None:

        params_cc = [
            curve_fit(fourier.fit_tdb_low, x_calc_params,
                      xy_train_all['tdb']),
            curve_fit(fourier.fit_tdb_high, x_calc_params,
                      xy_train_all['tdb']),
            curve_fit(fourier.fit_rh_low, x_calc_params,
                      xy_train_all['rh']),
            curve_fit(fourier.fit_rh_high, x_calc_params,
                      xy_train_all['rh'])
            ]

        ffit_cc = [fourier.fit('tdb_low', x_fit_models, *params_cc[0][0]),
                   fourier.fit('tdb_high', x_fit_models, *params_cc[1][0]),
                   fourier.fit('rh_low', x_fit_models, *params_cc[2][0]),
                   fourier.fit('rh_high', x_fit_models, *params_cc[3][0])]

    # Now subtract the low- and high-frequency fourier fits
    # (whichever is applicable) from the raw values to get the
    # 'de-meaned' values (values from which the mean has
    # been removed).

    sans_means = pd.concat([x - y for x, y in
                            zip([xy_train["tdb"], xy_train["rh"]], ffit)],
                           axis=1)
    sans_means.index = xy_train.index

    # Fit ARIMA models.

    selmdl = list()
    resid = np.zeros([sans_means["tdb"].shape[0], NUM_VARS])

    for idx, ser in enumerate(sans_means):
        mdl_temp, resid[:, idx] = select_models(
            arma_params, sans_means[ser])
        selmdl.append(mdl_temp)

    print(("Done with fitting models to TDB and RH.\r\n"
           "Simulating the learnt model to get synthetic noise series. "
           "This might take some time.\r\n"))

    resampled = np.zeros([STD_LEN_OUT, NUM_VARS, n_samples])

    for midx, mdl in enumerate(selmdl):
        for sample_num in range(0, n_samples):
            resampled_temp = mdl.simulate(nsimulations=STD_LEN_OUT)
            resampled[:, midx, sample_num] = ((resampled_temp-np.mean(resampled_temp))/np.std(resampled_temp))*np.std(resid) + np.mean(resid)
        # End n for loop.
    # End mdl for loop.

    # Add the resampled time series back to the fourier series.

    if cc_data is None:

        xout = create_future_no_cc(
            xy_train, sans_means, ffit, resampled, n_samples, bounds)

    else:

        cc_models = set(cc_data.index.get_level_values(0))
        xout = list()  # ([xy_train] * n_samples)

        for model in tqdm(cc_models):

            this_cc_out = cc_data.loc[model]
            gcm_years = np.unique(this_cc_out.index.year)

            for yidx, future_year in enumerate(gcm_years):

                # Select only this year of cc model outputs.
                cctable = this_cc_out[str(future_year) + '-01-01':
                                      str(future_year) + '-12-31']
                # leap_idx = [idx for idx, x in enumerate(cctable.index)
                # if x == pd.to_datetime(str(future_year) + '-02-29 12:00:00')]
                # cctable = cctable.drop(cctable.index[leap_idx])
                cctable = petite.remove_leap_day(cctable)

                if cctable.shape[0] < 365:
                    continue

                for nidx in range(0, n_samples):

                    xout_temp = copy.deepcopy(xy_train)

                    future_index = pd.date_range(
                        start=str(future_year) + "-01-01 00:00:00",
                        end=str(future_year) + "-12-31 23:00:00",
                        freq='1H')
                    # Remove leap days.
                    future_index = future_index[
                        ~((future_index.month == 2) &
                          (future_index.day == 29))]
                    xout_temp.index = future_index
                    xout_temp['year'] = future_index.year

                    for idx, var in enumerate(cc_cols):

                        if var[0] == "rh":
                            huss = cctable["huss"].values
                            # Convert specific humifity to humidity ratio.
                            w = -huss / (huss - 1)

                            # Convert humidity ratio (w) to
                            # Relative Humidity (RH).
                            rh = petite.w2rh(
                                w, cctable["tas"].values,
                                cctable["ps"].values)

                            # Is there some way to replace the fourier fit at
                            # a finer grain instead of repeating the daily
                            # mean value 24 times?
                            ccvar = np.repeat(rh, [24], axis=0)

                        elif var[0] == "tdb":
                            ccvar = np.repeat(
                                cctable[var[1]].values - 273.15, [24],
                                axis=0)

                        else:
                            ccvar = np.repeat(
                                cctable[var[1]].values, [24], axis=0)

                        # Add the resampled time series to the high-frequency
                        # fourier fit and the cc model output.

                        if var[0] == 'tdb':
                            xout_temp[var[0]] = (
                                resampled[:, idx, nidx] + ffit_cc[1] -
                                ffit_cc[0] + ccvar)

                        elif var[0] == 'rh':
                            xout_temp[var[0]] = (
                                resampled[:, idx, nidx] + ffit_cc[3] -
                                ffit_cc[2] + ccvar)
                        else:
                            xout_temp[var[0]] = ccvar

                        xout_temp[var[0]] = petite.quantilecleaner(
                            xout_temp[var[0]], xy_train, var[0])

                    xout_temp['tdp'] = petite.calc_tdp(
                        xout_temp["tdb"], xout_temp["rh"])
                    xout_temp['tdp'] = petite.quantilecleaner(
                        xout_temp['tdp'], xy_train, 'tdp')

                    xout.append(xout_temp)

    # End for loop.

    # End loop over samples.

    # Calculate TDP.

    xout = nearest_neighbour(xout, xy_train_all, 'tdb', 'ghi')
    # xout = nearest_neighbour(xout, xy_train_all, 'tdb', 'wspd')

    # tdp = (np.asarray([x.loc[:, 'tdp'] for x in xout])).T
    # tdb = (np.asarray([x.loc[:, 'tdb'] for x in xout])).T

    # for idx, df in enumerate(xout):
    #     if np.any(df["tdp"] > df["tdb"]):
    #         print(np.where(df["tdp"] > df["tdb"]))
    #         df["tdp"] = petite.tdpcleaner(df['tdp'], df['tdb'])
    #         xout[idx] = df

    # Save the outputs as a pickle.
    pickle.dump(xout, open(picklepath, 'wb'))

    # End nidx loop.

    return ffit, selmdl, xout


def sampler(picklepath, year=0, n=0, counter=0):
    """Only opens the pickle of saved samples and returns ONE sample."""

    try:

        if isinstance(picklepath, list):
            xout = picklepath
        else:
            xout = pickle.load(open(picklepath, 'rb'))

        if np.logical_not(year == 0 and n == 0):
            yidx = [idx for idx, x in enumerate(xout)
                    if np.unique(x.index.year) == year]

            sample = xout[yidx[n]]

        else:
            sample = xout[counter]

    except AttributeError:

        print("I could not open the pickle file with samples. " +
              "Please check it exists at {0}.".format(picklepath))
        sample = None

    return sample


def create_future_no_cc(rec, sans_means, ffit, resampled, n_samples, bounds):
    # First make the xout array using all variables. Variables other
    # than RH and TDB are just repeated from the incoming files.
    xout = list()

    all_years = np.unique(rec.index.year)

    if len(all_years) == 1:
        select_year = all_years[0]
    else:
        all_years = all_years[:-1]
        select_year = all_years[np.random.randint(0, len(all_years), 1)[0]]

    # Keep only that one year of data.
    rec_year = rec[str(select_year) + '-01-01':
                   str(select_year) + '-12-31']

    rec_year = petite.remove_leap_day(rec_year)

    if rec_year.shape[0] > STD_LEN_OUT:
        rec_year = rec_year.iloc[0:STD_LEN_OUT, :]

    # Clean the generated temperature values using extreme percentiles
    # as proxies for 'outliers'.

    # Clean the RH values using phyiscal limits (0-100).

    # Add the fourier fits from the training data to the
    # resampled/resimulated ARMA model outputs.

    for nidx in range(0, n_samples):

        # Copy the master datatable of all values.
        xout_temp = copy.deepcopy(rec_year)

        for idx, var in enumerate(sans_means[["tdb", "rh"]]):

            syn = pd.Series(data=resampled[:, idx, nidx] + ffit[idx],
                            index=pd.date_range(
                                start="2223-01-01 00:00:00",
                                end="2223-12-31 23:00:00",
                                freq='1H'))

            # Replace only var (tdb or rh).
            # Also send it to the quantile cleaner.
            xout_temp[var] = petite.quantilecleaner(
                syn, rec, var, bounds=bounds)

        xout.append(xout_temp)

    return xout


def nearest_neighbour(syn, rec, basevar, othervar):

    # Calculate daily means of temperature.
    mean_list = {basevar: list(), othervar: list()}

    for var in [basevar, othervar]:
        for df in syn:

            df_dm = df[var].resample('1D').mean()

            if len(df_dm) > 365:
                df_dm = petite.remove_leap_day(df_dm)

            mean_list[var].append(df_dm)

    if othervar == 'ghi':
        othervar_idx = [x for x, y in enumerate(rec)
                        if y in ['ghi', 'dhi', 'dni']]
    elif othervar == 'wspd':
        othervar_idx = [x for x, y in enumerate(rec)
                        if y in ['wspd', 'wdir']]
    else:
        othervar_idx = [x for x, y in enumerate(rec)
                        if y in [othervar]]

    # Number of nearest neighbours to keep when varying solar quantities.
    nn_top = 10
    # days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for this_month in range(1, 13):

        print('Month ' + str(this_month))

        # This month's indices.
        idx_this_month_rec = rec.index.month == this_month

        rec_this_month = rec.iloc[idx_this_month_rec, :]

        rec_means_this_month = (np.asarray(
            [rec_this_month[basevar].resample(
                '1D').mean().dropna(),
             rec_this_month[othervar].resample(
                 '1D').mean().dropna()])).T

        # Find the solar data for this month.
        othervar_this_month = np.asarray(
            [rec_this_month.iloc[:, x]
             for x in othervar_idx]).T
        # Reshape into day-sized blocks.
        othervar_this_month = np.reshape(
            othervar_this_month, [-1, 24, len(othervar_idx)])

        # Scale values for calculating the nearest neighbour.
        scaler_rec = StandardScaler()
        scaler_rec.fit(rec_means_this_month)
        rec_means_scaled = scaler_rec.transform(rec_means_this_month)

        # Cycle through each array of daily means.
        for sample_idx, (syn_sample_tdb, syn_sample_ghi) in enumerate(zip(
                mean_list[basevar], mean_list[othervar])):

            idx_this_month_syn = syn_sample_tdb.index.month == this_month

            syn_sample = np.asarray(
                [syn_sample_tdb[idx_this_month_syn].values,
                 syn_sample_ghi[idx_this_month_syn].values]).T

            scaler_syn = StandardScaler()
            scaler_syn.fit(syn_sample)
            syn_sample_scaled = scaler_syn.transform(syn_sample)

            nearest_nbours = list()

            for day_sample in syn_sample_scaled:

                # Sort samples by Euclidean distance.
                # argsort gives the arguments (indices) from sorting.
                nbours = np.argsort(
                    np.asarray([petite.euclidean(day_sample, x)
                                for x in rec_means_scaled]))

                # Keep only the first nn_top samples.
                nbours = nbours[:nn_top]
                # Select only one of those.
                nbours = nbours[np.random.randint(0, len(nbours), size=1)]
                # Save it as an integer.
                nearest_nbours.append(int(nbours))

            # Array to store the hourly samples.
            othervar_samples = np.zeros([len(nearest_nbours),
                                         24, len(othervar_idx)])

            # Cycle through the nearest neighbours.
            for ng_idx, nbour_idx in enumerate(nearest_nbours):
                othervar_samples[ng_idx, :, :] = othervar_this_month[
                    nbour_idx, :, :]

            # Reshape solar samples to be continuous.
            othervar_samples = np.reshape(
                othervar_samples, [-1, len(othervar_idx)])

            # Put the solar samples back in to syn.
            for sidx, othervar_col in enumerate(othervar_idx):
                cleaned_solar = petite.solarcleaner(
                    pd.Series(othervar_samples[:, sidx]),
                    rec.iloc[:, othervar_col])

                this_month_idx = [idx for idx, x in
                                  enumerate(syn[sample_idx].index.month)
                                  if x == this_month]

                syn[sample_idx].iloc[this_month_idx,
                                     othervar_col] = cleaned_solar.values

        # End syn_sample loop

    # End month loop.

    return syn
