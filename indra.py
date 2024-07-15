#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:32:27 2017

@author: Parag Rastogi

This script is called from the command line. It only parses the arguments
and invokes Indra.

Create Synthetic Weather based on some recorded data.
The algorithm works by creating synthetic time series over
short periods based on short histories. These short series
may be comined to obtain a longer series.
Script originally written by Parag Rastogi. Started: July 2017
@author = Parag Rastogi

Description of algorithm:
    1. Load data.
    2. Scale data using a standard scaler (subtract mean and divide by std).
    3. Enter model-fitting loop:
        a. Select 14 days of history (less when beginning).
        b. Use history to train model for next day.
        c. Sample from next day to obtain synthetic "predictions".
        d. Once the "predictions" are obtained for every day of the year,
           we are left with synthetic time series.
    4. Un-scale the data using the same scaler as (2) above.
    5. Clean / post-process the data if needed, e.g., oscillation of solar
       values around sunrise and sunset.
"""

# For parsing the arguments.
import argparse
import os
import glob
import pickle
import time
import pandas as pd

# These custom functions load and clean recorded data.
# For now, we are only concerned with ncdc and nsrdb.
import wfileio as wf

from petites import setseed
import resampling as resampling

# Custom functions to calculate error metrics - not currently used.
# import losses.
# from losses import rmseloss
# from losses import maeloss

WEATHER_FMTS = ["espr", "epw", "csv", "fin4"]


def indra(train=False, station_code="abc", n_samples=10,
          path_file_in="wf_in.epw", path_file_out="wf_out.epw",
          file_type="epw", store_path=".",
          climate_change=False, path_cc_file='ccfile.p',
          cc_scenario='rcp85', epoch=None,
          randseed=None, year=0, variant=0,
          arma_params=None,
          bounds=None):

    # Reassign defaults if incoming list params are None
    # (i.e., nothing passed.)
    if arma_params is None:
        arma_params = [2, 2, 1, 1, 24]

    if bounds is None:
        bounds = [0.01, 99.9]

    # ------------------
    # Some initialisation house work.

    # Convert incoming station_code to lowercase.
    station_code = station_code.lower()

    # Make a folder named using the station code in case no path to
    # folder was passed.
    if store_path == '.':
        store_path = station_code

    # Store everything in a folder named <station_code>.
    if not os.path.isdir(store_path):
        os.makedirs(store_path)

    # if isinstance(store_path, str):

    if epoch is not None:
        # These will be the files where the outputs will be stored.
        path_model_save = os.path.join(
            store_path, 'model_{:d}_{:d}.p'.format(epoch[0], epoch[1]))
        # Save output time series.
        path_syn_save = os.path.join(
            store_path, 'syn_{:d}_{:d}.p'.format(epoch[0], epoch[1]))
        path_counter_save = os.path.join(
            store_path, 'counter_{:d}_{:d}.p'.format(epoch[0], epoch[1]))

    else:
        # This is for the sampling run, where a list of dataframes has
        # been passed.
        # These will be the files where the outputs will be stored.
        path_model_save = os.path.join(
            store_path, 'model.p')
        # Save output time series.
        path_syn_save = os.path.join(
            store_path, 'syn.p')
        path_counter_save = os.path.join(
            store_path, 'counter.p')

    # ----------------

    if train:

        # The learning/sampling functions rely on random sampling. For one
        # run, the random seed is constant/immutable; changing it during a
        # run would not make sense. This makes the runs repeatable -- keep
        # track of the seed and you can reproduce exactly the same random
        # number draws as before.

        # If the user did not specify a random seed, then the generator
        # uses the current time, in seconds since some past year, which
        # differs between Unix and Windows. Anyhow, this is saved in the
        # model output in case the results need to be reproduced.
        if randseed is None:
            randseed = int(time.time())

        # Set the seed with either the input random seed or the one
        # assigned just before.
        setseed(randseed)

        # See accompanying script "wfileio".
        # try:
        if os.path.isfile(path_file_in):
            xy_train, locdata, header = wf.get_weather(station_code, path_file_in)

        elif os.path.isdir(path_file_in):

            list_wfiles = ([glob.glob(os.path.join(path_file_in, "*." + x))
                            for x in WEATHER_FMTS] +
                           [glob.glob(
                            os.path.join(path_file_in, "*." + x.upper()))
                            for x in WEATHER_FMTS])
            list_wfiles = sum(list_wfiles, [])

            xy_list = list()

            for file in list_wfiles:
                xy_temp, locdata, header = wf.get_weather(station_code, file)
                xy_list.append(xy_temp)

            xy_train = pd.concat(xy_list, sort=False)

        print("Successfully retrieved weather data.\r\n")

        # Train the models.
        print("Training the model. Go get a coffee or something...\r\n")

        if climate_change:

            cc_data = pickle.load(open(path_cc_file, 'rb'))
            cc_data = cc_data[cc_scenario]
            cc_models = set(cc_data.index.get_level_values(0))

            # Pass only the relevant epochs to resampling.
            # For some reason, some models have repetitions and NaNs.
            # This will drop models with no data.

            temp_dict = dict()
            for model in cc_models:

                temp = cc_data.loc[model]
                temp = temp.dropna(how='any')
                # Some times there are non-unique indices, as in duplicate
                # days. Get rid of them by taking the means.
                temp = temp.groupby(temp.index).mean()
                orig_index = temp.index

                if orig_index.shape[0] > 0:
                    temp_dict[model] = temp[
                        (orig_index.year <= epoch[1]) &
                        (orig_index.year >= epoch[0])]

            # import ipdb; ipdb.set_trace()

            cc_data = pd.concat(temp_dict)

        else:
            cc_data = None

        # Hard-coded the scenario as of now - should be added as a
        # parameter later.
        # cc_scenario = 'rcp85'

        # Call resampling with null selmdl and ffit, since those
        # haven"t been trained yet.
        ffit, selmdl, _ = resampling.trainer(
            xy_train, n_samples=n_samples,
            picklepath=path_syn_save,
            arma_params=arma_params,
            bounds=bounds, cc_data=cc_data)

        # The non-seasonal order of the model. This exists in both
        # ARIMA and SARIMAX models, so it has to exist in the output
        # of resampling.
        order = [(p.model.k_ar, 0, p.model.k_ma) for p in selmdl]
        # Also the endogenous variable.
        endog = [p.model.endog for p in selmdl]

        params = [p.params for p in selmdl]

        try:
            # Try to find the seasonal order. If it exists, save the
            # sarimax model. This should almost always be the case.
            seasonal_order = [
                (int(mdl.model.k_seasonal_ar / mdl.model.seasonal_periods),
                 0,
                 int(mdl.model.k_seasonal_ma / mdl.model.seasonal_periods),
                 mdl.model.seasonal_periods)
                for mdl in selmdl]

            arma_save = dict(order=order, params=params,
                             seasonal_order=seasonal_order,
                             ffit=ffit, endog=endog,
                             randseed=randseed)

        except Exception:
            # Otherwise, ask for forgiveness and save the ARIMA model.
            arma_save = dict(order=order, params=params, endog=endog,
                             ffit=ffit, randseed=randseed)

        with open(path_model_save, "wb") as open_file:
            pickle.dump(arma_save, open_file)

        # Save counter.
        csave = dict(n_samples=n_samples, randseed=randseed, counter=0)
        # with open(path_counter_save, "wb") as open_file:
        pickle.dump(csave, open(path_counter_save, "wb"))

        print(("I've saved the model for station '{0}'. "
               "You can now ask me for samples in folder '{1}'."
               "\r\n").format(station_code, store_path))

    else:

        # Call the functions in sampling mode.

        # The output, xout, is a numpy nd-array with the standard
        # columns ("month", "day", "hour", "tdb", "tdp", "rh",
        # "ghi", "dni", "dhi", "wspd", "wdr")

        # In this MC framework, the "year" of weather data is meaningless.
        # If climate change models or UHI models are added, the years will
        # mean something. For now, any number will do.

        # Load counter.
        csave = pickle.load(open(path_counter_save, 'rb'))

        if climate_change:
            sample = resampling.sampler(
                picklepath=path_syn_save, year=year, n=variant)

        else:
            # Sample number has not exceeded number of samples.
            if csave['counter'] < csave['n_samples']:
                sample = resampling.sampler(
                    picklepath=path_syn_save, counter=csave['counter'])
                csave['counter'] += 1
                pickle.dump(csave, open(path_counter_save, "wb"))
            else:
                print('You are asking me for more samples than I have.' +
                      'You generated {:d} '.format(csave['n_samples']) +
                      'samples, I have given you ' +
                      '{:d} samples.'.format(csave['counter']))
                print('Next call will restart from the first sample.')
                csave['counter'] = 0
                pickle.dump(csave, open(path_counter_save, "wb"))
                return

        if os.path.isdir(path_file_in):

            list_wfiles = [glob.glob(os.path.join(path_file_in, "*." + x))
                           for x in WEATHER_FMTS]
            list_wfiles = sum(list_wfiles, [])

        else:
            list_wfiles = [path_file_in]

        _, locdata, header = wf.get_weather(            station_code, list_wfiles[0])

        # Save / write-out synthetic time series.
        wf.give_weather(sample, locdata, station_code, header,
                        file_type=file_type,
                        path_file_out=path_file_out,
                        masterfile=list_wfiles[0])



# Define a parser.
PARSER = argparse.ArgumentParser(
    description="This is INDRA, a generator of synthetic weather " +
    "time series. This function both 'learns' the structure of data " +
    "and samples from the learnt model. Both run modes need 'seed' " +
    "data, i.e., some input weather data.\r\n", prog='INDRA',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

PARSER.add_argument("--train", type=int, choices=[0, 1], default=0,
                    help="Enter 0 for no seed data (sampling mode), " +
                    "or 1 if you are passing seed data (training or " +
                    "initalisation mode).")
PARSER.add_argument("--station_code", type=str, default="abc",
                    help="Make up a station code. " +
                    "If you are not passing seed data, and want me to " +
                    "pick up a saved model, please use the station code" +
                    " of the saved model.")
PARSER.add_argument("--n_samples", type=int, default=10,
                    help="How many samples do you want out?")
PARSER.add_argument("--path_file_in", type=str, help="Path to a weather " +
                    "file (seed file).", default="wf_in.a")
PARSER.add_argument("--path_file_out", type=str, help="Path to where the " +
                    "synthetic data will be written. If you ask for more " +
                    "than one sample, I will append an integer to the name.",
                    default="wf_out.a")
PARSER.add_argument("--file_type", type=str, default="espr",
                    help=("What kind of input weather file "
                          "are you giving me? Default is the ESP-r ascii "
                          "format [espr]. For now, I can read EPW [epw] and "
                          "ESP-r ascii files. If you pass a plain csv [csv] "
                          "or python pickle [py] file, it must contain a "
                          "table with the requisite data in the correct "
                          "order. See file data_in_spec.txt for the format."))
# Indra needs the data to be a numpy nd-array arranged exactly so:
# month, day of year, hour, tdb, tdp, rh, ghi, dni, dhi, wspd, wdr
PARSER.add_argument("--store_path", type=str, default="SyntheticWeather",
                    help="Path to the folder where all outputs will go." +
                    " Default behaviour is to create a folder in the " +
                    "present working directory called SyntheticWeather.")
PARSER.add_argument("--climate_change", type=int, choices=[0, 1], default=0,
                    help="Enter 0 to not include climate change models, or" +
                    " 1 to do so. If you want to use a CC model, you have" +
                    " to pass a path to the file containing those outputs.")
PARSER.add_argument("--epochs", type=str, default=None,
                    help='Future epochs (decades usually) if using a ' +
                    'climate model to add a signal that shifts the ' +
                    'current distribution. Enter as pairs of numbers ' +
                    'separated by commas, e.g., 2015, 2060')
PARSER.add_argument("--path_cc_file", type=str, default="ccfile.p",
                    help="Path to the file containing CC model outputs.")
# PARSER.add_argument("--station_coordinates", type=str, default="[0, 0, 0]",
#                     help="Station latitude, longitude, altitude. " +
#                     "Not currently used.")
PARSER.add_argument("--randseed", type=int, default=42,
                    help="Set the seed for this sampling " +
                    "run. If you don't know what this " +
                    "is, don't worry. The default is 42. Obviously.")
PARSER.add_argument("--arma_params", type=str, default="[2,2,1,1,24]",
                    help=("A list of UPPER LIMITS of the number of SARMA "
                          "terms [AR, MA, Seasonal AR, Seasonal MA, "
                          "Seasonality] to use in the model. Input should "
                          "look like a python list, i.e., a,b,c , WITHOUT "
                          "SPACES. If you don't know what this is, " +
                          "don't worry. The default is 2,2,1,1,24. "
                          "The default frequency of Indra is hours, so "
                          "seasonality should be declared in hours."))
PARSER.add_argument("--bounds", type=str, default="[1,99]",
                    help=("Lower and upper bound percentile values to "
                          "use for cleaning the synthetic data. Input "
                          "should look like a python list, i.e., [a,b,c], "
                          "WITHOUT SPACES. The defaults bounds are the "
                          "1 and 99 percentiles, i.e., [1, 99]."))

ARGS = PARSER.parse_args()

train = bool(ARGS.train)
station_code = ARGS.station_code.lower()
n_samples = ARGS.n_samples
path_file_in = ARGS.path_file_in
path_file_out = ARGS.path_file_out
file_type = ARGS.file_type
store_path = ARGS.store_path
# station_coordinates = [float(x.strip("[").strip("]"))
#                        for x in ARGS.station_coordinates.split(",")]
climate_change = ARGS.climate_change
epochs = ARGS.epochs
path_cc_file = ARGS.path_cc_file
randseed = ARGS.randseed
arma_params = [int(x.strip("[").strip("]"))
               for x in ARGS.arma_params.split(",")]
bounds = [float(x.strip("[").strip("]")) for x in ARGS.bounds.split(",")]

if ARGS.epochs is None and climate_change:
    epochs = [2051, 2060]
elif ARGS.epochs is not None and climate_change:
    list_years = ARGS.epochs.split(",")
    epochs = [[int(x), int(y)]
              for x, y in zip(list_years[0::2], list_years[1::2])]
else:
    epochs = None


print("\r\nInvoking indra for {0}.\r\n".format(station_code))

if store_path == "SyntheticWeather":
    store_path = store_path + '_' + station_code

# Call indra using the processed arguments.
if __name__ == "__main__":
    indra(train, station_code=station_code,
          n_samples=n_samples,
          path_file_in=path_file_in,
          path_file_out=path_file_out,
          file_type=file_type,
          store_path=store_path,
          climate_change=climate_change,
          path_cc_file=path_cc_file,
          randseed=randseed,
          arma_params=arma_params,
          bounds=bounds)
