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
from . import wfileio as wf

from .petites import setseed
from . import resampling as resampling
from .logging_utils import get_logger, setup_logger
from . import config as config_module

# Custom functions to calculate error metrics - not currently used.
# import losses.
# from losses import rmseloss
# from losses import maeloss

WEATHER_FMTS = ["epw", "csv", "tsv", "parquet"]


def indra(
    train=False,
    station_code="abc",
    n_samples=10,
    path_file_in="wf_in.epw",
    path_file_out="wf_out.epw",
    file_type="epw",
    store_path=".",
    climate_change=False,
    path_cc_file="ccfile.p",
    cc_scenario="ssp585",
    epoch=None,
    randseed=None,
    year=0,
    variant=0,
    arma_params=None,
    bounds=None,
    arma_caps=None,
    cache_models=False,
    model_cache_path=None,
    n_jobs=1,
    logger=None,
):
    logger = get_logger(logger)

    # Reassign defaults if incoming list params are None
    # (i.e., nothing passed.)
    if arma_params is None:
        arma_params = [2, 2, 1, 1, 24]

    if bounds is None:
        bounds = [0.01, 99.9]

    # Apply caps to arma_params if passed.
    if arma_caps is not None:
        for idx in range(min(4, len(arma_params))):
            arma_params[idx] = min(arma_params[idx], arma_caps[idx])

    # ------------------
    # Some initialisation house work.

    # Convert incoming station_code to lowercase.
    station_code = station_code.lower()

    # Make a folder named using the station code in case no path to
    # folder was passed.
    if store_path == ".":
        store_path = station_code

    # Store everything in a folder named <station_code>.
    if not os.path.isdir(store_path):
        os.makedirs(store_path)

    # if isinstance(store_path, str):

    if epoch is not None:
        # These will be the files where the outputs will be stored.
        path_model_save = os.path.join(
            store_path, "model_{:d}_{:d}.p".format(epoch[0], epoch[1])
        )
        # Save output time series.
        path_syn_save = os.path.join(
            store_path, "syn_{:d}_{:d}.p".format(epoch[0], epoch[1])
        )
        path_counter_save = os.path.join(
            store_path, "counter_{:d}_{:d}.p".format(epoch[0], epoch[1])
        )

    else:
        # This is for the sampling run, where a list of dataframes has
        # been passed.
        # These will be the files where the outputs will be stored.
        path_model_save = os.path.join(store_path, "model.p")
        # Save output time series.
        path_syn_save = os.path.join(store_path, "syn.p")
        path_counter_save = os.path.join(store_path, "counter.p")

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
        logger.info("Using random seed: %d", randseed)

        # Set the seed with either the input random seed or the one
        # assigned just before.
        setseed(randseed, logger=logger)

        # See accompanying script "wfileio".
        # try:
        if os.path.isfile(path_file_in):
            xy_train, locdata, header = wf.get_weather(
                station_code, path_file_in, logger=logger
            )

        elif os.path.isdir(path_file_in):

            list_wfiles = [
                glob.glob(os.path.join(path_file_in, "*." + x)) for x in WEATHER_FMTS
            ] + [
                glob.glob(os.path.join(path_file_in, "*." + x.upper()))
                for x in WEATHER_FMTS
            ]
            list_wfiles = sum(list_wfiles, [])

            xy_list = list()

            for file in list_wfiles:
                xy_temp, locdata, header = wf.get_weather(
                    station_code, file, logger=logger
                )
                xy_list.append(xy_temp)

            xy_train = pd.concat(xy_list, sort=False)
        else:
            logger.error(
                "The path_file_in '%s' is neither a file nor a folder. Exiting.",
                path_file_in,
            )
            return

        logger.info("Successfully retrieved weather data.")

        # Train the models.
        logger.info("Training the model. Go get a coffee or something...")

        if climate_change:

            cc_data = pickle.load(open(path_cc_file, "rb"))
            cc_data = cc_data[cc_scenario]
            cc_models = set(cc_data.index.get_level_values(0))

            # Pass only the relevant epochs to resampling.
            # For some reason, some models have repetitions and NaNs.
            # This will drop models with no data.

            temp_dict = dict()
            for model in cc_models:

                temp = cc_data.loc[model]
                temp = temp.dropna(how="any")
                # Some times there are non-unique indices, as in duplicate
                # days. Get rid of them by taking the means.
                temp = temp.groupby(temp.index).mean()
                orig_index = temp.index

                if orig_index.shape[0] > 0:
                    temp_dict[model] = temp[
                        (orig_index.year <= epoch[1]) & (orig_index.year >= epoch[0])
                    ]

            # import ipdb; ipdb.set_trace()

            cc_data = pd.concat(temp_dict)

        else:
            cc_data = None

        # Hard-coded the scenario as of now - should be added as a
        # parameter later.
        # cc_scenario = 'rcp85'

        # Call resampling with null selmdl and ffit, since those
        # haven"t been trained yet.
        cachepath = None
        if model_cache_path:
            cachepath = model_cache_path
        elif cache_models:
            cachepath = os.path.join(store_path, "model_cache.p")

        ffit, selmdl, _ = resampling.trainer(
            xy_train,
            n_samples=n_samples,
            picklepath=path_syn_save,
            arma_params=arma_params,
            bounds=bounds,
            cc_data=cc_data,
            cachepath=cachepath,
            use_cache=cache_models,
            n_jobs=n_jobs,
            logger=logger,
        )

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
                (
                    int(mdl.model.k_seasonal_ar / mdl.model.seasonal_periods),
                    0,
                    int(mdl.model.k_seasonal_ma / mdl.model.seasonal_periods),
                    mdl.model.seasonal_periods,
                )
                for mdl in selmdl
            ]

            arma_save = dict(
                order=order,
                params=params,
                seasonal_order=seasonal_order,
                ffit=ffit,
                endog=endog,
                randseed=randseed,
            )

        except Exception:
            # Otherwise, ask for forgiveness and save the ARIMA model.
            arma_save = dict(
                order=order, params=params, endog=endog, ffit=ffit, randseed=randseed
            )

        with open(path_model_save, "wb") as open_file:
            pickle.dump(arma_save, open_file)

        # Save counter.
        csave = dict(n_samples=n_samples, randseed=randseed, counter=0)
        # with open(path_counter_save, "wb") as open_file:
        pickle.dump(csave, open(path_counter_save, "wb"))

        logger.info(
            "I've saved the model for station '%s'. You can now ask me for samples in folder '%s'.",
            station_code,
            store_path,
        )

    else:

        # Call the functions in sampling mode.

        # The output, xout, is a numpy nd-array with the standard
        # columns ("month", "day", "hour", "tdb", "tdp", "rh",
        # "ghi", "dni", "dhi", "wspd", "wdr")

        # In this MC framework, the "year" of weather data is meaningless.
        # If climate change models or UHI models are added, the years will
        # mean something. For now, any number will do.

        # Auto-train if sampling artifacts are missing.
        if not os.path.isfile(path_syn_save):
            logger.info(
                "Synthetic samples missing; running training to generate %s.",
                path_syn_save,
            )
            indra(
                train=True,
                station_code=station_code,
                n_samples=n_samples,
                path_file_in=path_file_in,
                path_file_out=path_file_out,
                file_type=file_type,
                store_path=store_path,
                climate_change=climate_change,
                path_cc_file=path_cc_file,
                cc_scenario=cc_scenario,
                epoch=epoch,
                randseed=randseed,
                year=year,
                variant=variant,
                arma_params=arma_params,
                bounds=bounds,
                arma_caps=arma_caps,
                cache_models=cache_models,
                model_cache_path=model_cache_path,
                n_jobs=n_jobs,
                logger=logger,
            )

        # Load counter (initialize if missing).
        if os.path.isfile(path_counter_save):
            csave = pickle.load(open(path_counter_save, "rb"))
        else:
            csave = dict(n_samples=n_samples, randseed=randseed, counter=0)
            pickle.dump(csave, open(path_counter_save, "wb"))
            logger.info(
                "Counter file missing; initialized new counter at %s.",
                path_counter_save,
            )

        if climate_change:
            sample = resampling.sampler(
                picklepath=path_syn_save, year=year, n=variant, logger=logger
            )

        else:
            # Sample number has not exceeded number of samples.
            counter = int(csave["counter"])  # type:ignore
            n_samples_saved = int(csave["n_samples"])  # type:ignore
            if counter < n_samples_saved:
                sample = resampling.sampler(
                    picklepath=path_syn_save, counter=counter, logger=logger
                )
                csave["counter"] = counter + 1
                pickle.dump(csave, open(path_counter_save, "wb"))
            else:
                logger.info(
                    "You are asking me for more samples than I have. "
                    "You generated %d samples, I have given you %d samples.",
                    n_samples_saved,
                    counter,
                )
                logger.info("Next call will restart from the first sample.")
                csave["counter"] = 0
                pickle.dump(csave, open(path_counter_save, "wb"))
                return

        if os.path.isdir(path_file_in):

            list_wfiles = [
                glob.glob(os.path.join(path_file_in, "*." + x)) for x in WEATHER_FMTS
            ]
            list_wfiles = sum(list_wfiles, [])

        else:
            list_wfiles = [path_file_in]

        _, locdata, header = wf.get_weather(station_code, list_wfiles[0], logger=logger)

        # Save / write-out synthetic time series.
        wf.give_weather(
            sample,
            locdata,
            station_code,
            header,
            file_type=file_type,
            path_file_out=path_file_out,
            masterfile=list_wfiles[0],
            logger=logger,
        )
    logger.info("indra success")


def build_parser(logger=None) -> argparse.ArgumentParser:
    return config_module.build_parser(logger=logger)


def main(argv: list[str] | None = None, logger=None) -> int:
    logger = logger or setup_logger()
    parser = build_parser(logger=logger)
    args = parser.parse_args(argv)

    if args.config:
        from . import config as config_main

        config_path = os.path.abspath(args.config)
        config = config_main.load_config(config_path, logger=logger)
        config_main.run_from_config(config, os.path.dirname(config_path), logger=logger)
        logger.info("main success")
        return 0
    else:
        default_config = "config.toml"
        if os.path.exists(default_config):
            from . import config as config_main

            config_path = os.path.abspath(default_config)
            config = config_main.load_config(config_path, logger=logger)
            config_main.run_from_config(
                config, os.path.dirname(config_path), logger=logger
            )
            logger.info("main success")
            return 0

    train = bool(args.train)
    station_code = args.station_code.lower()
    n_samples = args.n_samples
    path_file_in = args.path_file_in
    path_file_out = args.path_file_out
    file_type = args.file_type
    store_path = args.store_path
    climate_change = args.climate_change
    epochs = args.epochs
    path_cc_file = args.path_cc_file
    randseed = args.randseed
    arma_params = [int(x.strip("[").strip("]")) for x in args.arma_params.split(",")]
    bounds = [float(x.strip("[").strip("]")) for x in args.bounds.split(",")]
    arma_caps = None
    if args.arma_caps:
        arma_caps = [int(x.strip("[").strip("]")) for x in args.arma_caps.split(",")]

    if args.epochs is None and climate_change:
        epochs = [2051, 2060]
    elif args.epochs is not None and climate_change:
        list_years = args.epochs.split(",")
        epochs = [[int(x), int(y)] for x, y in zip(list_years[0::2], list_years[1::2])]
    else:
        epochs = None

    logger.info("Invoking indra for %s.", station_code)

    if store_path == "SyntheticWeather":
        store_path = store_path + "_" + station_code

    indra(
        train,
        station_code=station_code,
        n_samples=n_samples,
        path_file_in=path_file_in,
        path_file_out=path_file_out,
        file_type=file_type,
        store_path=store_path,
        climate_change=climate_change,
        path_cc_file=path_cc_file,
        randseed=randseed,
        arma_params=arma_params,
        bounds=bounds,
        arma_caps=arma_caps,
        cache_models=bool(args.cache_models),
        model_cache_path=args.model_cache_path,
        n_jobs=args.n_jobs,
        logger=logger,
    )
    logger.info("main success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
