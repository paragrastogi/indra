# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 20:29:16 2017

@author: rasto
"""
from sys import stdout
from itertools import product
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from tqdm import tqdm


def select_models(arma_params, ts_in):

    '''Select the most parsimonious SARMA model.'''

    # Set ranges for various model parameters.
    # Each range is one more than what we are
    # interested in because range cuts off at end-1.
    # arma_params = [arp_ub, maq_ub, sarp_ub, smaq_ub, seasonality]

    aic_curr = 0
    selaic = np.infty
    mod_fit_curr = None

    counter = 0
    # Total iterations expected.
    # total_iters = np.prod([x+1 for x in arma_params[:-1]])
    # print(total_iters)

    # Loop through all possible combinations of ar, ma, sar, and sma lags.

    print("Iteration number: ")

    for p, q, pp, qq in product(
            range(0, arma_params[0]+1), range(0, arma_params[1]+1),
            range(0, arma_params[2]+1), range(0, arma_params[3]+1)):

        if p == 0 and q == 0:
            continue

        model = SARIMAX(
            ts_in, order=(p, 0, q),
            seasonal_order=(pp, 0, qq, arma_params[4]),
            trend=None)

        try:
            mod_fit_curr = model.fit(
                disp=0, cov_type="robust",
                full_output=True)
            aic_curr = mod_fit_curr.aic

            if np.isnan(aic_curr):
                counter += 1
                continue

            # if counter > 0:
            if aic_curr < selaic:
                [selaic, selmdl] = [aic_curr, mod_fit_curr]

            # elif counter == 0:
            #     [selaic, selmdl] = [aic_curr, mod_fit_curr]

            counter += 1

        except Exception as err:
            # print('fit threw an error')
            continue

        # Print out a heartbeat.
        print("{0} ...".format(counter))

    # End p, q, pp, qq nested loops.

    resid = selmdl.resid

    return selmdl, resid
