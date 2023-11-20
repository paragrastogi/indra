# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:01:13 2018

@author: rasto
"""

import os
import zipfile
import numpy as np
import pandas as pd
from lib.fileio.wfileio import give_weather
import math
import lib.misc.petites as petite


def typical_file_selector(zip_buffer, d_xout, epochs, path_templatefile, templatefile_type, bounds = [1, 99]):

# %%

    # Produce one TMY file for each CC model.
    ccmodels = list(set([x['model'] for x in d_xout]))
    
    # Get scenario from d_xout - they are all the same scenario
    scenario = d_xout[0]['scenario']

    bin_length = 50

    # %%

    weights = {'vars':['max tdb', 'min tdb', 'mean tdb', 'max tdp', 'min tdp', 'mean tdp', 'max wspd', 'mean wspd', 'sum ghi', 'sum dni'],
               'tmy3':[x/20 for x in [1, 1, 2, 1, 1, 2, 1, 1, 5, 5]],
               'sandia':[x/24 for x in [1, 1, 2, 1, 1, 2, 2, 2, 12, 12]]}
    
    print('Starting typical file selector...')
    
    with zipfile.ZipFile(
        file = zip_buffer,                  # File object
        mode = 'a',                         # To append to file
        compression = zipfile.ZIP_DEFLATED, # Standard zip compression
        allowZip64 = True                   # Allow file to be bigger than 4GiB
        ) as zip_file:
        
        for model_idx, model in enumerate(ccmodels):
    
            xout_list = list()
            for outdict in d_xout:
                if outdict['model']==model:
                    xout_temp = outdict['xout']
                    xout_temp['sample_number'] = outdict['sample_number']
                    xout_list.append(xout_temp)
    
            xout = pd.concat(xout_list)
    
            df_grouped = xout.loc[:, ('tdb', 'tdp', 'wspd', 'ghi', 'dni', 'sample_number', 'year', 'month')].groupby([xout['sample_number'], xout.index.year, xout.index.month, xout.index.day])
    
            dailyness = pd.DataFrame(
                data=np.hstack((df_grouped.max().values[:, :3], df_grouped.min().values[:, :2], df_grouped.mean().values[:, :3], df_grouped.sum().values[:, 3:5], df_grouped.first().values[:, -3:])),
                index=xout.index[0:-1:24],
                columns=['max tdb', 'max tdp', 'max wspd', 'min tdb', 'min tdp', 'mean tdb', 'mean tdp', 'mean wspd', 'sum ghi', 'sum dni', 'sample_number', 'year', 'month'])
    
            bins = dailyness.apply(lambda x: np.linspace(np.floor(np.percentile(x, 5)), np.floor(np.percentile(x, 95)), bin_length))
            bins.drop(['sample_number', 'year', 'month'], axis=1, inplace=True)
    
            dailyness_grouped_all = dailyness.iloc[:,:-3].groupby(by=[dailyness.month])
    
            ecdfs_all = pd.DataFrame(
                data=np.array([dailyness_grouped_all[var].apply(lambda x: petite.ecdf(x, bins=bins[var])[0]).values for var in bins.columns]).T,
                index=pd.date_range(start=xout.index[0], end=xout.index[-1], freq='1M')[0:12],
                columns=bins.columns)
    
            dailyness_grouped = dailyness.iloc[:,:-3].groupby(by=[dailyness.sample_number, dailyness.year, dailyness.month])
    
            ecdfs_monthly = pd.DataFrame(
                data=np.array([dailyness_grouped[var].apply(lambda x: petite.ecdf(x, bins=bins[var])[0]).values for var in bins.columns]).T,
                index=pd.DatetimeIndex(np.tile(pd.date_range(start=xout.index[0], end=xout.index[-1], freq='1M'), len(xout['sample_number'].unique()))),
                columns=bins.columns)
    
            ecdfs_monthly['sample_number'] = dailyness_grouped.first().index.get_level_values(0)
    
            fs_monthly = pd.DataFrame(data=np.NaN * np.ones(ecdfs_monthly.shape),
                                  index=ecdfs_monthly.index, columns=ecdfs_monthly.columns)
            fs_monthly['sample_number'] = ecdfs_monthly['sample_number']
    
        # %%
    
            # Calculate the Finkelstein-Schaefer statistic for each month in each year for each variant.
    
            composite_tmy = list()
    
            min_fs_list = list()
    
            for month in range(1, 13):
    
                ecdfs_this_month = ecdfs_monthly.loc[ecdfs_monthly.index.month==month, [x for x in ecdfs_monthly.columns if x != 'sample_number']]
                ecdfs_sample_number = ecdfs_monthly.loc[ecdfs_monthly.index.month==month, 'sample_number']
    
                fs_this_month = (ecdfs_this_month - np.tile(ecdfs_all[ecdfs_all.index.month==month].values, (ecdfs_this_month.shape[0],1)))
    
                w_fs_this_month = list()
                for x in range(0, fs_this_month.shape[0]):
                    for y in range(0, fs_this_month.shape[1]):
                        fs_this_month.iloc[x, y] = np.sum(fs_this_month.iloc[x, y]) / bin_length
    
                    w_fs_this_month.append(np.sum(fs_this_month.iloc[x,:]*weights['tmy3']))
    
                w_fs_this_month = np.array(w_fs_this_month)
    
                fs_monthly.loc[fs_monthly.index.month==month, [x for x in fs_monthly.columns if x != 'sample_number']] = fs_this_month
    
                min_fs = [ecdfs_this_month.index[np.argmin(w_fs_this_month)], ecdfs_sample_number.iloc[np.argmin(w_fs_this_month)]]
                min_fs_list.append(min_fs)
    
                # Start composing the TMY month-by-month.
                composite_tmy.append(xout[np.logical_and(np.logical_and(xout.index.month == min_fs[0].month, xout.index.year == min_fs[0].year), xout['sample_number'] == min_fs[1])])
    
            # Concatenate the list of monthwise dataframes into a year.
            composite_tmy = pd.concat(composite_tmy)
            mid_decade_year = math.floor((epochs[1] - epochs[0]) / 2) + epochs[0]
    
            composite_tmy['year'] = mid_decade_year
            new_index = pd.date_range(
                start='{:04d}-01-01 00:00:00'.format(mid_decade_year),
                end='{:04d}-12-31 23:00:00'.format(mid_decade_year),
                freq='1H')
            composite_tmy.index = new_index[~((new_index.month == 2) &
                              (new_index.day == 29))]
        
            del composite_tmy['sample_number']
    
            # Smooth based on first difference - this should change the month joins.
            
            composite_tmy.loc[:, 'atmpr'] = petite.atmprcleaner(composite_tmy.loc[:, 'atmpr'], tol=0.1)
            
            composite_tmy = petite.firstdiffcleaner(
                datain = composite_tmy,
                xy_train = xout,
                bounds = bounds,
                use_cols = ['tdb', 'tdp', 'rh', 'atmpr'],
                smoothing_window_hrs = 6,
                fit_window_hrs = 24,
                poly_order = 3)
            
            file_name = os.path.join('typical/{:s}'.format(scenario), 'ccmodel_{:02d}_years_{:04d}_{:04d}.{:s}'.format(model_idx, epochs[0], epochs[1], templatefile_type))
            
            # Write out something so we know what's going on
            print('Adding file {} to archive'.format(file_name))
    
            file_bytes, file_type = give_weather(
                df = composite_tmy,
                templatefile = path_templatefile,
                write_as = templatefile_type,
                path_file_out = None)
            
            # Add bytes to file
            zip_file.writestr(zinfo_or_arcname = file_name, data = file_bytes.getvalue())
            
        
