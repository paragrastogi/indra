import os
import numpy as np
import pandas as pd
import csv
import re
from scipy import interpolate


import petites as petite

"""
This file contains functions to:
    1. load weather data from "typical" and "actual" (recorded) weather
       data files.
    2. Write out synthetic weather data to EPW or ESPr weather file formats.
    3. Associated helper functions.
"""

__author__ = "Parag Rastogi"

# ISSUES TO ADDRESS
# 1. Harmonize WMO numbers - if the incoming number is 5 digits,
# add leading zero (e.g., Geneva)
# 2. Implement something to convert GHI to DNI and DHI.
# Maybe use Erbs model like before.

# %%

# Useful strings and constants.

# List of keywords that identify TMY and AMY files.
keywords = dict(tmy=("nrel", "iwec", "ishrae", "cwec",
                     "igdg", "tmy3", "meteonorm"),
                amy=("ncdc", "nsrdb", "nrel_indiasolar",
                     "ms", "WY2", "nasa_saudi"))
wformats = ("epw", "espr", "csv", "fin4")

# List of values that could be NaNs.
nanlist = ("9900", "-9900", "9999", "99", "-99", "9999.9", "999.9", " ", "-")

# A generic ESPR header that can be used in a print or str
# command with format specifiers.
espr_generic_header = """*CLIMATE
# ascii weather file from {0},
# defined in: {1}
# col 1: Diffuse solar on the horizontal (W/m**2)
# col 2: External dry bulb temperature   (Tenths DEG.C)
# col 3: Direct normal solar intensity   (W/m**2)
# col 4: Prevailing wind speed           (Tenths m/s)
# col 5: Wind direction     (clockwise deg from north)
# col 6: Relative humidity               (Percent)
{2}               # site name
 {3},{4},{5},{6}   # year, latitude, long diff, direct normal rad flag
 {7},{8}    # period (julian days)"""

# # The standard columns used by indra.
# std_cols = ("year", "month", "day", "hour", "tdb", "tdp", "rh",
#             "ghi", "dni", "dhi", "wspd", "wdr")

def get_weather(stcode, fpath):

    # This function calls the relevant reader based on the file_type.

    # Initialise as a non-object.
    wdata = None
    locdata = None
    header = None

    file_type = os.path.splitext(fpath)[-1].replace('.', '').lower()

    if not os.path.isfile(fpath):
        print("I cannot find file {0}.".format(fpath) +
              " Returning empty dataframe.\r\n")
        return wdata, locdata, header

    # Load data for given station.

    if file_type == "pickle":
        try:
            wdata_array = pd.read_pickle(fpath)
            return wdata_array
        except Exception as err:
            print("You asked me to read a pickle but I could not. " +
                  "Trying all other formats.\r\n")
            print("Error: " + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == "epw":

        try:
            wdata, locdata, header = read_epw(fpath)
            # Remove leap day.
            wdata = petite.remove_leap_day(wdata)
        except Exception as err:
            print("Error: " + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == "espr":

        try:
            wdata, locdata, header, columns = read_espr(fpath)
            # Remove leap day.
            wdata = petite.remove_leap_day(wdata)
        except Exception as err:
            print("Error: " + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == "csv" or fpath[-4:] == ".csv":

        try:
            wdata = pd.read_csv(fpath, header=0)
            wdata.columns = ["year", "month", "day", "hour", "tdb", "tdp", "rh",
                             "ghi", "dni", "dhi", "wspd", "wdr"]
            # Location data is nonsensical, except for station code,
            # which will be reassigned later in this function.
            locdata = dict(loc=stcode, lat="00", long="00",
                           tz="00", alt="00", wmo="000000")
            header = ("# Unknown incoming file format " +
                      "(not epw or espr)\r\n" +
                      "# Dummy location data: " +
                      "loc: {0}".format(locdata["loc"]) +
                      "lat: {0}".format(locdata["lat"]) +
                      "long: {0}".format(locdata["long"]) +
                      "tz: {0}".format(locdata["tz"]) +
                      "alt: {0}".format(locdata["alt"]) +
                      "wmo: {0}".format(locdata["wmo"]) +
                      "\r\n")
            # Remove leap day.
            wdata = petite.remove_leap_day(wdata)

        except Exception as err:
            print("Error: " + str(err))
            wdata = None
            header = None
            locdata = None

    elif file_type == 'fin4' or fpath[:-4] == 'fin4':

        try:
            wdata, locdata, header = read_fin4(fpath)
        except Exception as err:
            print("Error: " + str(err))
            import ipdb; ipdb.set_trace()
            wdata = None
            header = None
            locdata = None

    # End file_type if statement.


    locdata["loc"] = stcode

    if wdata is None:
        print("I could not read the file you gave me with the format " +
              "you specified. Trying all readers.\r\n")

        return wdata, locdata, header

    else:
        # Remove leap day.
        wdata = petite.remove_leap_day(wdata)

        if len(np.unique(wdata['year'].values)) > 1:
            # Incoming file is probably a TMY or TRY file,
            # so insert a dummy year.
            wdata["year"] = 2223

        date_index = pd.date_range(
            start='{:d}-01-01 00:00:00'.format(int(wdata["year"][0])),
            end='{:d}-12-31 23:00:00'.format(int(wdata["year"][0])),
            freq='1H')
        wdata.index = date_index[
            ~((date_index.day == 29) & (date_index.month == 2))]

        return wdata, locdata, header



def read_fin4(fpath):

    header_cols = ["year", "month", "day", "hour",
                   "tdb", "tdp", "atmpr", "sky", "osky",
                   "wspd", "wdir", "ghi", "dni", "Pres",
                   "Rain", "vis", "chgt", "solarz"]

    locdata = dict()
    hlines = 3
    header = list()

    with open(fpath, 'r') as openfile:
        for ln in range(0, hlines):
            header.append(openfile.readline())

    wdata = pd.read_csv(
        fpath, sep='\s+', skiprows=[0, 1, 2],
        names=header_cols, dtype=str, index_col=False)

    temp_index = pd.date_range(
        start='{:d}-01-01 00:00:00'.format(int(wdata["year"][0])),
        end='{:d}-12-31 23:00:00'.format(int(wdata["year"][0])),
        freq='1H')

    if wdata.shape[0] == temp_index.shape[0]:
        wdata.index = temp_index
    elif wdata.shape[0] < temp_index.shape[0]:
        wdata.index = temp_index[~((temp_index.day == 29) &
                                   (temp_index.month == 2))]

    wdata = wdata.dropna(axis=1, how='all')

    for col in wdata.columns:

        wdata[col] = wdata[col].apply(
            lambda x: float(''.join(re.findall('[0-9.-]', x))))

        if col in ['year', 'month', 'day', 'hour']:
            wdata[col] = wdata[col].apply(lambda x: int(x))

    wdata = petite.remove_leap_day(wdata)

    wdata['rh'] = pd.Series(petite.calc_rh(wdata['tdb'], wdata['tdp']),
                            index=wdata.index)

    return wdata, locdata, header



# Number of days in each month.
m_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def day_of_year(month, day):

    month = month.astype(int) - 1
    doy = np.zeros_like(day, dtype=int)

    for m, mon in enumerate(month):
        doy[m] = (day[m] + np.sum(m_days[0:mon])).astype(int)

    return doy

# End function day_of_year



def day_of_month(day):

    month = np.zeros_like(day, dtype=int)
    dom = np.zeros_like(day, dtype=int)

    for d, doy in enumerate(day):

        rem = doy
        prev_ndays = 0

        for m, ndays in enumerate(m_days):
            # The iterator "m" starts at zero.

            if rem <= 0:
                # Iterator has now reached the incomplete month.

                # The previous month is the correct month.
                # Not subtracting 1 because the iterator starts at 0.
                month[d] = m

                # Add the negative remainder to the previous month"s days.
                dom[d] = rem + prev_ndays
                break

            # Subtract number of days in this month from the day.
            rem -= ndays
            # Store number of days from previous month.
            prev_ndays = ndays

    return month, dom

# End function day_of_month


# %%
epw_colnames = ["Year", "Month", "Day", "Hour", "Minute", "QualFlags",
                "TDB", "TDP", "RH", "ATMPR", "ETRH", "ETRN", "HIR",
                "GHI", "DNI", "DHI", "GHE", "DNE", "DHE", "ZL",
                "WDR", "WSPD", "TSKY", "OSKY", "VIS", "CHGT",
                "PWO", "PWC", "PWT", "AOPT", "SDPT",
                "SLAST", "UnknownVar1", "UnknownVar2", "UnknownVar3"]


def read_epw(fpath, epw_colnames=epw_colnames):

    # Names of the columns in EPW files. Usually ignore the last
    # three columns.

    # Number of header lines expected.
    hlines = 8

    # Convert the names to lowercase.
    epw_colnames = [x.lower() for x in epw_colnames]

    # Read table, ignoring header lines.
    wdata = pd.read_csv(fpath, delimiter=",", skiprows=hlines,
                        header=None, names=epw_colnames,
                        index_col=False)

    if len(wdata['year'].unique()) > 1:
        wdata['year'] = 2223

    dates= pd.date_range(start='{}-01-01 00:00:00'.format(wdata['year'].unique()[0]), end='{}-12-31 23:00:00'.format(wdata['year'].unique()[0]), freq='1H')

    if len(dates) > wdata.shape[0]:
        dates = dates[~((dates.month == 2) & (dates.day == 29))]

    wdata.index = dates
    wdata = petite.remove_leap_day(wdata)

    if len(wdata.columns) == 35:
        # Some files have three extra columns
        # (usually the TMY files from USDOE).
        # Delete those columns if found.
        wdata = wdata.drop(["unknownvar1", "unknownvar2",
                            "unknownvar3"], axis=1)

    # Read header and assign all metadata.
    header = list()
    hf = open(fpath, "r")
    for ln in range(0, hlines):
        header.append(hf.readline())

    infoline = (header[0].strip()).split(",")

    locdata = dict(loc=infoline[1], lat=infoline[6], long=infoline[7],
                   tz=infoline[8], alt=infoline[9], wmo=infoline[5])

    if wdata.empty:

        print("Could not locate a file with given station name." +
              " Returning empty table.\r\n")

    return wdata, locdata, header

# ----------- END read_epw function -----------



def read_espr(fpath):

    # Missing functionality - reject call if path points to binary file.

    # Uniform date index for all tmy weather data tables.
    dates = pd.date_range("1/1/2223", periods=8760, freq="H")

    fpath_fldr, fpath_name = os.path.split(fpath)
    sitename = fpath_name.split(sep=".")
    sitename = sitename[0]

    with open(fpath, "r") as f:
        content = f.readlines()

    # Read first line to get format.
    if content[0].strip() == '*WEATHER 2':
        hlines = 14
        iver = 2
    elif content[0].strip() == '*CLIMATE 2':
        hlines = 13
        iver = 1
    elif content[0].strip() == '*CLIMATE':
        hlines = 12
        iver = 0
    else:
        print('Error: Format of ESP-r weather file not recognised\r\n')
        clmdata = None
        locdata = None
        header = None
        return clmdata, locdata, header

    # Split the contents into a header and body.
    header = content[0:hlines]

    # Find the year of the current file.
    yline = [line for line in header if "year" in line]

    yline = yline[0].split("#")[0]

    if "," in yline:
        yline_split = yline.split(",")
    else:
        yline_split = yline.split()
    year = yline_split[0].strip()

    locline = [line for line in header if ("latitude" in line)][0]

    if iver == 0 or iver == 1:
        siteline = [line for line in header if ("site name" in line)][0]
    elif iver == 2:
        siteline = header[1]

    if "," in locline:
        locline = locline.split(",")
    else:
        locline = locline.split()

    if "," in siteline:
        siteline = siteline.split(",")
    else:
        siteline = siteline.split()

    locdata = dict(loc=siteline[0], lat=locline[1], long=locline[2],
                   tz="00", alt="0000", wmo="000000")
    # ESP-r files do not contain timezone, altitude, or WMO number.

    # Decide what parameters are in which columns depending on file
    # version.
    if iver == 0:
        dhicol = 0
        tdbcol = 1
        dnicol = 2
        wspdcol = 3
        wdrcol = 4
        rhcol = 5
        esp_columns = ["dhi", "tdb", "dni", "wspd", "wdr", "rh"]
    elif iver == 1:
        colslist = header[12].strip().split(',')
        esp_columns = [None]*6
        tdbcol = int(colslist[0])-1
        esp_columns[tdbcol] = 'tdb'
        dhicol = int(colslist[1])-1
        esp_columns[dhicol] = 'dhi'
        dnicol = int(colslist[2])-1
        esp_columns[dnicol] = 'dni'
        wspdcol = int(colslist[4])-1
        esp_columns[wspdcol] = 'wspd'
        wdrcol = int(colslist[5])-1
        esp_columns[wdrcol] = 'wdr'
        rhcol = int(colslist[6])-1
        esp_columns[rhcol] = 'rh'
    elif iver == 2:
        esp_columns = [None]*6
        tdbcol = int(header[4].strip().split('|')[1])-1
        esp_columns[tdbcol] = 'tdb'
        dhicol = int(header[5].strip().split('|')[1])-1
        esp_columns[dhicol] = 'dhi'
        dnicol = int(header[6].strip().split('|')[1])-1
        esp_columns[dnicol] = 'dni'
        wspdcol = int(header[8].strip().split('|')[1])-1
        esp_columns[wspdcol] = 'wspd'
        wdrcol = int(header[9].strip().split('|')[1])-1
        esp_columns[wdrcol] = 'wdr'
        rhcol = int(header[10].strip().split('|')[1])-1
        esp_columns[rhcol] = 'rh'


    body = content[hlines:]

    del content

    # Find the lines with day tags.
    daylines = [[idx, line] for [idx, line] in enumerate(body)
                if "day" in line]

    dataout = np.zeros([8760, 11])

    dcount = 0

    for idx, day in daylines:

        # Get the next 24 lines.
        daylist = np.asarray(body[idx+1:idx+25])

        # Split each line of the current daylist into separate strings.
        if "," in daylist[0]:
            splitlist = [element.split(",") for element in daylist]
        else:
            splitlist = [element.split() for element in daylist]

        # Convert each element to a integer, then convert the resulting
        # list to a numpy array.
        daydata = np.asarray([list(map(int, x)) for x in splitlist])

        # Today's time slice.
        dayslice = range(dcount, dcount+24, 1)

        # This will split the day-month header line on the gaps.
        if "," in day:
            splitday = day.split(",")
        else:
            splitday = day.split(" ")

        # Remove blanks.
        splitday = [x for x in splitday if x != ""]
        splitday = [x for x in splitday if x != " "]

        # Month.
        dataout[dayslice, 0] = np.repeat(int(splitday[-1]), len(dayslice))

        # Day of month.
        dataout[dayslice, 1] = np.repeat(int(splitday[2]), len(dayslice))

        # Hour (of day).
        dataout[dayslice, 2] = np.arange(0, 24, 1)

        # tdb, input is in deci-degrees, convert to degrees.
        dataout[dayslice, 3] = daydata[:, tdbcol]/10

        # tdp is calculated after this loop.

        # rh, in percent.
        dataout[dayslice, 5] = daydata[:, rhcol]

        # ghi is calculated after this loop.

        # dni, in W/m2.
        dataout[dayslice, 7] = daydata[:, dnicol]

        # dhi, in W/m2.
        dataout[dayslice, 8] = daydata[:, dhicol]

        # wspd, input is in deci-m/s.
        dataout[dayslice, 9] = daydata[:, wspdcol]/10

        # wdr, clockwise deg from north.
        dataout[dayslice, 10] = daydata[:, wdrcol]

        dcount += 24

    # tdp, calculated from tdb and rh.
    dataout[:, 4] = petite.calc_tdp(dataout[:, 3], dataout[:, 5])

    # ghi, in W/m2.
    dataout[:, 6] = dataout[:, 7] + dataout[:, 8]

    # wspd can have bogus values (999)
    dataout[dataout[:, 10] >= 999., 10] = np.nan
    idx = np.arange(0, dataout.shape[0])
    duds = np.logical_or(np.isinf(dataout[:, 10]), np.isnan(dataout[:, 10]))
    int_func = interpolate.interp1d(
        idx[np.logical_not(duds)], dataout[np.logical_not(duds), 10],
        kind="nearest", fill_value="extrapolate")
    dataout[duds, 10] = int_func(idx[duds])

    dataout = np.concatenate((np.reshape(np.repeat(int(year), 8760),
                                         [-1, 1]), dataout), axis=1)

    clmdata = pd.DataFrame(data=dataout, index=dates,
                           columns=["year", "month", "day", "hour",
                                    "tdb", "tdp", "rh",
                                    "ghi", "dni", "dhi", "wspd", "wdr"])

    return clmdata, locdata, header, esp_columns

# ----------- END read_espr function -----------


def give_weather(df, locdata, stcode, header,
                 masterfile="GEN_IWEC.epw", file_type="epw",
                 path_file_out=".", std_cols=None):

    file_type = file_type.lower()

    if file_type == 'csv' and isinstance(df, pd.DataFrame):
        std_cols = df.columns

    # If no columns were passed, infer them from the columns of the dataframe.
    if std_cols is None:
        std_cols = df.columns

    # Check if incoming temperature values are in Kelvin.
    for col in ['tdb', 'tdp']:
        if any(df.loc[:, col] > 200):
            df.loc[:, col] = df.loc[:, col] - 273.15

    success = False

    year = np.unique(df.index.year)[0]

    # Convert date columns to integers.
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], downcast='unsigned')
    if 'day' in df.columns:
        df['day'] = pd.to_numeric(df['day'], downcast='unsigned')
    if 'hour' in df.columns:
        df['hour'] = pd.to_numeric(df['hour'], downcast='unsigned')

    # If last hour was interpreted as first hour of next year, you might
    # have two years.
    # This happens if the incoming file has hours from 1 to 24.
    if isinstance(year, list):
        counts = np.bincount(year)
        year = np.argmax(counts)

    if path_file_out == ".":
        # Make a standardised name for output file.
        filepath = os.path.join(
            path_file_out, "wf_out_{0}_{1}".format(
                np.random.randint(0, 99, 1), year))
    else:
        # Files need to be renamed so strip out the extension.
        filepath = path_file_out.replace(
            ".a", "").replace(".epw", "").replace(".csv", "")

    # if str(year) not in filepath:
    #     filepath = filepath + "_{:04d}".format(year)

    # # If no variant or counter number is found, add it.
    # if re.findall('\d', filepath):
    #     filepath = filepath + "_{:04d}".format(year)

    if file_type == "espr":

        esp_master, locdata, header, esp_columns = read_espr(masterfile)

        # Replace the year in the header.
        yline = [line for line in header if "year" in line]
        yval = yline[0].split(",")
        yline[0] = yline[0].replace(yval[0], str(year))
        header = [yline[0] if "year" in line else line
                  for line in header]
        # Cut out the last new-line character since numpy savetxt
        # puts in a newline character after the header anyway.
        header[-1] = header[-1][:-1]

        for col in esp_columns:
            esp_master.loc[:, col] = df[col].values
            if col in ["tdb", "wspd"]:
                # Deci-degrees and deci-m/s respectively.
                esp_master.loc[:, col] *= 10
        # Create a datetime index for this year.
        esp_master.index = pd.date_range(
            start='{:04d}-01-01 00:00:00'.format(year),
            end='{:04d}-12-31 23:00:00'.format(year),
            freq='1H')

        # Save month and day to write out to file as separate rows.
        monthday = (esp_master.loc[:, ["day", "month"]]).astype(int)

        # Drop those columns that will not be written out.
        esp_master = esp_master.drop(
            ["year", "month", "day", "hour", "ghi", "tdp"],
            axis=1)
        # Re-arrange the columns into the espr clm file order.
        esp_master = esp_master[esp_columns]
        # Convert all data to int.
        esp_master = esp_master.astype(int)

        master_aslist = esp_master.values.tolist()

        md_master = 0
        for md in range(0, monthday.shape[0], 24):
            md_list = [str("* day {0} month {1}".format(
                monthday["day"][md], monthday["month"][md]))]
            master_aslist.insert(md_master, md_list)
            md_master += 25

        # Write the header to file - though the delimiter is
        # mostly meaningless in this case.
        with open(filepath, "w") as f:
            f.write(''.join(header)+'\n')

            spamwriter = csv.writer(f, delimiter=",", quotechar="",
                                    quoting=csv.QUOTE_NONE,
                                    escapechar=" ",
                                    lineterminator="\n ")
            for line in master_aslist[:-1]:
                spamwriter.writerow(line)

            spamwriter = csv.writer(f, delimiter=",", quotechar="",
                                    quoting=csv.QUOTE_NONE,
                                    lineterminator="\n\n")
            spamwriter.writerow(master_aslist[-1])

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

        # End espr writer.

    elif file_type == "epw":

        if filepath.split(".")[-1] != "epw":
            filepath = filepath + ".epw"

        epw_fmt = (["%4u", "%2u", "%2u", "%2u", "%2u", "%44s"] +
                   ((np.repeat("%5.2f", len(epw_colnames) - (6 + 3))
                     ).tolist()))

        epw_master, locdata, header = read_epw(masterfile)
        # Cut out the last new-line character since numpy savetxt
        # puts in a newline character after the header anyway.
        header[-1] = header[-1][:-1]


        # These columns will be replaced.
        epw_columns = ["tdb", "tdp", "rh", "ghi", "dni", "dhi", "wspd", "wdr"]


        for col in epw_columns:
            epw_master.loc[:, col] = df[col].values

        # Replace the year of the master file.
        epw_master["year"] = year



        np.savetxt(filepath, df.values, fmt=epw_fmt,
                   delimiter=",", header="".join(header),
                   comments="")

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

        # End EPW writer.

    elif file_type == "fin4":
        if filepath.split(".")[-1] != "fin4":
            filepath = filepath + "fin4"

        _, _, header = read_fin4(masterfile)

        # Strip the last end-of-line character.
        header[-1] = header[-1].strip('\r').strip('\n')

        # Convert pressure to millibars.
        df['atmpr'] = df['atmpr'] / 100

        df = df.drop(labels='rh', axis=1)
        df['tdp'] = petite.tdpcleaner(df['tdp'], df['tdb'])

        # df.to_csv(filepath, sep=" ", header=" ".join(header), index=False)
        fin_fmt = (["%4d", "%2d", "%2d", "%2d"] +
                   ((np.repeat("%6.1f", len(df.columns) - (4))
                     ).tolist()))

        with open(filepath, 'wb') as openfile:
            np.savetxt(openfile, df.values, fmt=fin_fmt,
                       delimiter=" ", header="".join(header),
                       comments="")














        # import ipdb; ipdb.set_trace()
















        if os.path.isfile(filepath):
            success = True
        else:
            success = False

    else:

        if filepath.split(".")[-1] != "csv":
            filepath = filepath + "csv"

        df.to_csv(filepath, sep=",", header=True, index=False)

        if os.path.isfile(filepath):
            success = True
        else:
            success = False

    if success:
        print("Write success.")
    else:
        print("Some error prevented file from being written.")

# ----------- End give_weather function. -----------









































































































