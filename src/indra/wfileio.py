import os
import re
import csv
import numpy as np
import pandas as pd

from . import petites as petite
from .logging_utils import get_logger


EPW_COLNAMES = [
    "year", "month", "day", "hour", "minute", "qualflags",
    "tdb", "tdp", "rh", "atmpr", "etrh", "etrn", "hir",
    "ghi", "dni", "dhi", "ghe", "dne", "dhe", "zl",
    "wdr", "wspd", "tsky", "osky", "vis", "chgt",
    "pwo", "pwc", "pwt", "aopt", "sdpt",
    "slast", "unknownvar1", "unknownvar2", "unknownvar3",
]

ALLOWED_INPUTS = {"epw", "csv", "tsv", "parquet"}

CANONICAL_MAP = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "dry_bulb_temperature": "tdb",
    "drybulb": "tdb",
    "dry_bulb": "tdb",
    "tdb": "tdb",
    "dew_point_temperature": "tdp",
    "dewpoint": "tdp",
    "dew_point": "tdp",
    "tdp": "tdp",
    "relative_humidity": "rh",
    "rh": "rh",
    "humidity_ratio": "w",
    "humratio": "w",
    "w": "w",
    "absolute_humidity": "abs_hum",
    "abs_humidity": "abs_hum",
    "wet_bulb_temperature": "twb",
    "wetbulb": "twb",
    "twb": "twb",
    "ghi": "ghi",
    "dni": "dni",
    "dhi": "dhi",
    "wspd": "wspd",
    "wdr": "wdr",
    "atmpr": "atmpr",
    "pressure": "atmpr",
}


def _normalize_columns(columns, logger=None):
    logger = get_logger(logger)
    normalized = {}
    for col in columns:
        key = re.sub(r"[^a-z0-9_]", "", col.strip().lower().replace(" ", "_"))
        normalized[col] = CANONICAL_MAP.get(key, key)
    logger.info("_normalize_columns success")
    return normalized


def _ensure_required_columns(df, logger=None):
    logger = get_logger(logger)
    if "tdb" not in df.columns:
        raise ValueError("Missing required dry bulb temperature column (tdb).")
    has_humidity = any(col in df.columns for col in ("tdp", "rh", "twb", "abs_hum", "w"))
    if not has_humidity:
        raise ValueError(
            "Must include at least one of: wet bulb, dew point, relative humidity, absolute humidity, humidity ratio."
        )
    logger.info("_ensure_required_columns success")


def _derive_missing_humidity(df, logger=None):
    logger = get_logger(logger)
    # Prefer tdp/rh; derive missing if possible.
    if "tdp" not in df.columns and "rh" in df.columns:
        df["tdp"] = petite.calc_tdp(df["tdb"].values, df["rh"].values)
    if "rh" not in df.columns and "tdp" in df.columns:
        df["rh"] = petite.calc_rh(df["tdb"].values, df["tdp"].values)
    if "rh" not in df.columns and "tdp" not in df.columns and "w" in df.columns:
        ps = df["atmpr"].values if "atmpr" in df.columns else 101325
        df["rh"] = petite.w2rh(df["w"].values, df["tdb"].values, ps=ps)
        df["tdp"] = petite.calc_tdp(df["tdb"].values, df["rh"].values)
    if "rh" not in df.columns and "tdp" not in df.columns and "twb" in df.columns:
        ps = df["atmpr"].values if "atmpr" in df.columns else 101325
        df["rh"] = petite.twb2rh(df["tdb"].values, df["twb"].values, ps=ps)
        df["tdp"] = petite.calc_tdp(df["tdb"].values, df["rh"].values)
    if "rh" not in df.columns and "tdp" not in df.columns and "abs_hum" in df.columns:
        df["rh"] = petite.abs_hum2rh(df["tdb"].values, df["abs_hum"].values)
        df["tdp"] = petite.calc_tdp(df["tdb"].values, df["rh"].values)
    if "rh" not in df.columns or "tdp" not in df.columns:
        raise ValueError(
            "Unable to derive required humidity fields (tdp and rh) from provided inputs."
        )
    logger.info("_derive_missing_humidity success")


def read_epw(fpath, logger=None):
    logger = get_logger(logger)
    hlines = 8
    wdata = pd.read_csv(
        fpath,
        delimiter=",",
        skiprows=hlines,
        header=None,
        names=EPW_COLNAMES,
        index_col=False,
    )
    header = []
    with open(fpath, "r") as hf:
        for _ in range(hlines):
            header.append(hf.readline())
    locdata = {}
    if header:
        infoline = (header[0].strip()).split(",")
        if len(infoline) > 9:
            locdata = dict(
                loc=infoline[1],
                lat=infoline[6],
                long=infoline[7],
                tz=infoline[8],
                alt=infoline[9],
                wmo=infoline[5],
            )
    if len(wdata["year"].unique()) > 1:
        wdata["year"] = 2223
    dates = pd.date_range(
        start=f"{wdata['year'].unique()[0]}-01-01 00:00:00",
        end=f"{wdata['year'].unique()[0]}-12-31 23:00:00",
        freq="1h",
    )
    if len(dates) > wdata.shape[0]:
        dates = dates[~((dates.month == 2) & (dates.day == 29))]
    wdata.index = dates
    wdata = petite.remove_leap_day(wdata)
    logger.info("read_epw success")
    return wdata, locdata, header


def _read_generic_table(fpath, file_type, logger=None):
    logger = get_logger(logger)
    if file_type == "csv":
        df = pd.read_csv(fpath)
    elif file_type == "tsv":
        df = pd.read_csv(fpath, sep="\t")
    elif file_type == "parquet":
        df = pd.read_parquet(fpath)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    df = df.rename(columns=_normalize_columns(df.columns, logger=logger))
    _ensure_required_columns(df, logger=logger)
    _derive_missing_humidity(df, logger=logger)
    logger.info("_read_generic_table success")
    return df


def _ensure_datetime_index(df, logger=None):
    logger = get_logger(logger)
    if isinstance(df.index, pd.DatetimeIndex):
        logger.info("_ensure_datetime_index success")
        return df
    if all(col in df.columns for col in ("year", "month", "day", "hour")):
        df["minute"] = df["minute"] if "minute" in df.columns else 0
        df.index = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute"]]
        )
    else:
        raise ValueError("Data must include year, month, day, hour (and optional minute).")
    df = petite.remove_leap_day(df)
    logger.info("_ensure_datetime_index success")
    return df


def get_weather(stcode, fpath, logger=None):
    logger = get_logger(logger)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Input file not found: {fpath}")

    file_type = os.path.splitext(fpath)[-1].replace(".", "").lower()
    if file_type not in ALLOWED_INPUTS:
        raise ValueError(f"Unsupported input file type: {file_type}")

    if file_type == "epw":
        wdata, locdata, header = read_epw(fpath, logger=logger)
    else:
        wdata = _read_generic_table(fpath, file_type, logger=logger)
        wdata = _ensure_datetime_index(wdata, logger=logger)
        locdata = dict(loc=stcode, lat="00", long="00", tz="00", alt="00", wmo="000000")
        header = [f"# Generic {file_type} input for {stcode}\n"]

    if len(np.unique(wdata["year"].values)) > 1:
        wdata["year"] = 2223
    wdata["loc"] = stcode
    logger.info("get_weather success")
    return wdata, locdata, header


def give_weather(df, locdata, stcode, header,
                 file_type="epw", path_file_out=".",
                 masterfile="", logger=None):
    logger = get_logger(logger)
    file_type = file_type.lower()
    if file_type not in {"epw", "csv", "tsv", "parquet"}:
        raise ValueError(f"Unsupported output file type: {file_type}")

    if path_file_out == ".":
        filepath = f"{path_file_out}/wf_out_{np.random.randint(0, 99, 1)[0]}"
    else:
        filepath = path_file_out.replace(".csv", "").replace(".tsv", "").replace(".parquet", "").replace(".epw", "")

    if file_type == "epw":
        if not masterfile:
            raise ValueError("masterfile is required for epw output.")
        epw_master, _, header = read_epw(masterfile, logger=logger)
        header[-1] = header[-1][:-1]
        epw_columns = ["tdb", "tdp", "rh", "ghi", "dni", "dhi", "wspd", "wdr"]
        for col in epw_columns:
            epw_master.loc[:, col] = df[col].astype(float).values
        epw_master["year"] = np.unique(df.index.year)[0]
        epw_fmt = (["%4u", "%2u", "%2u", "%2u", "%2u", "%44s"] +
                   (np.repeat("%5.2f", len(EPW_COLNAMES) - 6).tolist()))
        outfile = f"{filepath}.epw"
        np.savetxt(outfile, epw_master.values, fmt=epw_fmt,
                   delimiter=",", header="".join(header), comments="")
    elif file_type == "csv":
        outfile = f"{filepath}.csv"
        df.to_csv(outfile, index=True)
    elif file_type == "tsv":
        outfile = f"{filepath}.tsv"
        df.to_csv(outfile, sep="\t", index=True)
    else:
        outfile = f"{filepath}.parquet"
        df.to_parquet(outfile, index=True)

    success = os.path.isfile(outfile)
    logger.info("give_weather success")
    return success
