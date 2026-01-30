import argparse
import os
import pickle
import sys

import pandas as pd

from .indra import indra as run_indra
from .logging_utils import get_logger, setup_logger

import tomllib



def _read_cmip6_variable(path: str, logger=None) -> pd.DataFrame:
    logger = get_logger(logger)
    df = pd.read_csv(path)
    if "Year" not in df.columns or "Day" not in df.columns:
        raise ValueError(f"CMIP6 file missing Year/Day columns: {path}")
    date = pd.to_datetime(df["Year"].astype(str), format="%Y") + pd.to_timedelta(
        df["Day"].astype(int) - 1, unit="D"
    )
    df = df.drop(columns=["Year", "Day"])
    df.index = date
    logger.info("read_cmip6_variable success")
    return df


def _join(*parts: str, logger=None) -> str:
    base = parts[0].rstrip("/")
    rest = [p.strip("/") for p in parts[1:]]
    return "/".join([base, *rest])


def build_cc_data(cmip6_dir: str, scenario: str, logger=None) -> pd.DataFrame:
    logger = get_logger(logger)
    dirname = cmip6_dir.rstrip("/").split("/")[-1].lower()
    tasmax_path = _join(cmip6_dir, f"{dirname}_tasmax_{scenario}.csv", logger=logger)
    tasmin_path = _join(cmip6_dir, f"{dirname}_tasmin_{scenario}.csv", logger=logger)
    hurs_path = _join(cmip6_dir, f"{dirname}_hurs_{scenario}.csv", logger=logger)

    if not os.path.exists(tasmax_path):
        raise FileNotFoundError(f"Missing tasmax file: {tasmax_path}")
    if not os.path.exists(tasmin_path):
        raise FileNotFoundError(f"Missing tasmin file: {tasmin_path}")
    if not os.path.exists(hurs_path):
        raise FileNotFoundError(f"Missing hurs file: {hurs_path}")

    tasmax = _read_cmip6_variable(tasmax_path, logger=logger)
    tasmin = _read_cmip6_variable(tasmin_path, logger=logger)
    hurs = _read_cmip6_variable(hurs_path, logger=logger)

    models = sorted(set(tasmax.columns) & set(tasmin.columns) & set(hurs.columns))
    if not models:
        raise ValueError(f"No overlapping models in CMIP6 data for {cmip6_dir}")

    frames = []
    for model in models:
        df = pd.DataFrame(
            {
                "tasmax": tasmax[model],
                "tasmin": tasmin[model],
                "tas": (tasmax[model] + tasmin[model]) / 2.0,
                "hurs": hurs[model],
            }
        )
        frames.append(df)

    cc_data = pd.concat(frames, keys=models, names=["model", "time"])
    logger.info("build_cc_data success")
    return cc_data


def load_config(path: str, logger=None) -> dict:
    logger = get_logger(logger)
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    logger.info("load_config success")
    return config


def _normalize_runs(config: dict, logger=None) -> list[dict]:
    logger = get_logger(logger)
    run_defaults = config.get("run", {})
    paths_defaults = config.get("paths", {})
    stations = config.get("stations")
    if not stations:
        logger.info("normalize_runs success")
        return [{"run": run_defaults, "paths": paths_defaults}]
    normalized = []
    for station in stations:
        run_cfg = {**run_defaults, **station.get("run", {})}
        paths_cfg = {**paths_defaults, **station.get("paths", {})}
        normalized.append({"run": run_cfg, "paths": paths_cfg})
    logger.info("normalize_runs success")
    return normalized


def _resolve_path(raw: str, config_dir: str, logger=None) -> str:
    logger = get_logger(logger)
    if os.path.isabs(raw):
        resolved = os.path.abspath(raw)
    else:
        resolved = os.path.abspath(_join(config_dir, raw, logger=logger))
    logger.info("resolve_path success")
    return resolved


def run_from_config(config: dict, config_dir: str, logger=None) -> list[str]:
    logger = get_logger(logger)
    outputs: list[str] = []

    for entry in _normalize_runs(config, logger=logger):
        run_cfg = entry["run"]
        paths_cfg = entry["paths"]

        station_code = run_cfg.get("station_code", "abc")
        n_samples = int(run_cfg.get("n_samples", 1))
        file_type = run_cfg.get("file_type", "epw")
        store_path = run_cfg.get("store_path", f"outputs/{station_code}")
        output_template = run_cfg.get(
            "output_template", f"{station_code}_syn_{{sample:02d}}.{file_type}"
        )
        climate_change = bool(run_cfg.get("climate_change", False))
        cc_scenario = run_cfg.get("cc_scenario", "ssp585")
        epoch = run_cfg.get("epoch")
        randseed = run_cfg.get("randseed")
        arma_params = run_cfg.get("arma_params")
        arma_caps = run_cfg.get("arma_caps")
        bounds = run_cfg.get("bounds")
        cache_models = bool(run_cfg.get("cache_models", False))
        model_cache_path = run_cfg.get("model_cache_path")
        n_jobs = run_cfg.get("n_jobs", 1)

        input_path = _resolve_path(paths_cfg.get("input_path", ""), config_dir, logger=logger)
        if model_cache_path:
            model_cache_path = _resolve_path(str(model_cache_path), config_dir, logger=logger)

        if not os.path.isabs(store_path):
            store_path = _join(config_dir, store_path, logger=logger)
        os.makedirs(store_path, exist_ok=True)

        cc_pickle = _join(store_path, "ccfile.p", logger=logger)
        if climate_change:
            cmip6_dir = _resolve_path(paths_cfg.get("cmip6_dir", ""), config_dir, logger=logger)
            cc_data = build_cc_data(cmip6_dir, cc_scenario, logger=logger)
            with open(cc_pickle, "wb") as handle:
                pickle.dump({cc_scenario: cc_data}, handle)

        run_indra(
            train=True,
            station_code=station_code,
            n_samples=n_samples,
            path_file_in=str(input_path),
            file_type=file_type,
            store_path=str(store_path),
            climate_change=climate_change,
            path_cc_file=str(cc_pickle),
            cc_scenario=cc_scenario,
            epoch=epoch,
            randseed=randseed,
            arma_params=arma_params,
            bounds=bounds,
            arma_caps=arma_caps,
            cache_models=cache_models,
            model_cache_path=model_cache_path,
            n_jobs=n_jobs,
            logger=logger,
        )

        for sample_idx in range(n_samples):
            output_path = _join(store_path, output_template.format(sample=sample_idx), logger=logger)
            run_indra(
                train=False,
                station_code=station_code,
                n_samples=n_samples,
                path_file_in=str(input_path),
                path_file_out=str(output_path),
                file_type=file_type,
                store_path=str(store_path),
                climate_change=climate_change,
                path_cc_file=str(cc_pickle),
                cc_scenario=cc_scenario,
                variant=sample_idx,
                logger=logger,
            )
            outputs.append(output_path)

    logger.info("run_from_config success")
    return outputs


def build_parser(logger=None) -> argparse.ArgumentParser:
    logger = get_logger(logger)
    parser = argparse.ArgumentParser(
        description="Run Indra with a TOML config file.",
        prog="indra",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config.toml",
    )
    logger.info("build_parser success")
    return parser


def main(argv: list[str] | None = None, logger=None) -> int:
    logger = logger or setup_logger()
    parser = build_parser(logger=logger)
    args = parser.parse_args(argv)
    config_path = os.path.abspath(args.config)
    config = load_config(config_path, logger=logger)
    run_from_config(config, os.path.dirname(config_path), logger=logger)
    logger.info("main success")
    return 0


def cli(logger=None) -> None:
    logger = logger or setup_logger()
    logger.info("cli start")
    raise SystemExit(main(sys.argv[1:], logger=logger))
