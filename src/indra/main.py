import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from .indra import indra as run_indra

import tomllib



def _read_cmip6_variable(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Year" not in df.columns or "Day" not in df.columns:
        raise ValueError(f"CMIP6 file missing Year/Day columns: {path}")
    date = pd.to_datetime(df["Year"].astype(str), format="%Y") + pd.to_timedelta(
        df["Day"].astype(int) - 1, unit="D"
    )
    df = df.drop(columns=["Year", "Day"])
    df.index = date
    return df


def build_cc_data(cmip6_dir: Path, scenario: str) -> pd.DataFrame:
    tasmax_path = cmip6_dir / f"{cmip6_dir.name.lower()}_tasmax_{scenario}.csv"
    tasmin_path = cmip6_dir / f"{cmip6_dir.name.lower()}_tasmin_{scenario}.csv"
    hurs_path = cmip6_dir / f"{cmip6_dir.name.lower()}_hurs_{scenario}.csv"

    if not tasmax_path.exists():
        raise FileNotFoundError(f"Missing tasmax file: {tasmax_path}")
    if not tasmin_path.exists():
        raise FileNotFoundError(f"Missing tasmin file: {tasmin_path}")
    if not hurs_path.exists():
        raise FileNotFoundError(f"Missing hurs file: {hurs_path}")

    tasmax = _read_cmip6_variable(tasmax_path)
    tasmin = _read_cmip6_variable(tasmin_path)
    hurs = _read_cmip6_variable(hurs_path)

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

    return pd.concat(frames, keys=models, names=["model", "time"])


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _normalize_runs(config: dict) -> list[dict]:
    run_defaults = config.get("run", {})
    paths_defaults = config.get("paths", {})
    stations = config.get("stations")
    if not stations:
        return [{"run": run_defaults, "paths": paths_defaults}]
    normalized = []
    for station in stations:
        run_cfg = {**run_defaults, **station.get("run", {})}
        paths_cfg = {**paths_defaults, **station.get("paths", {})}
        normalized.append({"run": run_cfg, "paths": paths_cfg})
    return normalized


def _resolve_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def run_from_config(config: dict, config_dir: Path) -> list[Path]:
    outputs: list[Path] = []

    for entry in _normalize_runs(config):
        run_cfg = entry["run"]
        paths_cfg = entry["paths"]

        station_code = run_cfg.get("station_code", "abc")
        n_samples = int(run_cfg.get("n_samples", 1))
        file_type = run_cfg.get("file_type", "epw")
        store_path = Path(run_cfg.get("store_path", f"outputs/{station_code}"))
        output_template = run_cfg.get(
            "output_template", f"{station_code}_syn_{{sample:02d}}.{file_type}"
        )
        climate_change = bool(run_cfg.get("climate_change", False))
        cc_scenario = run_cfg.get("cc_scenario", "ssp585")
        epoch = run_cfg.get("epoch")
        randseed = run_cfg.get("randseed")
        arma_params = run_cfg.get("arma_params")
        bounds = run_cfg.get("bounds")

        input_path = _resolve_path(paths_cfg.get("input_path", ""), config_dir)

        store_path = (
            store_path if store_path.is_absolute() else (config_dir / store_path)
        )
        store_path.mkdir(parents=True, exist_ok=True)

        cc_pickle = store_path / "ccfile.p"
        if climate_change:
            cmip6_dir = _resolve_path(paths_cfg.get("cmip6_dir", ""), config_dir)
            cc_data = build_cc_data(cmip6_dir, cc_scenario)
            with cc_pickle.open("wb") as handle:
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
        )

        for sample_idx in range(n_samples):
            output_path = store_path / output_template.format(sample=sample_idx)
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
            )
            outputs.append(output_path)

    return outputs


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    run_from_config(config, config_path.parent)
    return 0


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))
