from __future__ import annotations

import os
import pytest

import indra.config as indra_main


# The weather-data root is machine-specific: set INDRA_WEATHER_DIR (see .env_example).
DATA_ROOT = os.environ.get("INDRA_WEATHER_DIR", "")
HISTORICAL = f"{DATA_ROOT}/Historical" if DATA_ROOT else ""

STATIONS = {
    "Atlanta": "USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP",
    "Glasgow": "GBR_SCT_Glasgow.Intl.AP.031400",
    "London": "GBR_ENG_London-Heathrow.Intl.AP",
    "Washington": "USA_VA_Dulles-Washington.Dulles.Intl.AP",
}


def _pick_epw(station_dir: str) -> str:
    epws = sorted([f for f in os.listdir(station_dir) if f.endswith(".epw")])
    assert epws, f"No EPW files found in {station_dir}"
    return f"{station_dir}/{epws[0]}"


def test_historical_epw_present() -> None:
    if not HISTORICAL or not os.path.exists(HISTORICAL):
        pytest.skip("Set INDRA_WEATHER_DIR to the weather-data folder.")
    for station_dir in STATIONS.values():
        station_path = f"{HISTORICAL}/{station_dir}"
        assert os.path.exists(station_path)
        _pick_epw(station_path)


@pytest.mark.slow
def test_run_from_config(tmp_path) -> None:
    if os.environ.get("INDRA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Set INDRA_RUN_INTEGRATION_TESTS=1 to run integration test.")
    if not HISTORICAL or not os.path.exists(HISTORICAL):
        pytest.skip("Historical data directory not available.")

    input_path = _pick_epw(f"{HISTORICAL}/{STATIONS['Atlanta']}")
    tmp_dir = str(tmp_path)
    config = {
        "run": {
            "station_code": "atl",
            "n_samples": 1,
            "file_type": "epw",
            "store_path": f"{tmp_dir}/outputs",
            "output_template": "atlanta_syn_{sample:02d}.epw",
            "climate_change": False,
        },
        "paths": {
            "input_path": str(input_path),
        },
    }

    outputs = indra_main.run_from_config(config, tmp_dir)
    assert outputs
    for output_path in outputs:
        assert os.path.exists(output_path)
