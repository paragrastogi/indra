from __future__ import annotations

from pathlib import Path

import os
import pytest

from indra import main as indra_main


DATA_ROOT = Path(
    "/Users/prastogi/Library/CloudStorage/OneDrive-Personal/ASHRAE/Handbook/2025/WeatherData"
)
HISTORICAL = DATA_ROOT / "Historical"

STATIONS = {
    "Atlanta": "USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP",
    "Glasgow": "GBR_SCT_Glasgow.Intl.AP.031400",
    "London": "GBR_ENG_London-Heathrow.Intl.AP",
    "Washington": "USA_VA_Dulles-Washington.Dulles.Intl.AP",
}


def _pick_epw(station_dir: Path) -> Path:
    epws = sorted(station_dir.glob("*.epw"))
    assert epws, f"No EPW files found in {station_dir}"
    return epws[0]


def test_historical_epw_present() -> None:
    if not HISTORICAL.exists():
        pytest.skip("Historical data directory not available.")
    for station_dir in STATIONS.values():
        assert (HISTORICAL / station_dir).exists()
        _pick_epw(HISTORICAL / station_dir)


@pytest.mark.slow
def test_run_from_config(tmp_path: Path) -> None:
    if os.environ.get("INDRA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Set INDRA_RUN_INTEGRATION_TESTS=1 to run integration test.")
    if not HISTORICAL.exists():
        pytest.skip("Historical data directory not available.")

    input_path = _pick_epw(HISTORICAL / STATIONS["Atlanta"])
    config = {
        "run": {
            "station_code": "atl",
            "n_samples": 1,
            "file_type": "epw",
            "store_path": str(tmp_path / "outputs"),
            "output_template": "atlanta_syn_{sample:02d}.epw",
            "climate_change": False,
        },
        "paths": {
            "input_path": str(input_path),
        },
    }

    outputs = indra_main.run_from_config(config, tmp_path)
    assert outputs
    for output_path in outputs:
        assert output_path.exists()
