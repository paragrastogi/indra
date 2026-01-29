from __future__ import annotations

from pathlib import Path

from indra import main as indra_main


def test_run_from_config_multiple_stations(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run_indra(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(indra_main, "run_indra", fake_run_indra)

    config = {
        "run": {
            "n_samples": 2,
            "file_type": "epw",
            "climate_change": False,
        },
        "stations": [
            {
                "run": {
                    "station_code": "aaa",
                    "store_path": str(tmp_path / "a"),
                    "output_template": "a_{sample:02d}.epw",
                },
                "paths": {"input_path": "a.epw"},
            },
            {
                "run": {
                    "station_code": "bbb",
                    "store_path": str(tmp_path / "b"),
                    "output_template": "b_{sample:02d}.epw",
                },
                "paths": {"input_path": "b.epw"},
            },
        ],
    }

    outputs = indra_main.run_from_config(config, tmp_path)

    assert len(outputs) == 4
    assert outputs[0].name == "a_00.epw"
    assert outputs[1].name == "a_01.epw"
    assert outputs[2].name == "b_00.epw"
    assert outputs[3].name == "b_01.epw"

    assert len(calls) == 6  # 2 stations -> 2 train + 4 sample calls
    assert calls[0]["train"] is True
    assert calls[1]["train"] is True
