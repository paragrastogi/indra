from __future__ import annotations

import indra.main as indra_main


def test_run_from_config_multiple_stations(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run_indra(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(indra_main, "run_indra", fake_run_indra)

    tmp_dir = str(tmp_path)
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
                    "store_path": f"{tmp_dir}/a",
                    "output_template": "a_{sample:02d}.epw",
                },
                "paths": {"input_path": "a.epw"},
            },
            {
                "run": {
                    "station_code": "bbb",
                    "store_path": f"{tmp_dir}/b",
                    "output_template": "b_{sample:02d}.epw",
                },
                "paths": {"input_path": "b.epw"},
            },
        ],
    }

    outputs = indra_main.run_from_config(config, tmp_dir)

    assert len(outputs) == 4
    assert outputs[0].split("/")[-1] == "a_00.epw"
    assert outputs[1].split("/")[-1] == "a_01.epw"
    assert outputs[2].split("/")[-1] == "b_00.epw"
    assert outputs[3].split("/")[-1] == "b_01.epw"

    assert len(calls) == 6  # 2 stations -> 2 train + 4 sample calls
    assert sum(1 for call in calls if call["train"] is True) == 2
