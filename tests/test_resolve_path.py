import os

import pytest

import indra.config as indra_config


def test_environment_variables_are_expanded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDRA_WEATHER_DIR", str(tmp_path))
    resolved = indra_config._resolve_path("${INDRA_WEATHER_DIR}/Historical/a.epw", "/elsewhere")
    assert resolved == os.path.join(str(tmp_path), "Historical", "a.epw")


def test_relative_paths_are_anchored_at_the_config(tmp_path) -> None:
    assert indra_config._resolve_path("data/a.epw", str(tmp_path)) == os.path.join(
        str(tmp_path), "data", "a.epw"
    )


def test_an_unset_variable_is_refused(monkeypatch) -> None:
    monkeypatch.delenv("INDRA_WEATHER_DIR", raising=False)
    with pytest.raises(ValueError, match="Unset environment variable"):
        indra_config._resolve_path("${INDRA_WEATHER_DIR}/a.epw", "/elsewhere")
