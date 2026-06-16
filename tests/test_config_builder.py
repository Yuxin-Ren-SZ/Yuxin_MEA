"""Tests for the Phase 3 config builder plumbing.

Covers:
- `validate_value` behavior across each ParamSpec.type
- new `ConfigManager` mutators (set_task_params, set_globals, validate_loaded,
  list_loaded_tasks)
- form_builder `collect_values` helper
- dashboard tolerates missing config; Settings page registers

Browser-driven callback tests are intentionally out of scope here.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from yuxin_mea.config import ConfigManager, ParamSpec, ValidationError, validate_value
from yuxin_mea.dashboard import build_app
from yuxin_mea.dashboard.components.form_builder import collect_values


# ---------------------------------------------------------------------------
# validate_value
# ---------------------------------------------------------------------------


def test_validate_int_within_bounds():
    spec = ParamSpec("int", 0, min=0, max=10)
    assert validate_value(spec, "5") == 5
    assert validate_value(spec, 0) == 0
    assert validate_value(spec, 10) == 10


def test_validate_int_below_min_rejected():
    spec = ParamSpec("int", 0, min=0)
    with pytest.raises(ValidationError, match="≥ 0"):
        validate_value(spec, -1)


def test_validate_int_above_max_rejected():
    spec = ParamSpec("int", 0, max=10)
    with pytest.raises(ValidationError, match="≤ 10"):
        validate_value(spec, 11)


def test_validate_float_coerces_str():
    spec = ParamSpec("float", 0.0)
    assert validate_value(spec, "3.14") == 3.14


def test_validate_str_choices_enforced():
    spec = ParamSpec("str", "a", choices=["a", "b", "c"])
    assert validate_value(spec, "b") == "b"
    with pytest.raises(ValidationError, match="Must be one of"):
        validate_value(spec, "z")


def test_validate_bool_coerces_strings():
    spec = ParamSpec("bool", False)
    assert validate_value(spec, "true") is True
    assert validate_value(spec, "False") is False
    assert validate_value(spec, "1") is True
    assert validate_value(spec, "0") is False


def test_validate_list_int_parses_csv():
    spec = ParamSpec("list_int", [])
    assert validate_value(spec, "1, 2, 3") == [1, 2, 3]
    assert validate_value(spec, [4, 5]) == [4, 5]


def test_validate_list_int_rejects_garbage():
    spec = ParamSpec("list_int", [])
    with pytest.raises(ValidationError, match="not a valid integer"):
        validate_value(spec, "1, foo, 3")


def test_validate_list_str_choices_enforced():
    spec = ParamSpec(
        "list_str", [],
        choices=["a", "b", "c"], multiselect=True,
    )
    assert validate_value(spec, "a, b") == ["a", "b"]
    with pytest.raises(ValidationError, match="Unknown choice"):
        validate_value(spec, "a, zzz")


def test_validate_nested_dict_recurses():
    spec = ParamSpec(
        "dict", {},
        nested_schema={
            "x": ParamSpec("int", 0, min=0),
            "y": ParamSpec("str", "ok"),
        },
    )
    assert validate_value(spec, {"x": 5, "y": "hi"}) == {"x": 5, "y": "hi"}
    # missing keys fall back to their nested defaults
    assert validate_value(spec, {}) == {"x": 0, "y": "ok"}


def test_validate_nested_dict_inner_failure_propagates():
    spec = ParamSpec(
        "dict", {},
        nested_schema={"x": ParamSpec("int", 0, min=0)},
    )
    with pytest.raises(ValidationError, match="≥ 0"):
        validate_value(spec, {"x": -1})


def test_validate_path_keeps_string():
    spec = ParamSpec("path", "")
    assert validate_value(spec, "/tmp/foo") == "/tmp/foo"
    assert validate_value(spec, None) == ""  # empty path is permitted


# ---------------------------------------------------------------------------
# ConfigManager mutators
# ---------------------------------------------------------------------------


def test_set_task_params_round_trips():
    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({
            "global": {"data_root": "/raw"},
            "tasks": {"preprocessing": {"bandpass_freq_min": 300}},
        }))
        cm = ConfigManager()
        cm.load(cfg)
        cm.set_task_params("preprocessing", {"bandpass_freq_min": 500,
                                             "bandpass_freq_max": 6000})
        cm.save(cfg)

        cm2 = ConfigManager()
        cm2.load(cfg)
        assert cm2.get_task_params("preprocessing") == {
            "bandpass_freq_min": 500,
            "bandpass_freq_max": 6000,
        }


def test_set_globals_replaces_atomically():
    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({
            "global": {"data_root": "/raw", "analysis_root": "/analysis"},
            "tasks": {},
        }))
        cm = ConfigManager()
        cm.load(cfg)
        # set_globals fully REPLACES — the original analysis_root is dropped.
        cm.set_globals({"data_root": "/new_raw"})
        cm.save(cfg)

        cm2 = ConfigManager()
        cm2.load(cfg)
        assert cm2.get_global("data_root") == "/new_raw"
        assert cm2.get_global("analysis_root") is None


def test_list_loaded_tasks_sorted():
    cm = ConfigManager()
    cm._task_loaded = {"sorting": {}, "preprocessing": {}, "analyzer": {}}
    assert cm.list_loaded_tasks() == ["analyzer", "preprocessing", "sorting"]


def test_validate_loaded_finds_unknown_keys():
    cm = ConfigManager()
    cm._task_loaded = {
        "preprocessing": {"bandpass_freq_min": 300, "removed_key": "x"},
        "sorting": {"sorter": "kilosort4"},
    }
    schemas = {
        "preprocessing": {"bandpass_freq_min": ParamSpec("int", 0)},
        "sorting": {"sorter": ParamSpec("str", "kilosort4")},
    }
    assert cm.validate_loaded(schemas) == {"preprocessing": ["removed_key"]}


def test_validate_loaded_clean_returns_empty():
    cm = ConfigManager()
    cm._task_loaded = {"preprocessing": {"bandpass_freq_min": 300}}
    schemas = {"preprocessing": {"bandpass_freq_min": ParamSpec("int", 0)}}
    assert cm.validate_loaded(schemas) == {}


# ---------------------------------------------------------------------------
# form_builder collect_values
# ---------------------------------------------------------------------------


def test_collect_values_parses_each_field():
    schema = {
        "n": ParamSpec("int", 0, min=0),
        "name": ParamSpec("str", "x"),
        "flag": ParamSpec("bool", False),
    }
    raw = {"n": "5", "name": "hi", "flag": [True]}  # bool widget emits a list
    parsed, errors = collect_values(schema, raw)
    assert errors == {}
    assert parsed == {"n": 5, "name": "hi", "flag": True}


def test_collect_values_returns_errors_dict():
    schema = {
        "n": ParamSpec("int", 0, min=0),
        "name": ParamSpec("str", "x", choices=["a", "b"]),
    }
    raw = {"n": "-1", "name": "zzz"}
    parsed, errors = collect_values(schema, raw)
    assert "n" in errors
    assert "name" in errors
    # Keys with errors must NOT appear in parsed
    assert "n" not in parsed
    assert "name" not in parsed


def test_collect_values_falls_back_to_spec_default():
    schema = {"n": ParamSpec("int", 42, min=0)}
    parsed, errors = collect_values(schema, {})  # nothing supplied
    assert parsed == {"n": 42}
    assert errors == {}


def test_collect_values_reconstructs_nested_dict_from_dotted_keys():
    """REGRESSION: nested-dict fields render with dotted IDs (`parent.child`).

    Pre-Phase-3.1, collect_values looked up by `parent` name in the flat
    dict, never found the key, and silently substituted spec.default — so
    every user edit to a nested field (e.g. SortingTask's
    `high_vram_sorter_kwargs.batch_size_seconds`) was discarded on Save.
    """
    schema = {
        "high_vram_sorter_kwargs": ParamSpec(
            "dict", {},
            nested_schema={
                "batch_size_seconds": ParamSpec("float", 2.0, min=0),
                "clear_cache": ParamSpec("bool", True),
            },
        ),
    }
    # Simulating what the Save callback sees: flat dict, dotted keys.
    raw = {
        "high_vram_sorter_kwargs.batch_size_seconds": "0.5",
        "high_vram_sorter_kwargs.clear_cache": [True],
    }
    parsed, errors = collect_values(schema, raw)
    assert errors == {}
    assert parsed == {
        "high_vram_sorter_kwargs": {
            "batch_size_seconds": 0.5,
            "clear_cache": True,
        },
    }


def test_collect_values_passes_through_when_no_dotted_keys():
    """Flat (non-nested) forms must still work after the reconstruction step."""
    schema = {"x": ParamSpec("int", 0)}
    parsed, errors = collect_values(schema, {"x": "5"})
    assert errors == {}
    assert parsed == {"x": 5}


# ---------------------------------------------------------------------------
# Dashboard tolerates missing config
# ---------------------------------------------------------------------------


def test_build_app_with_missing_config_does_not_raise():
    with TemporaryDirectory() as tmp:
        missing = Path(tmp) / "no_such.json"
        app = build_app(missing)
        ctx = app.server.config["YUXIN_MEA"]
        assert ctx["config_exists"] is False
        assert ctx["analysis_root"] is None
        assert ctx["data_root"] is None
        assert ctx["figure_root"] is None
        assert ctx["config_path"] == missing


def test_build_app_with_existing_config_flips_exists_flag():
    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({
            "global": {
                "data_root": str(Path(tmp) / "raw"),
                "analysis_root": str(tmp),
                "figure_root": str(Path(tmp) / "fig"),
            },
            "tasks": {},
        }))
        app = build_app(cfg)
        ctx = app.server.config["YUXIN_MEA"]
        assert ctx["config_exists"] is True
        assert ctx["figure_root"] == Path(tmp) / "fig"


def test_settings_page_registered():
    """After build_app, /settings is in dash.page_registry."""
    import dash

    with TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "pipeline_config.json"
        cfg.write_text(json.dumps({"global": {}, "tasks": {}}))
        build_app(cfg)
        paths = {p["path"] for p in dash.page_registry.values()}
        assert "/settings" in paths


# ---------------------------------------------------------------------------
# Example config round-trips through schemas
# ---------------------------------------------------------------------------


def test_example_config_round_trips_clean():
    """config/pipeline_config.example.json must validate against all task schemas."""
    from yuxin_mea.tasks import (
        AnalyzerTask, AutoCurationTask, AutoMergeTask, BurstDetectionTask,
        PreprocessingTask, SortingTask,
    )

    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "config" / "pipeline_config.example.json"
    assert example.exists(), f"Missing {example}"

    cm = ConfigManager()
    cm.load(example)
    schemas = {tc.task_name: tc.params_schema() for tc in [
        PreprocessingTask, SortingTask, AutoMergeTask, AnalyzerTask,
        AutoCurationTask, BurstDetectionTask,
    ]}
    assert cm.validate_loaded(schemas) == {}, (
        "Example config has keys not in any task schema. "
        "Regenerate it via the Phase 3 example-rebuild snippet."
    )
