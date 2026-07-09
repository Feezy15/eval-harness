"""Config loading & validation.

The config *is* the experiment definition (no hardcoded params), so validation
errors here are the difference between "typo fails loudly at load time" and
"silently running a different experiment than you think".
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from collab_eval.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_YAML = REPO_ROOT / "configs" / "smoke.yaml"


def _smoke_dict() -> dict:
    with SMOKE_YAML.open() as f:
        return yaml.safe_load(f)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_smoke_config_loads_with_expected_values():
    cfg = load_config(SMOKE_YAML)
    assert cfg.run_name == "smoke"
    assert cfg.output_dir == Path("results")
    assert cfg.seeds == [0, 1]
    assert cfg.max_turns == 3
    assert [t.name for t in cfg.tasks] == ["toy"]
    assert cfg.models[0].provider == "mock"
    assert cfg.models[0].model == "mock-agent"
    assert cfg.models[0].temperature == 0.0
    assert cfg.user_sim.effort_levels == ["passive", "moderate", "active_steering"]
    assert cfg.user_sim.temperature == 0.7
    assert cfg.judge.rubric_version == "v0"


def test_unknown_top_level_key_rejected(tmp_path):
    data = _smoke_dict()
    data["max_turnz"] = 5  # typo'd knob must not be silently ignored
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_unknown_nested_key_rejected(tmp_path):
    data = _smoke_dict()
    data["models"][0]["temprature"] = 0.7
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_missing_required_field_rejected(tmp_path):
    data = _smoke_dict()
    del data["run_name"]
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_invalid_effort_level_rejected(tmp_path):
    data = _smoke_dict()
    data["user_sim"]["effort_levels"] = ["passive", "sleepy"]
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


# --- duplicate matrix cells ---------------------------------------------------
# A duplicate seed/effort/task/model entry re-runs the same cell and writes rows
# with identical identity — aggregation would silently pool them. Reject at load.


def test_duplicate_seeds_rejected(tmp_path):
    data = _smoke_dict()
    data["seeds"] = [0, 0, 1]
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(_write_yaml(tmp_path, data))


def test_duplicate_effort_levels_rejected(tmp_path):
    data = _smoke_dict()
    data["user_sim"]["effort_levels"] = ["passive", "passive"]
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(_write_yaml(tmp_path, data))


def test_duplicate_task_names_rejected(tmp_path):
    data = _smoke_dict()
    data["tasks"] = [{"name": "toy"}, {"name": "toy"}]
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(_write_yaml(tmp_path, data))


# --- model labels -------------------------------------------------------------
# The label is the model entry's identity in episode ids and result rows, so
# entries sharing a provider+model (e.g. a temperature ablation) must be told
# apart by explicit labels — otherwise their results are indistinguishable.


def test_model_label_defaults_to_provider_model():
    cfg = load_config(SMOKE_YAML)
    assert cfg.models[0].label == "mock:mock-agent"


def test_temperature_variants_without_labels_rejected(tmp_path):
    data = _smoke_dict()
    data["models"] = [
        {"provider": "mock", "model": "mock-agent", "temperature": 0.0},
        {"provider": "mock", "model": "mock-agent", "temperature": 1.0},
    ]
    with pytest.raises(ValidationError, match="label"):
        load_config(_write_yaml(tmp_path, data))


def test_temperature_variants_with_distinct_labels_accepted(tmp_path):
    data = _smoke_dict()
    data["models"] = [
        {"provider": "mock", "model": "mock-agent", "temperature": 0.0, "label": "agent-t0"},
        {"provider": "mock", "model": "mock-agent", "temperature": 1.0, "label": "agent-t1"},
    ]
    cfg = load_config(_write_yaml(tmp_path, data))
    assert [m.label for m in cfg.models] == ["agent-t0", "agent-t1"]


def test_duplicate_explicit_labels_rejected(tmp_path):
    data = _smoke_dict()
    data["models"] = [
        {"provider": "mock", "model": "agent-a", "label": "same"},
        {"provider": "mock", "model": "agent-b", "label": "same"},
    ]
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(_write_yaml(tmp_path, data))


def test_empty_label_rejected(tmp_path):
    # An empty label would make the runner fall back to the raw model name,
    # reopening the identity collision labels exist to prevent.
    data = _smoke_dict()
    data["models"][0]["label"] = ""
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_whitespace_label_rejected(tmp_path):
    data = _smoke_dict()
    data["models"][0]["label"] = "   "
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_negative_temperature_rejected(tmp_path):
    data = _smoke_dict()
    data["models"][0]["temperature"] = -0.5
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_negative_user_sim_temperature_rejected(tmp_path):
    data = _smoke_dict()
    data["user_sim"]["temperature"] = -0.5
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


# --- cache config ---------------------------------------------------------------


def test_cache_defaults_enabled_with_default_dir():
    cfg = load_config(SMOKE_YAML)
    # smoke.yaml explicitly opts out (mock model, zero cost, CI filesystem-clean).
    assert cfg.cache.enabled is False
    assert cfg.cache.dir == "llm_cache"


def test_cache_defaults_when_block_omitted(tmp_path):
    data = _smoke_dict()
    data.pop("cache", None)
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.cache.enabled is True
    assert cfg.cache.dir == "llm_cache"


def test_cache_unknown_key_rejected(tmp_path):
    data = _smoke_dict()
    data["cache"] = {"enabled": True, "diir": "x"}
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


# --- max_tokens ---------------------------------------------------------------


def test_model_max_tokens_defaults_to_none():
    cfg = load_config(SMOKE_YAML)
    assert cfg.models[0].max_tokens is None
    assert cfg.user_sim.max_tokens is None


def test_model_max_tokens_can_be_set(tmp_path):
    data = _smoke_dict()
    data["models"][0]["max_tokens"] = 512
    data["user_sim"]["max_tokens"] = 256
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.models[0].max_tokens == 512
    assert cfg.user_sim.max_tokens == 256


def test_model_max_tokens_below_one_rejected(tmp_path):
    data = _smoke_dict()
    data["models"][0]["max_tokens"] = 0
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))


def test_user_sim_max_tokens_below_one_rejected(tmp_path):
    data = _smoke_dict()
    data["user_sim"]["max_tokens"] = 0
    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path, data))
