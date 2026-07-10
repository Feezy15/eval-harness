"""Config loading & validation.

The config *is* the experiment definition (no hardcoded params), so validation
errors here are the difference between "typo fails loudly at load time" and
"silently running a different experiment than you think".
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from collab_eval.config import experiment_hash, load_config
from conftest import SMOKE_YAML, smoke_dict, write_yaml


def test_smoke_config_loads_with_expected_values():
    cfg = load_config(SMOKE_YAML)
    assert cfg.run_name == "smoke"
    assert cfg.output_dir == Path("results")
    assert cfg.seeds == [0, 1]
    assert cfg.max_turns == 3
    assert [t.name for t in cfg.tasks] == ["toy", "trip_planning"]
    assert cfg.models[0].provider == "mock"
    assert cfg.models[0].model == "mock-agent"
    assert cfg.models[0].temperature == 0.0
    assert cfg.user_sim.effort_levels == ["passive", "moderate", "active_steering"]
    assert cfg.user_sim.temperature == 0.7
    assert cfg.judge.rubric_version == "v0"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.__setitem__("max_turnz", 5),  # typo'd knob
        lambda data: data["models"][0].__setitem__("temprature", 0.7),
        lambda data: data.__setitem__("cache", {"enabled": True, "diir": "x"}),
    ],
    ids=["top_level", "nested", "cache_block"],
)
def test_unknown_key_rejected(tmp_path, mutate):
    data = smoke_dict()
    mutate(data)
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_missing_required_field_rejected(tmp_path):
    data = smoke_dict()
    del data["run_name"]
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_invalid_effort_level_rejected(tmp_path):
    data = smoke_dict()
    data["user_sim"]["effort_levels"] = ["passive", "sleepy"]
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


# --- duplicate matrix cells ---------------------------------------------------
# A duplicate seed/effort/task/model entry re-runs the same cell and writes rows
# with identical identity — aggregation would silently pool them. Reject at load.


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.__setitem__("seeds", [0, 0, 1]),
        lambda data: data["user_sim"].__setitem__("effort_levels", ["passive", "passive"]),
        lambda data: data.__setitem__("tasks", [{"name": "toy"}, {"name": "toy"}]),
        lambda data: data.__setitem__(
            "models",
            [
                {"provider": "mock", "model": "agent-a", "label": "same"},
                {"provider": "mock", "model": "agent-b", "label": "same"},
            ],
        ),
    ],
    ids=["seeds", "effort_levels", "task_names", "explicit_labels"],
)
def test_duplicate_matrix_cell_rejected(tmp_path, mutate):
    data = smoke_dict()
    mutate(data)
    with pytest.raises(ValidationError, match="duplicate"):
        load_config(write_yaml(tmp_path, data))


# --- model labels -------------------------------------------------------------
# The label is the model entry's identity in episode ids and result rows, so
# entries sharing a provider+model (e.g. a temperature ablation) must be told
# apart by explicit labels — otherwise their results are indistinguishable.


def test_model_label_defaults_to_provider_model():
    cfg = load_config(SMOKE_YAML)
    assert cfg.models[0].label == "mock:mock-agent"


def test_temperature_variants_without_labels_rejected(tmp_path):
    data = smoke_dict()
    data["models"] = [
        {"provider": "mock", "model": "mock-agent", "temperature": 0.0},
        {"provider": "mock", "model": "mock-agent", "temperature": 1.0},
    ]
    with pytest.raises(ValidationError, match="label"):
        load_config(write_yaml(tmp_path, data))


def test_temperature_variants_with_distinct_labels_accepted(tmp_path):
    data = smoke_dict()
    data["models"] = [
        {"provider": "mock", "model": "mock-agent", "temperature": 0.0, "label": "agent-t0"},
        {"provider": "mock", "model": "mock-agent", "temperature": 1.0, "label": "agent-t1"},
    ]
    cfg = load_config(write_yaml(tmp_path, data))
    assert [m.label for m in cfg.models] == ["agent-t0", "agent-t1"]


@pytest.mark.parametrize("label", ["", "   "], ids=["empty", "whitespace"])
def test_blank_label_rejected(tmp_path, label):
    # An empty/whitespace label would make the runner fall back to the raw
    # model name, reopening the identity collision labels exist to prevent.
    data = smoke_dict()
    data["models"][0]["label"] = label
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


@pytest.mark.parametrize("block", ["models", "user_sim"], ids=["model", "user_sim"])
def test_negative_temperature_rejected(tmp_path, block):
    data = smoke_dict()
    if block == "models":
        data["models"][0]["temperature"] = -0.5
    else:
        data["user_sim"]["temperature"] = -0.5
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


# --- cache config ---------------------------------------------------------------


def test_cache_defaults_enabled_with_default_dir():
    cfg = load_config(SMOKE_YAML)
    # smoke.yaml explicitly opts out (mock model, zero cost, CI filesystem-clean).
    assert cfg.cache.enabled is False
    assert cfg.cache.dir == "llm_cache"


def test_cache_defaults_when_block_omitted(tmp_path):
    data = smoke_dict()
    data.pop("cache", None)
    cfg = load_config(write_yaml(tmp_path, data))
    assert cfg.cache.enabled is True
    assert cfg.cache.dir == "llm_cache"


# --- max_tokens ---------------------------------------------------------------


def test_model_max_tokens_defaults_to_none():
    cfg = load_config(SMOKE_YAML)
    assert cfg.models[0].max_tokens is None
    assert cfg.user_sim.max_tokens is None


@pytest.mark.parametrize("block", ["models", "user_sim"], ids=["model", "user_sim"])
def test_max_tokens_below_one_rejected(tmp_path, block):
    data = smoke_dict()
    if block == "models":
        data["models"][0]["max_tokens"] = 0
    else:
        data["user_sim"]["max_tokens"] = 0
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


# --- experiment_hash: operational blocks (telemetry, cache) don't feed identity --


@pytest.mark.parametrize(
    "block",
    ["telemetry", "cache"],
)
def test_experiment_hash_blind_to_operational_block(tmp_path, block):
    # A run traced/cached and a run untraced/uncached are the SAME experiment:
    # these blocks must not feed the config hash, or flipping an operational
    # knob would split otherwise-identical results across two experiment
    # identities.
    data = smoke_dict()
    data.pop(block, None)
    enabled_extra = {"exporter": "console"} if block == "telemetry" else {"dir": "llm_cache"}
    variants = {
        "absent": dict(data),
        "disabled": {**data, block: {"enabled": False}},
        "enabled": {**data, block: {"enabled": True, **enabled_extra}},
    }
    hashes = {}
    for label, variant in variants.items():
        path = tmp_path / f"{label}.yaml"
        path.write_text(yaml.safe_dump(variant))
        hashes[label] = experiment_hash(load_config(path))
    assert len(set(hashes.values())) == 1, hashes

    # ...but a real experiment knob still changes the hash.
    changed = {**data, "max_turns": data["max_turns"] + 1}
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    assert experiment_hash(load_config(path)) != hashes["absent"]
