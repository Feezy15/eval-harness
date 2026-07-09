"""Disk response cache: deterministic keys, verbatim replay, and the
run_matrix integration that makes real-API reruns free.

Caching is operational, not experiment-defining (`config.experiment_hash`
excludes it): a cached rerun of a matrix must reproduce the exact same
episodes as an uncached one, because a cache hit must be indistinguishable
from a cache miss in every measured field except `started_at`.
"""

from collections.abc import Sequence
from pathlib import Path

import yaml

from collab_eval.config import experiment_hash, load_config
from collab_eval.models.base import AgentModel
from collab_eval.models.cache import CachedModel, ResponseCache, cache_key
from collab_eval.runner import run_matrix
from collab_eval.types import Message, ModelResponse, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_YAML = REPO_ROOT / "configs" / "smoke.yaml"


def _conv() -> list[Message]:
    return [Message(role="system", content="s"), Message(role="user", content="hello")]


# --- cache_key ----------------------------------------------------------------


def test_cache_key_is_deterministic():
    params = {"provider": "openai", "model": "m", "temperature": 0.0, "seed": 0}
    assert cache_key(params, _conv()) == cache_key(dict(params), _conv())


def test_cache_key_changes_with_seed_alone():
    base = {"provider": "openai", "model": "m", "temperature": 0.0}
    k1 = cache_key({**base, "seed": 0}, _conv())
    k2 = cache_key({**base, "seed": 1}, _conv())
    assert k1 != k2


def test_cache_key_changes_with_messages():
    params = {"provider": "openai", "model": "m", "temperature": 0.0, "seed": 0}
    other = [Message(role="system", content="s"), Message(role="user", content="different")]
    assert cache_key(params, _conv()) != cache_key(params, other)


def test_cache_key_changes_with_temperature():
    conv = _conv()
    k1 = cache_key({"provider": "openai", "model": "m", "temperature": 0.0, "seed": 0}, conv)
    k2 = cache_key({"provider": "openai", "model": "m", "temperature": 0.5, "seed": 0}, conv)
    assert k1 != k2


def test_cache_key_changes_with_model():
    conv = _conv()
    k1 = cache_key({"provider": "openai", "model": "m1", "temperature": 0.0, "seed": 0}, conv)
    k2 = cache_key({"provider": "openai", "model": "m2", "temperature": 0.0, "seed": 0}, conv)
    assert k1 != k2


# --- ResponseCache --------------------------------------------------------------


def test_response_cache_miss_returns_none(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    assert cache.get("nonexistent-key") is None


def test_response_cache_round_trips_via_pydantic(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    response = ModelResponse(
        message=Message(role="assistant", content="hi"),
        usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.001, latency_s=0.5),
    )
    cache.put("key1", response)
    assert cache.get("key1") == response


def test_response_cache_creates_dir_lazily(tmp_path):
    cache_dir = tmp_path / "cache"
    ResponseCache(cache_dir)
    assert not cache_dir.exists()

    cache = ResponseCache(cache_dir)
    cache.put(
        "k",
        ModelResponse(
            message=Message(role="assistant", content="x"),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0, latency_s=0.0),
        ),
    )
    assert cache_dir.exists()


# --- CachedModel ----------------------------------------------------------------


class _CountingStub(AgentModel):
    """Stub inner model that counts real calls, so tests can prove a cache hit
    never reaches it."""

    def __init__(self):
        self.name = "stub:counting"
        self.model = "stub-model"
        self.temperature = 0.5
        self.calls = 0

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            message=Message(role="assistant", content=f"reply {self.calls}"),
            usage=Usage(
                input_tokens=10,
                output_tokens=self.calls,
                cost_usd=0.001 * self.calls,
                latency_s=1.23,
            ),
        )


def test_cached_model_forwards_name_model_temperature(tmp_path):
    inner = _CountingStub()
    cached = CachedModel(inner, ResponseCache(tmp_path / "cache"), key_params={"seed": 0})
    assert cached.name == inner.name
    assert cached.model == inner.model
    assert cached.temperature == inner.temperature


def test_cached_model_second_identical_call_replays_without_calling_inner(tmp_path):
    inner = _CountingStub()
    cached = CachedModel(
        inner, ResponseCache(tmp_path / "cache"), key_params={"provider": "stub", "seed": 0}
    )
    first = cached.next_turn(_conv())
    assert inner.calls == 1

    second = cached.next_turn(_conv())
    assert inner.calls == 1  # no second call reached the inner model
    # Usage replayed verbatim -- including the original latency/cost -- so
    # measurements stay invariant to whether a given run hit the cache.
    assert second == first


def test_cached_model_distinct_key_params_both_call_inner(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    inner = _CountingStub()
    cached_seed0 = CachedModel(inner, cache, key_params={"provider": "stub", "seed": 0})
    cached_seed1 = CachedModel(inner, cache, key_params={"provider": "stub", "seed": 1})
    cached_seed0.next_turn(_conv())
    cached_seed1.next_turn(_conv())
    assert inner.calls == 2


# --- run_matrix integration -------------------------------------------------------


def _with_cache(cfg, *, enabled: bool, cache_dir: Path):
    return cfg.model_copy(
        update={"cache": cfg.cache.model_copy(update={"enabled": enabled, "dir": str(cache_dir)})}
    )


def test_run_matrix_cache_enabled_matches_disabled_except_started_at(tmp_path):
    cfg = load_config(SMOKE_YAML)
    disabled = run_matrix(cfg, output_dir=tmp_path / "disabled")

    cfg_cached = _with_cache(cfg, enabled=True, cache_dir=tmp_path / "cachedir")
    enabled = run_matrix(cfg_cached, output_dir=tmp_path / "enabled")

    for off, on in zip(disabled, enabled, strict=True):
        assert off.model_dump(exclude={"started_at"}) == on.model_dump(exclude={"started_at"})


def test_run_matrix_hits_cache_on_second_run(tmp_path):
    cfg = load_config(SMOKE_YAML)
    cfg_cached = _with_cache(cfg, enabled=True, cache_dir=tmp_path / "cachedir")

    first = run_matrix(cfg_cached, output_dir=tmp_path / "run1")
    second = run_matrix(cfg_cached, output_dir=tmp_path / "run2")

    for a, b in zip(first, second, strict=True):
        assert a.model_dump(exclude={"started_at"}) == b.model_dump(exclude={"started_at"})


# --- experiment_hash is blind to the cache block ----------------------------------


def test_experiment_hash_equal_for_cache_absent_enabled_disabled(tmp_path):
    data = yaml.safe_load(SMOKE_YAML.read_text())
    data.pop("cache", None)
    variants = {
        "absent": dict(data),
        "disabled": {**data, "cache": {"enabled": False}},
        "enabled": {**data, "cache": {"enabled": True, "dir": "llm_cache"}},
    }
    hashes = {}
    for label, variant in variants.items():
        path = tmp_path / f"{label}.yaml"
        path.write_text(yaml.safe_dump(variant))
        hashes[label] = experiment_hash(load_config(path))
    assert len(set(hashes.values())) == 1, hashes
