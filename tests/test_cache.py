"""Disk response cache: deterministic keys, verbatim replay, and the
run_matrix integration that makes real-API reruns free.

Caching is operational, not experiment-defining (`config.experiment_hash`
excludes it): a cached rerun of a matrix must reproduce the exact same
episodes as an uncached one, because a cache hit must be indistinguishable
from a cache miss in every measured field except `started_at`.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from collab_eval.config import load_config
from collab_eval.models.base import AgentModel
from collab_eval.models.cache import CachedModel, ResponseCache, cache_key
from collab_eval.runner import run_matrix
from collab_eval.types import Message, ModelResponse, Usage
from conftest import SMOKE_YAML, conv


def _conv() -> list[Message]:
    return conv()


# --- cache_key ----------------------------------------------------------------


def test_cache_key_is_deterministic():
    params = {"provider": "openai", "model": "m", "temperature": 0.0, "seed": 0}
    assert cache_key(params, _conv()) == cache_key(dict(params), _conv())


_BASE_PARAMS = {"provider": "openai", "model": "m", "temperature": 0.0, "seed": 0}


@pytest.mark.parametrize(
    "params_a, conv_a, params_b, conv_b",
    [
        (_BASE_PARAMS, _conv(), {**_BASE_PARAMS, "seed": 1}, _conv()),
        (
            _BASE_PARAMS,
            _conv(),
            _BASE_PARAMS,
            [Message(role="system", content="s"), Message(role="user", content="different")],
        ),
        (_BASE_PARAMS, _conv(), {**_BASE_PARAMS, "temperature": 0.5}, _conv()),
        (_BASE_PARAMS, _conv(), {**_BASE_PARAMS, "model": "m2"}, _conv()),
    ],
    ids=["seed", "messages", "temperature", "model"],
)
def test_cache_key_changes_with(params_a, conv_a, params_b, conv_b):
    assert cache_key(params_a, conv_a) != cache_key(params_b, conv_b)


# --- ResponseCache --------------------------------------------------------------


def test_response_cache_miss_returns_none(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    assert cache.get("nonexistent-key") is None


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


def test_run_matrix_cache_disabled_enabled_first_and_enabled_second_all_match(tmp_path):
    # Cache-disabled, first cache-enabled run, and second (cache-hitting) run
    # must all produce field-for-field identical episodes (bar `started_at`):
    # a cache hit is indistinguishable from a miss in every measured field.
    cfg = load_config(SMOKE_YAML)
    disabled = run_matrix(cfg, output_dir=tmp_path / "disabled")

    cfg_cached = _with_cache(cfg, enabled=True, cache_dir=tmp_path / "cachedir")
    enabled_first = run_matrix(cfg_cached, output_dir=tmp_path / "enabled1")
    enabled_second = run_matrix(cfg_cached, output_dir=tmp_path / "enabled2")

    for off, on1, on2 in zip(disabled, enabled_first, enabled_second, strict=True):
        dumped = off.model_dump(exclude={"started_at"})
        assert on1.model_dump(exclude={"started_at"}) == dumped
        assert on2.model_dump(exclude={"started_at"}) == dumped
