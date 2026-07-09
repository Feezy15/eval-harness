"""Disk response cache for real LLM calls.

Exists purely to make reruns of a real-model matrix free and reproducible —
not to change what a run measures. Every call the harness caches is keyed on
the full request identity (provider, model, sampling params, seed) plus the
exact conversation, so a hit can only ever replay what an identical call
already produced.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from collab_eval.models.base import AgentModel
from collab_eval.types import Message, ModelResponse


def cache_key(key_params: Mapping[str, object], conversation: Sequence[Message]) -> str:
    """Sha256 hex over canonical (sorted-key) JSON of key_params + conversation.

    `seed` belongs in `key_params` (the caller's job, not this function's):
    real provider APIs can't truly seed sampling, so without it, replicate
    cells that share an identical message prefix would collide on one cache
    entry and collapse into a single sample instead of staying independent.
    """
    payload = {
        "key_params": dict(key_params),
        "conversation": [m.model_dump() for m in conversation],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    """One JSON file per cache key. The directory is created lazily (on first
    `put`) so a disabled or always-missing cache never touches the filesystem."""

    def __init__(self, dir: str | Path):
        self._dir = Path(dir)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return ModelResponse.model_validate_json(path.read_text())

    def put(self, key: str, response: ModelResponse) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(response.model_dump_json())


class CachedModel(AgentModel):
    """Composition wrapper: on a cache hit, returns the stored `ModelResponse`
    VERBATIM -- including its original usage (tokens, cost_usd, latency_s).

    Replaying the original usage rather than recomputing or zeroing it is the
    whole point: analysis (cost/latency-vs-utility curves in particular) must
    be invariant to whether a given run happened to hit the cache. The cache
    avoids re-spending; it must never change a measurement.
    """

    def __init__(self, inner: AgentModel, cache: ResponseCache, key_params: Mapping[str, object]):
        self._inner = inner
        self._cache = cache
        self._key_params = dict(key_params)
        self.name = inner.name
        self.model = inner.model
        self.temperature = inner.temperature

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        key = cache_key(self._key_params, conversation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        response = self._inner.next_turn(conversation)
        self._cache.put(key, response)
        return response
