"""Shared test scaffolding: the smoke config path, YAML mutate-and-reload
helpers, a small conversation builder, and a reusable "stops immediately"
stub backend — duplicated near-identically across the suite before this file
existed.

Plain module-level names, not fixtures: none of this needs fixture semantics
(no setup/teardown, no request-scoped state), so a fixture would just add
indirection over a function call.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from collab_eval.config import load_config
from collab_eval.models.base import AgentModel
from collab_eval.runner import run_matrix
from collab_eval.types import EpisodeResult, Message, ModelResponse, Usage
from collab_eval.user_sim import STOP_SENTINEL

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_YAML = REPO_ROOT / "configs" / "smoke.yaml"


def smoke_dict() -> dict:
    with SMOKE_YAML.open() as f:
        return yaml.safe_load(f)


def _expected_episodes() -> int:
    data = smoke_dict()
    return (
        len(data["tasks"])
        * len(data["models"])
        * len(data["user_sim"]["effort_levels"])
        * len(data["seeds"])
    )


EXPECTED_EPISODES = _expected_episodes()


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def conv(system: str = "s", user: str = "hello") -> list[Message]:
    return [Message(role="system", content=system), Message(role="user", content=user)]


class AlwaysStopsModel(AgentModel):
    """Stub backend that signals stop on its first reply.

    Used to end an episode after exactly one agent turn without depending on
    a real user-sim model's judgment. `content` defaults to the bare sentinel;
    pass a wrapped/prefixed variant to exercise the sentinel-detection edges
    that `_signals_stop` (not this stub) is responsible for.
    """

    name = "stub:always-stops"
    model = "stub-user"
    temperature = 0.0

    def __init__(self, content: str = STOP_SENTINEL):
        self._content = content

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=self._content),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01),
        )


class RecordingModel(AgentModel):
    """Stub backend that replies with a fixed message and records every
    conversation it's called with — used to assert on exactly what a given
    channel (agent vs. user-sim backend) was shown, e.g. that private
    per-consumer context (`Task.user_context`, `Task.judge_context`) never
    crosses into a channel it doesn't belong to.
    """

    name = "stub:recording"
    model = "stub-model"
    temperature = 0.0

    def __init__(self, reply: str = STOP_SENTINEL):
        self._reply = reply
        self.received: list[list[Message]] = []

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        self.received.append(list(conversation))
        return ModelResponse(
            message=Message(role="assistant", content=self._reply),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01),
        )


@dataclass(frozen=True)
class SmokeRun:
    """A single, already-executed matrix run and where it wrote its output."""

    results: list[EpisodeResult]
    output_dir: Path


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> SmokeRun:
    """Runs the unmodified smoke config through `run_matrix` exactly once for
    the whole session, so every test that only *reads* the resulting episodes
    (or the JSONL/CSV files) doesn't pay for its own matrix execution.

    Contract: read-only. Do not mutate `.results` or its `EpisodeResult`
    objects, and do not write into `.output_dir` — either would leak state
    into every other test that shares this fixture. A test that needs a
    modified config (different models, temperature, telemetry, ...) or that
    checks the effects of running the matrix twice must call `run_matrix`
    itself rather than adapt this fixture.
    """
    output_dir = tmp_path_factory.mktemp("smoke_run")
    results = run_matrix(load_config(SMOKE_YAML), output_dir=output_dir)
    return SmokeRun(results=results, output_dir=output_dir)
