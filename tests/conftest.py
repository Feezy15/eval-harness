"""Shared test scaffolding: the smoke config path, YAML mutate-and-reload
helpers, a small conversation builder, and the scripted stub backends.

Plain module-level names, not fixtures: none of this needs fixture semantics
(no setup/teardown, no request-scoped state), so a fixture would just add
indirection over a function call.
"""

import json
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


# One fixed usage for every scripted reply, so per-episode totals in tests are
# exact multiples of it — cost accounting stays assertable without arithmetic
# scattered across test files.
STUB_USAGE = Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01)


class ScriptedModel(AgentModel):
    """Stub backend with scripted replies: `contents` in order, the last one
    repeating, every conversation recorded in `.calls`.

    One stub covers what used to be four near-copies: a fixed reply is a
    one-item script; no arguments means "stop immediately" (the bare sentinel,
    for ending an episode after exactly one agent turn); a call count is
    `len(.calls)`; channel-isolation assertions read `.calls` contents; and a
    multi-item script exercises retry/repair paths that need the reply to
    change between calls.
    """

    name = "stub:scripted"
    model = "stub-model"
    temperature = 0.0

    def __init__(self, *contents: str):
        self._contents = contents or (STOP_SENTINEL,)
        self.calls: list[list[Message]] = []

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        self.calls.append(list(conversation))
        content = self._contents[min(len(self.calls), len(self._contents)) - 1]
        return ModelResponse(
            message=Message(role="assistant", content=content),
            usage=STUB_USAGE,
        )


def canned_checklist(met_flags: list[bool]) -> str:
    """A well-formed judge checklist response with the given verdicts."""
    return json.dumps(
        {
            "criteria": [
                {"index": i, "reasoning": f"reason {i}", "met": met}
                for i, met in enumerate(met_flags, start=1)
            ]
        }
    )


class FakeProviderModel(AgentModel):
    """Stand-in for a real provider's AgentModel: same constructor shape as
    MockModel/OpenAIModel/AnthropicModel, but no network and no API key —
    anything resolving providers through MODEL_REGISTRY just needs something
    registrable. Always replies with an all-met checklist sized to ToyTask's
    three criteria."""

    def __init__(self, model: str, seed: int, temperature: float, max_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.name = f"fake:{model}"

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=canned_checklist([True, True, True])),
            usage=STUB_USAGE,
        )


class FailingModel(AgentModel):
    """Provider stand-in whose calls always blow up: exercises per-episode
    exception isolation without depending on any real provider failure mode."""

    def __init__(self, model: str, seed: int, temperature: float, max_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.name = f"failing:{model}"

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        raise RuntimeError("scripted provider failure")


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
