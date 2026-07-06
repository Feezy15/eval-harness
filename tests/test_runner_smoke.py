"""End-to-end smoke: the full episode matrix on the mock model, no keys, no network.

Pins the output contract (JSONL that
round-trips through EpisodeResult, a flat CSV summary, transcript structure, both
episode stop conditions, and replay determinism).
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import yaml

from collab_eval.config import load_config
from collab_eval.judge import MockJudge
from collab_eval.models.base import AgentModel
from collab_eval.models.mock import MockModel
from collab_eval.runner import run_episode, run_matrix
from collab_eval.tasks.toy import ToyTask
from collab_eval.types import EpisodeResult, Message, ModelResponse, Usage
from collab_eval.user_sim import STOP_SENTINEL, UserSimulator

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_YAML = REPO_ROOT / "configs" / "smoke.yaml"

# smoke.yaml: 1 task x 1 model x 3 effort levels x 2 seeds
EXPECTED_EPISODES = 6


def test_matrix_writes_jsonl_that_round_trips(tmp_path):
    cfg = load_config(SMOKE_YAML)
    results = run_matrix(cfg, output_dir=tmp_path)
    assert len(results) == EXPECTED_EPISODES

    lines = (tmp_path / "smoke.jsonl").read_text().splitlines()
    assert len(lines) == EXPECTED_EPISODES
    parsed = [EpisodeResult.model_validate_json(line) for line in lines]

    # Every (effort, seed) cell of the matrix ran exactly once.
    cells = {(r.task, r.model, r.effort, r.seed) for r in parsed}
    assert len(cells) == EXPECTED_EPISODES
    assert {r.effort for r in parsed} == {"passive", "moderate", "active_steering"}
    assert {r.seed for r in parsed} == {0, 1}


def test_matrix_writes_flat_csv_summary(tmp_path):
    cfg = load_config(SMOKE_YAML)
    run_matrix(cfg, output_dir=tmp_path)

    with (tmp_path / "smoke.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == EXPECTED_EPISODES
    expected_cols = {
        "run_name",
        "episode_id",
        "task",
        "model",
        "effort",
        "seed",
        "n_agent_turns",
        "n_user_turns",
        "temperature",
        "score",
        "rubric_version",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_s",
        "config_hash",
        "started_at",
    }
    assert expected_cols <= set(rows[0].keys())
    for row in rows:
        assert 0.0 <= float(row["score"]) <= 1.0
        assert float(row["cost_usd"]) > 0
        assert float(row["latency_s"]) > 0


def test_transcript_structure_turn_cap_and_totals(tmp_path):
    cfg = load_config(SMOKE_YAML)
    results = run_matrix(cfg, output_dir=tmp_path)

    for ep in results:
        roles = [m.role for m in ep.transcript]
        # Opens with the task framing and the (seeded) user goal…
        assert roles[:2] == ["system", "user"]
        # …then strictly alternates assistant/user: multi-turn, not a monologue.
        for i, role in enumerate(roles[2:]):
            assert role == ("assistant" if i % 2 == 0 else "user")

        # The mock user-sim never volunteers to stop, so the runner's hard cap
        # must be what ends the episode.
        n_agent = sum(1 for r in roles if r == "assistant")
        assert n_agent == cfg.max_turns

        # Totals are the sum of every logged call (agent + user-sim + judge):
        # cost accounting must have no untracked calls.
        recomputed = sum((t.usage for t in ep.turns), Usage.zero()) + ep.judge.usage
        assert ep.totals == recomputed


def test_temperature_variants_are_distinct_conditions(tmp_path):
    # Two entries sharing a base model but differing in temperature are two
    # experimental conditions: identity (episode_id, model column) and the
    # recorded temperature must keep their result rows apart, or aggregation
    # pools different sampling distributions into one curve.
    data = yaml.safe_load(SMOKE_YAML.read_text())
    data["models"] = [
        {"provider": "mock", "model": "mock-agent", "temperature": 0.0, "label": "agent-t0"},
        {"provider": "mock", "model": "mock-agent", "temperature": 1.0, "label": "agent-t1"},
    ]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data))
    results = run_matrix(load_config(cfg_path), output_dir=tmp_path)

    assert len(results) == 2 * EXPECTED_EPISODES
    ids = [ep.episode_id for ep in results]
    assert len(set(ids)) == len(ids)
    assert {ep.model for ep in results} == {"agent-t0", "agent-t1"}
    for ep in results:
        assert ep.temperature == (0.0 if ep.model == "agent-t0" else 1.0)

    with (tmp_path / "smoke.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {(row["model"], row["temperature"]) for row in rows} == {
        ("agent-t0", "0.0"),
        ("agent-t1", "1.0"),
    }


def test_rerun_is_deterministic(tmp_path):
    cfg = load_config(SMOKE_YAML)
    first = run_matrix(cfg, output_dir=tmp_path / "a")
    second = run_matrix(cfg, output_dir=tmp_path / "b")
    for ep_a, ep_b in zip(first, second, strict=True):
        assert ep_a.transcript == ep_b.transcript
        assert ep_a.judge.score == ep_b.judge.score
        assert ep_a.totals == ep_b.totals


class _AlwaysStopsModel(AgentModel):
    """Stub user-sim backend that immediately signals it has what it needs."""

    name = "stub:always-stops"

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=f"Looks good, thanks. {STOP_SENTINEL}"),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01),
        )


def test_user_sim_stop_signal_ends_episode_early():
    task = ToyTask()
    agent = MockModel(model="mock-agent", seed=0)
    user_sim = UserSimulator(model=_AlwaysStopsModel(), effort="passive")
    judge = MockJudge(model="mock-judge", rubric_version="v0")

    ep = run_episode(task=task, agent=agent, user_sim=user_sim, judge=judge, seed=0, max_turns=5)

    # One agent turn, then the user is satisfied — well under the cap.
    assert sum(1 for m in ep.transcript if m.role == "assistant") == 1
    # The stop signal itself is not a conversational turn: it must not leak
    # sentinel text into the transcript the judge scores.
    assert all(STOP_SENTINEL not in m.content for m in ep.transcript)
