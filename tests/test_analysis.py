from datetime import datetime
from pathlib import Path

from collab_eval.analysis import load_results, plot_utility_vs_effort, utility_by_effort
from collab_eval.types import EffortLevel, EpisodeResult, JudgeScore, Usage
from conftest import conv

EFFORT_ORDER = ["passive", "moderate", "active_steering"]


def _episode(task: str, model: str, effort: EffortLevel, seed: int, score: float) -> EpisodeResult:
    """Minimal valid episode; only the fields aggregation keys on vary."""
    usage = Usage.zero()
    return EpisodeResult(
        run_name="unit",
        episode_id=f"{task}__{model}__{effort}__s{seed}",
        task=task,
        model=model,
        temperature=0.0,
        effort=effort,
        seed=seed,
        user_context="",
        transcript=conv(),
        turns=[],
        totals=usage,
        judge=JudgeScore(score=score, rationale="", rubric_version="v0", usage=usage),
        started_at=datetime(2026, 1, 1),
        config_hash="c",
        prompts_hash="p",
    )


def test_utility_by_effort_aggregates_and_orders_by_treatment():
    # Deliberately fed in alphabetical-effort order to prove the output uses
    # treatment order (passive -> moderate -> active_steering), not sort order.
    episodes = [
        _episode("trip", "model-a", "active_steering", 0, 1.0),
        _episode("trip", "model-a", "active_steering", 1, 0.5),
        _episode("trip", "model-a", "moderate", 0, 0.5),
        _episode("trip", "model-a", "moderate", 1, 0.5),
        _episode("trip", "model-a", "passive", 0, 0.0),
        _episode("trip", "model-a", "passive", 1, 0.5),
        _episode("trip", "model-b", "passive", 0, 1.0),
        _episode("other", "model-a", "passive", 0, 0.25),
    ]

    df = utility_by_effort(episodes)

    # One row per observed (task, model, effort) cell — no phantom cells.
    assert len(df) == 5

    trip_a = df[(df["task"] == "trip") & (df["model"] == "model-a")]
    assert list(trip_a["effort"]) == EFFORT_ORDER
    assert list(trip_a["mean_score"]) == [0.25, 0.5, 0.75]
    assert list(trip_a["n"]) == [2, 2, 2]

    # Same (model, effort) in another task is its own cell, not pooled in.
    other = df[df["task"] == "other"]
    assert list(other["mean_score"]) == [0.25]
    assert list(other["n"]) == [1]


def test_load_and_plot_smoke_results(smoke_run, tmp_path: Path):
    """End-to-end on real smoke output: JSONL round-trips and the figure renders"""
    episodes = load_results(smoke_run.output_dir / "smoke.jsonl")
    assert [e.episode_id for e in episodes] == [e.episode_id for e in smoke_run.results]
    assert [e.judge.score for e in episodes] == [e.judge.score for e in smoke_run.results]

    out_path = tmp_path / "utility_vs_effort.png"
    assert plot_utility_vs_effort(episodes, out_path) == out_path
    assert out_path.stat().st_size > 0
