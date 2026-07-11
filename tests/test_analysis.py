from datetime import datetime
from pathlib import Path

from collab_eval.analysis import (
    effort_manipulation,
    load_results,
    plot_utility_vs_effort,
    utility_by_effort,
)
from collab_eval.types import EffortLevel, EpisodeResult, JudgeScore, Message, TurnRecord, Usage
from conftest import conv

EFFORT_ORDER = ["passive", "moderate", "active_steering"]


def _episode(
    task: str,
    model: str,
    effort: EffortLevel,
    seed: int,
    score: float,
    turns: list[TurnRecord] | None = None,
) -> EpisodeResult:
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
        turns=turns or [],
        totals=usage,
        judge=JudgeScore(score=score, rationale="", rubric_version="v0", usage=usage),
        started_at=datetime(2026, 1, 1),
        config_hash="c",
        prompts_hash="p",
    )


def _turn(actor: str, content: str | None, output_tokens: int) -> TurnRecord:
    message = (
        None
        if content is None
        else Message(role="user" if actor == "user_sim" else "assistant", content=content)
    )
    return TurnRecord(
        turn_index=0,
        actor=actor,
        message=message,
        usage=Usage(input_tokens=1, output_tokens=output_tokens, cost_usd=0.0, latency_s=0.0),
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


def test_effort_manipulation_summarizes_sim_behavior_per_effort():
    def sim(content: str | None, tokens: int) -> TurnRecord:
        return _turn("user_sim", content, tokens)

    # Agent turns carry deliberately huge token counts: they must never leak
    # into the sim-behavior stats.
    def agent() -> TurnRecord:
        return _turn("agent", "agent reply", 999)

    episodes = [
        # Fed active-first to prove the output uses treatment order.
        _episode(
            "trip",
            "m",
            "active_steering",
            0,
            1.0,
            turns=[
                agent(),
                sim("change the hotel now", 10),
                agent(),
                sim("add a museum day", 10),
                agent(),
                sim("book the earlier train", 10),
            ],
        ),
        # A stop probe (message=None) is a costed sim call but not a user turn.
        _episode("trip", "m", "passive", 0, 0.0, turns=[agent(), sim("ok fine", 5), sim(None, 3)]),
        _episode("trip", "m", "passive", 1, 0.0, turns=[agent(), sim("ok fine", 5), sim(None, 3)]),
        # Same effort in another task is its own cell, not pooled in.
        _episode("other", "m", "passive", 0, 0.5, turns=[agent(), sim("a b c d e f", 20)]),
    ]

    df = effort_manipulation(episodes)

    trip = df[df["task"] == "trip"]
    # Treatment order, and no phantom row for the unobserved "moderate".
    assert list(trip["effort"]) == ["passive", "active_steering"]
    passive, active = trip.iloc[0], trip.iloc[1]

    assert passive["n_episodes"] == 2 and active["n_episodes"] == 1
    assert passive["mean_user_turns"] == 1.0 and active["mean_user_turns"] == 3.0
    assert passive["mean_words_per_user_message"] == 2.0
    assert active["mean_words_per_user_message"] == 4.0
    # Sim output tokens include the stop probe: it is real sim generation.
    assert passive["mean_sim_output_tokens"] == 8.0 and active["mean_sim_output_tokens"] == 30.0

    other = df[df["task"] == "other"].iloc[0]
    assert other["mean_user_turns"] == 1.0
    assert other["mean_words_per_user_message"] == 6.0
    assert other["mean_sim_output_tokens"] == 20.0


def test_load_and_plot_smoke_results(smoke_run, tmp_path: Path):
    """End-to-end on real smoke output: JSONL round-trips and the figure renders"""
    episodes = load_results(smoke_run.output_dir / "smoke.jsonl")
    assert [e.episode_id for e in episodes] == [e.episode_id for e in smoke_run.results]
    assert [e.judge.score for e in episodes] == [e.judge.score for e in smoke_run.results]

    out_path = tmp_path / "utility_vs_effort.png"
    assert plot_utility_vs_effort(episodes, out_path) == out_path
    assert out_path.stat().st_size > 0
