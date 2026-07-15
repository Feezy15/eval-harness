import math
from datetime import datetime
from pathlib import Path

import pytest

from collab_eval.analysis import (
    agent_usage,
    cost_latency_by_effort,
    effort_manipulation,
    load_results,
    plot_cost_latency_frontier,
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
    totals: Usage | None = None,
    judge_usage: Usage | None = None,
) -> EpisodeResult:
    """Minimal valid episode; only the fields aggregation keys on vary."""
    zero = Usage.zero()
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
        totals=totals if totals is not None else zero,
        judge=JudgeScore(
            score=score,
            rationale="",
            rubric_version="v0",
            usage=judge_usage if judge_usage is not None else zero,
        ),
        started_at=datetime(2026, 1, 1),
        config_hash="c",
        prompts_hash="p",
    )


def _turn(
    actor: str,
    content: str | None,
    output_tokens: int,
    cost_usd: float = 0.0,
    latency_s: float = 0.0,
) -> TurnRecord:
    message = (
        None
        if content is None
        else Message(role="user" if actor == "user_sim" else "assistant", content=content)
    )
    return TurnRecord(
        turn_index=0,
        actor=actor,
        message=message,
        usage=Usage(
            input_tokens=1, output_tokens=output_tokens, cost_usd=cost_usd, latency_s=latency_s
        ),
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


def test_agent_usage_sums_only_agent_turns():
    # Distinct, hand-computable values per actor so a leak from any other
    # actor's usage into the sum would change the assertion.
    agent_turns = [
        _turn("agent", "hi", 10, cost_usd=1.0, latency_s=0.1),
        _turn("agent", "there", 20, cost_usd=2.0, latency_s=0.2),
    ]
    user_turns = [_turn("user_sim", "ok", 5, cost_usd=3.0, latency_s=0.3)]
    judge_usage = Usage(input_tokens=9, output_tokens=9, cost_usd=4.0, latency_s=0.4)
    episode = _episode(
        "trip",
        "model-a",
        "passive",
        0,
        0.5,
        turns=agent_turns + user_turns,
        judge_usage=judge_usage,
    )

    usage = agent_usage(episode)

    assert usage.input_tokens == 2
    assert usage.output_tokens == 30
    assert usage.cost_usd == pytest.approx(3.0)
    assert usage.latency_s == pytest.approx(0.3)


def test_cost_latency_by_effort_ratio_of_means_and_zero_cost_nan():
    def agent_turn(cost: float, latency: float) -> TurnRecord:
        return _turn("agent", "reply", 10, cost_usd=cost, latency_s=latency)

    def totals(cost: float, latency: float) -> Usage:
        return Usage(input_tokens=1, output_tokens=1, cost_usd=cost, latency_s=latency)

    episodes = [
        # Fed out of treatment order to prove the output uses EFFORT_ORDER, not
        # arrival order (mirrors the utility_by_effort ordering test).
        _episode(
            "trip",
            "model-a",
            "active_steering",
            0,
            0.9,
            turns=[agent_turn(3.0, 1.5)],
            totals=totals(30.0, 15.0),
        ),
        _episode(
            "trip",
            "model-a",
            "moderate",
            0,
            0.6,
            turns=[agent_turn(2.0, 1.0)],
            totals=totals(20.0, 10.0),
        ),
        # Two passive episodes with different scores AND different agent costs:
        # ratio-of-means (0.5/2.0=0.25) must differ from the mean of per-episode
        # ratios (0/1=0.0, 1/3=0.333 -> mean 0.1667), proving which arithmetic ran.
        _episode(
            "trip",
            "model-a",
            "passive",
            0,
            0.0,
            turns=[agent_turn(1.0, 0.5)],
            totals=totals(10.0, 5.0),
        ),
        _episode(
            "trip",
            "model-a",
            "passive",
            1,
            1.0,
            turns=[agent_turn(3.0, 1.5)],
            totals=totals(30.0, 15.0),
        ),
        # Zero agent cost/latency (mock-run shape) -> NaN ratios, not inf/crash.
        _episode(
            "trip",
            "model-b",
            "passive",
            0,
            0.8,
            turns=[agent_turn(0.0, 0.0)],
            totals=totals(0.0, 0.0),
        ),
        # Same effort in another task is its own cell, not pooled into trip's passive.
        _episode(
            "other",
            "model-a",
            "passive",
            0,
            0.25,
            turns=[agent_turn(5.0, 2.5)],
            totals=totals(50.0, 25.0),
        ),
    ]

    df = cost_latency_by_effort(episodes)

    trip_a = df[(df["task"] == "trip") & (df["model"] == "model-a")]
    assert list(trip_a["effort"]) == EFFORT_ORDER

    passive = trip_a.iloc[0]
    assert passive["n"] == 2
    assert passive["mean_score"] == pytest.approx(0.5)
    assert passive["mean_agent_cost_usd"] == pytest.approx(2.0)
    assert passive["mean_agent_latency_s"] == pytest.approx(1.0)
    assert passive["mean_total_cost_usd"] == pytest.approx(20.0)
    assert passive["utility_per_dollar"] == pytest.approx(0.25)
    assert passive["utility_per_second"] == pytest.approx(0.5)
    # Ratio-of-means, not mean-of-ratios: the two arithmetics disagree here.
    assert passive["utility_per_dollar"] != pytest.approx(0.5 * (0.0 / 1.0 + 1.0 / 3.0))

    zero_cost = df[(df["task"] == "trip") & (df["model"] == "model-b")].iloc[0]
    assert math.isnan(zero_cost["utility_per_dollar"])
    assert math.isnan(zero_cost["utility_per_second"])

    other = df[df["task"] == "other"]
    assert list(other["mean_score"]) == [0.25]
    assert other.iloc[0]["utility_per_dollar"] == pytest.approx(0.05)


def test_load_and_plot_smoke_results(smoke_run, tmp_path: Path):
    """End-to-end on real smoke output: JSONL round-trips and the figure renders"""
    episodes = load_results(smoke_run.output_dir / "smoke.jsonl")
    assert [e.episode_id for e in episodes] == [e.episode_id for e in smoke_run.results]
    assert [e.judge.score for e in episodes] == [e.judge.score for e in smoke_run.results]

    out_path = tmp_path / "utility_vs_effort.png"
    frontier_out_path = tmp_path / "cost_latency_frontier.png"
    assert plot_utility_vs_effort(episodes, out_path) == out_path
    assert plot_cost_latency_frontier(episodes, frontier_out_path) == frontier_out_path
    assert out_path.stat().st_size > 0
