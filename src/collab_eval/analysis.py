"""Utility-vs-effort analysis: turns a results JSONL into a curve.

Reads the JSONL (the source of truth — the CSV is a derived summary), aggregates
judge scores by (task, model, effort), and renders one figure per run.
"""

import argparse
from pathlib import Path
from typing import get_args

import matplotlib

matplotlib.use("Agg")  # headless: this module must render without a display (CI, smoke run)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from collab_eval.types import EffortLevel, EpisodeResult, Usage  # noqa: E402

EFFORT_ORDER: list[str] = list(get_args(EffortLevel))

# Fixed slot order, not a generated/cycled palette: colors must stay stable and
# validated (contrast-checked) across runs regardless of which models happen to
# appear, and must never balloon past what's been checked for readability.
_MODEL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
# Darker ink per slot for direct line-end labels — slots 2-3 fail 3:1 contrast
# on white at the base hue, so labels use a hand-picked darker variant instead.
_MODEL_LABEL_COLORS = ["#1a4d8f", "#0d6b48", "#8a6400", "#005200"]
_MODEL_MARKERS = ["o", "s", "^", "D"]

_GRID_COLOR = "#e1e0d9"
_SPINE_COLOR = "#c3c2b7"
_TICK_LABEL_COLOR = "#898781"
_INK_COLOR = "#0b0b0b"

_DODGE_SPREAD = 0.12  # total horizontal spread across models, centered on the tick
_SEED_DODGE_STEP = 0.02  # per-seed offset within a model's cluster

# One label position per effort level, not a shared offset: a plateau puts two
# effort levels on nearly the same frontier point, and identical offsets would
# overprint their labels.
_EFFORT_LABEL_OFFSETS = [(4, 6), (4, -12), (4, 14)]


def load_results(path: Path) -> list[EpisodeResult]:
    with path.open() as f:
        episodes = [EpisodeResult.model_validate_json(line) for line in f if line.strip()]
    if not episodes:
        raise ValueError(f"No episodes in {path} — nothing to analyze")
    return episodes


def agent_usage(result: EpisodeResult) -> Usage:
    """Sum Usage over turns where actor == "agent" (deployer-side spend only).

    Episode `totals` includes user-sim and judge calls too, which stand in for
    human effort and measurement overhead respectively — neither is spend the
    deployer incurs in production, so cost/latency-vs-utility must key on this
    narrower sum instead.
    """
    usage = Usage.zero()
    for turn in result.turns:
        if turn.actor == "agent":
            usage = usage + turn.usage
    return usage


def cost_latency_by_effort(results: list[EpisodeResult]) -> pd.DataFrame:
    """Deployer-side agent cost/latency and utility-per-unit-spend, by (task, model, effort).

    Ratios are ratio-of-means (mean_score / mean_agent_cost_usd), not means of
    per-episode ratios: with a handful of episodes per cell, one cheap
    low-scoring episode would dominate an averaged ratio.
    """
    rows = []
    for r in results:
        u = agent_usage(r)
        rows.append(
            {
                "task": r.task,
                "model": r.model,
                "effort": r.effort,
                "score": r.judge.score,
                "agent_cost_usd": u.cost_usd,
                "agent_latency_s": u.latency_s,
                "total_cost_usd": r.totals.cost_usd,
            }
        )
    df = pd.DataFrame(rows)
    df["effort"] = pd.Categorical(df["effort"], categories=EFFORT_ORDER, ordered=True)
    grouped = (
        df.groupby(["task", "model", "effort"], observed=True)
        .agg(
            n=("score", "count"),
            mean_score=("score", "mean"),
            mean_agent_cost_usd=("agent_cost_usd", "mean"),
            mean_agent_latency_s=("agent_latency_s", "mean"),
            mean_total_cost_usd=("total_cost_usd", "mean"),
        )
        .reset_index()
    )
    grouped["n"] = grouped["n"].astype(int)
    # Zero-cost/zero-latency cells (mock runs carry no real spend) -> NaN, not
    # a divide-by-zero error: there's no meaningful rate to report.
    grouped["utility_per_dollar"] = grouped["mean_score"] / grouped["mean_agent_cost_usd"].replace(
        0, float("nan")
    )
    grouped["utility_per_second"] = grouped["mean_score"] / grouped["mean_agent_latency_s"].replace(
        0, float("nan")
    )
    grouped = grouped.sort_values(["task", "model", "effort"]).reset_index(drop=True)
    return grouped[
        [
            "task",
            "model",
            "effort",
            "n",
            "mean_score",
            "mean_agent_cost_usd",
            "mean_agent_latency_s",
            "mean_total_cost_usd",
            "utility_per_dollar",
            "utility_per_second",
        ]
    ]


def utility_by_effort(results: list[EpisodeResult]) -> pd.DataFrame:
    rows = [
        {
            "task": r.task,
            "model": r.model,
            "effort": r.effort,
            "score": r.judge.score,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df["effort"] = pd.Categorical(df["effort"], categories=EFFORT_ORDER, ordered=True)
    grouped = (
        df.groupby(["task", "model", "effort"], observed=True)["score"]
        .agg(mean_score="mean", n="count")
        .reset_index()
    )
    grouped["n"] = grouped["n"].astype(int)
    grouped = grouped.sort_values(["task", "model", "effort"]).reset_index(drop=True)
    return grouped[["task", "model", "effort", "mean_score", "n"]]


def effort_manipulation(results: list[EpisodeResult]) -> pd.DataFrame:
    """Summarize simulated-user behavior per (task, effort), pooled across models/seeds.

    A rising or flat utility curve is ambiguous on its own: it could reflect a
    true null, or it could mean the effort levels never actually produced
    distinguishable user behavior (treatment failure). This does not judge
    separation — it just computes the numbers a reader needs to check it.
    """
    rows = []
    for r in results:
        sim_turns = [t for t in r.turns if t.actor == "user_sim"]
        user_turns = [t for t in sim_turns if t.message is not None]
        words = sum(len(t.message.content.split()) for t in user_turns)
        rows.append(
            {
                "task": r.task,
                "effort": r.effort,
                "n_user_turns": len(user_turns),
                "words": words,
                "sim_output_tokens": sum(t.usage.output_tokens for t in sim_turns),
            }
        )
    df = pd.DataFrame(rows)
    df["effort"] = pd.Categorical(df["effort"], categories=EFFORT_ORDER, ordered=True)

    grouped = (
        df.groupby(["task", "effort"], observed=True)
        .agg(
            n_episodes=("n_user_turns", "count"),
            mean_user_turns=("n_user_turns", "mean"),
            total_words=("words", "sum"),
            total_user_messages=("n_user_turns", "sum"),
            mean_sim_output_tokens=("sim_output_tokens", "mean"),
        )
        .reset_index()
    )
    # Zero user messages in a cell yields NaN, not a divide-by-zero error:
    # there is nothing to average, and that's a fact worth surfacing, not hiding.
    grouped["mean_words_per_user_message"] = grouped["total_words"] / grouped[
        "total_user_messages"
    ].replace(0, float("nan"))
    grouped["n_episodes"] = grouped["n_episodes"].astype(int)
    grouped = grouped.sort_values(["task", "effort"]).reset_index(drop=True)
    return grouped[
        [
            "task",
            "effort",
            "n_episodes",
            "mean_user_turns",
            "mean_words_per_user_message",
            "mean_sim_output_tokens",
        ]
    ]


def _model_order(results: list[EpisodeResult]) -> list[str]:
    order: list[str] = []
    for r in results:
        if r.model not in order:
            order.append(r.model)
    return order


def plot_utility_vs_effort(results: list[EpisodeResult], out_path: Path) -> Path:
    models = _model_order(results)
    if len(models) > len(_MODEL_COLORS):
        raise ValueError(
            f"plot_utility_vs_effort supports at most {len(_MODEL_COLORS)} models "
            f"(got {len(models)}); reduce the model count in the config."
        )
    model_color = dict(zip(models, _MODEL_COLORS, strict=False))
    model_label_color = dict(zip(models, _MODEL_LABEL_COLORS, strict=False))
    model_marker = dict(zip(models, _MODEL_MARKERS, strict=False))
    n_models = len(models)

    df = utility_by_effort(results)
    seed_lookup: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for r in results:
        seed_lookup.setdefault((r.task, r.model, r.effort), []).append((r.seed, r.judge.score))

    tasks = sorted(df["task"].unique())
    run_name = results[0].run_name

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    x_positions = range(len(EFFORT_ORDER))
    # Offsets center the model cluster on each tick; a single model gets 0 offset.
    if n_models > 1:
        model_offsets = {
            model: (i - (n_models - 1) / 2) * (_DODGE_SPREAD / max(n_models - 1, 1))
            for i, model in enumerate(models)
        }
    else:
        model_offsets = {models[0]: 0.0} if models else {}

    for ax, task in zip(axes, tasks, strict=True):
        task_df = df[df["task"] == task]
        for model in models:
            m_df = task_df[task_df["model"] == model].sort_values("effort")
            if m_df.empty:
                continue
            color = model_color[model]
            offset = model_offsets[model]
            xs = [EFFORT_ORDER.index(e) + offset for e in m_df["effort"]]
            ys = list(m_df["mean_score"])

            for x, effort in zip(xs, m_df["effort"], strict=True):
                seeds = sorted(seed_lookup.get((task, model, effort), []))
                n_seeds = len(seeds)
                for rank, (_seed, score) in enumerate(seeds):
                    seed_offset = (rank - (n_seeds - 1) / 2) * _SEED_DODGE_STEP
                    # Faint individual replicates, not error bars: at N=2-5 seeds
                    # an error bar overstates precision the sample doesn't have.
                    ax.plot(
                        x + seed_offset,
                        score,
                        marker=model_marker[model],
                        markersize=4,
                        color=color,
                        alpha=0.35,
                        linestyle="none",
                    )

            ax.plot(
                xs,
                ys,
                marker=model_marker[model],
                markersize=7,
                linewidth=2,
                color=color,
                label=model,
            )
            ax.annotate(
                model,
                xy=(xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=model_label_color[model],
                fontsize=9,
                va="center",
            )

        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(EFFORT_ORDER)
        ax.set_xlim(-0.5, len(EFFORT_ORDER) - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("user effort", color=_TICK_LABEL_COLOR)
        ax.set_title(task, color=_INK_COLOR)

        ax.yaxis.grid(True, color=_GRID_COLOR)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_SPINE_COLOR)
        ax.tick_params(colors=_TICK_LABEL_COLOR)

    axes[0].set_ylabel("utility (judge met-fraction)", color=_TICK_LABEL_COLOR)

    # Three stacked bands with reserved room — suptitle, legend, axes — or the
    # default anchors superimpose the first two at the figure's top edge.
    fig.subplots_adjust(top=0.82)
    if n_models >= 2:
        # Explicit handles: only the mean lines carry labels, and without them
        # fig.legend pairs the label list with the first artists it finds (the
        # unlabeled seed dots), giving every entry the first model's swatch.
        handles_by_label: dict[str, object] = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                handles_by_label.setdefault(label, handle)
        fig.legend(
            handles=[handles_by_label[m] for m in models if m in handles_by_label],
            loc="upper center",
            ncol=n_models,
            frameon=False,
            bbox_to_anchor=(0.5, 0.93),
        )
    fig.suptitle(f"utility vs. effort — {run_name}", color=_INK_COLOR, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cost_latency_frontier(results: list[EpisodeResult], out_path: Path) -> Path:
    """Two-panel (cost, latency) x utility frontier, one row per task."""
    models = _model_order(results)
    if len(models) > len(_MODEL_COLORS):
        raise ValueError(
            f"plot_cost_latency_frontier supports at most {len(_MODEL_COLORS)} models "
            f"(got {len(models)}); reduce the model count in the config."
        )
    model_color = dict(zip(models, _MODEL_COLORS, strict=False))
    model_label_color = dict(zip(models, _MODEL_LABEL_COLORS, strict=False))
    model_marker = dict(zip(models, _MODEL_MARKERS, strict=False))
    n_models = len(models)

    df = cost_latency_by_effort(results)
    seed_lookup: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    for r in results:
        u = agent_usage(r)
        seed_lookup.setdefault((r.task, r.model, r.effort), []).append(
            (u.cost_usd, u.latency_s, r.judge.score)
        )

    tasks = sorted(df["task"].unique())
    run_name = results[0].run_name

    fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 5 * len(tasks)))
    if len(tasks) == 1:
        axes = axes.reshape(1, 2)

    panels = [
        (0, "mean_agent_cost_usd", "agent cost per episode ($)"),
        (1, "mean_agent_latency_s", "agent latency per episode (s)"),
    ]

    for row_i, task in enumerate(tasks):
        task_df = df[df["task"] == task]
        for panel_idx, x_col, x_label in panels:
            ax = axes[row_i][panel_idx]
            for model in models:
                m_df = task_df[task_df["model"] == model].sort_values("effort")
                if m_df.empty:
                    continue
                color = model_color[model]
                xs = list(m_df[x_col])
                ys = list(m_df["mean_score"])
                efforts = list(m_df["effort"])

                for effort in efforts:
                    for cost, latency, score in seed_lookup.get((task, model, effort), []):
                        x = cost if panel_idx == 0 else latency
                        ax.plot(
                            x,
                            score,
                            marker=model_marker[model],
                            markersize=4,
                            color=color,
                            alpha=0.35,
                            linestyle="none",
                        )

                ax.plot(
                    xs,
                    ys,
                    marker=model_marker[model],
                    markersize=7,
                    linewidth=2,
                    color=color,
                    label=model,
                )
                for x, y, effort in zip(xs, ys, efforts, strict=True):
                    ax.annotate(
                        str(effort),
                        xy=(x, y),
                        xytext=_EFFORT_LABEL_OFFSETS[EFFORT_ORDER.index(str(effort))],
                        textcoords="offset points",
                        fontsize=7,
                        color=_TICK_LABEL_COLOR,
                    )
                ax.annotate(
                    model,
                    xy=(xs[-1], ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=model_label_color[model],
                    fontsize=9,
                    va="center",
                )

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(x_label, color=_TICK_LABEL_COLOR)
            # Label the task once per panel pair: the pair reads as one row, and
            # a repeated title collides with the legend band on narrow figures.
            if panel_idx == 0:
                ax.set_ylabel("utility (judge met-fraction)", color=_TICK_LABEL_COLOR)
                ax.set_title(task, color=_INK_COLOR)

            ax.yaxis.grid(True, color=_GRID_COLOR)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(_SPINE_COLOR)
            ax.tick_params(colors=_TICK_LABEL_COLOR)

    # Three stacked bands with reserved room — suptitle, legend, axes — or the
    # default anchors superimpose the first two at the figure's top edge.
    fig.subplots_adjust(top=0.82)
    if n_models >= 2:
        handles_by_label: dict[str, object] = {}
        for row in axes:
            for ax in row:
                for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                    handles_by_label.setdefault(label, handle)
        fig.legend(
            handles=[handles_by_label[m] for m in models if m in handles_by_label],
            loc="upper center",
            ncol=n_models,
            frameon=False,
            bbox_to_anchor=(0.5, 0.93),
        )
    fig.suptitle(f"cost/latency vs. utility frontier — {run_name}", color=_INK_COLOR, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot utility vs. user effort from a results JSONL."
    )
    parser.add_argument("--results", required=True, type=Path, help="Path to a results JSONL")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output PNG path (default: <results parent>/<results stem>_utility_vs_effort.png)",
    )
    parser.add_argument(
        "--frontier-out",
        type=Path,
        help=(
            "Frontier PNG path (default: <results parent>/<results stem>_cost_latency_frontier.png)"
        ),
    )
    args = parser.parse_args(argv)

    out_path = args.out or (args.results.parent / f"{args.results.stem}_utility_vs_effort.png")
    frontier_out_path = args.frontier_out or (
        args.results.parent / f"{args.results.stem}_cost_latency_frontier.png"
    )

    episodes = load_results(args.results)
    df = utility_by_effort(episodes)
    for row in df.itertuples(index=False):
        print(f"  {row.task}/{row.model}/{row.effort}: mean_score={row.mean_score:.3f} n={row.n}")

    print("effort manipulation check (sim behavior per effort):")
    manip_df = effort_manipulation(episodes)
    for row in manip_df.itertuples(index=False):
        print(
            f"  {row.task}/{row.effort}: n_episodes={row.n_episodes} "
            f"mean_user_turns={row.mean_user_turns:.2f} "
            f"mean_words_per_user_message={row.mean_words_per_user_message:.1f} "
            f"mean_sim_output_tokens={row.mean_sim_output_tokens:.1f}"
        )

    print("agent cost/latency per cell (deployer-side spend; ratios NaN when spend is zero):")
    cl_df = cost_latency_by_effort(episodes)
    for row in cl_df.itertuples(index=False):
        print(
            f"  {row.task}/{row.model}/{row.effort}: n={row.n} "
            f"mean_score={row.mean_score:.3f} "
            f"agent_cost=${row.mean_agent_cost_usd:.4f} "
            f"agent_latency={row.mean_agent_latency_s:.1f}s "
            f"total_cost=${row.mean_total_cost_usd:.4f} "
            f"utility_per_dollar={row.utility_per_dollar:.1f} "
            f"utility_per_second={row.utility_per_second:.3f}"
        )

    plot_utility_vs_effort(episodes, out_path)
    print(f"wrote {out_path}")
    plot_cost_latency_frontier(episodes, frontier_out_path)
    print(f"wrote {frontier_out_path}")


if __name__ == "__main__":
    main()
