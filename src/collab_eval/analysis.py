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

from collab_eval.types import EffortLevel, EpisodeResult  # noqa: E402

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


def load_results(path: Path) -> list[EpisodeResult]:
    with path.open() as f:
        return [EpisodeResult.model_validate_json(line) for line in f if line.strip()]


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

    if n_models >= 2:
        fig.legend(models, loc="upper center", ncol=n_models, frameon=False)

    fig.suptitle(f"utility vs. effort — {run_name}", color=_INK_COLOR)
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
    args = parser.parse_args(argv)

    out_path = args.out or (args.results.parent / f"{args.results.stem}_utility_vs_effort.png")

    episodes = load_results(args.results)
    df = utility_by_effort(episodes)
    for row in df.itertuples(index=False):
        print(f"  {row.task}/{row.model}/{row.effort}: mean_score={row.mean_score:.3f} n={row.n}")

    plot_utility_vs_effort(episodes, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
