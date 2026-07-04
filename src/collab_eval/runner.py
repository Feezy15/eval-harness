"""Runner: orchestrates the (task x model x effort x seed) matrix.

Logs everything the analysis needs — transcript, per-call tokens/cost/latency,
judge score — as one JSONL line per episode (appended as each episode finishes,
so a crash mid-matrix loses nothing already run) plus a flat CSV summary
derived from the JSONL records.
"""

import argparse
import csv
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from collab_eval.config import Config, ModelConfig, load_config
from collab_eval.judge import JUDGE_REGISTRY, Judge
from collab_eval.models import MODEL_REGISTRY, AgentModel
from collab_eval.tasks import TASK_REGISTRY, Task
from collab_eval.types import EpisodeResult, Message, TurnRecord, Usage
from collab_eval.user_sim import UserSimulator


def _lookup[T](registry: Mapping[str, T], key: str, kind: str) -> T:
    """Registry lookup with an error that says what *is* available."""
    try:
        return registry[key]
    except KeyError:
        raise ValueError(f"Unknown {kind} {key!r}; available: {sorted(registry)}") from None


def _build_model(cfg: ModelConfig, seed: int) -> AgentModel:
    cls = _lookup(MODEL_REGISTRY, cfg.provider, "model provider")
    return cls(model=cfg.model, seed=seed, temperature=cfg.temperature)


def run_episode(
    task: Task,
    agent: AgentModel,
    user_sim: UserSimulator,
    judge: Judge,
    *,
    seed: int,
    max_turns: int,
    run_name: str = "adhoc",
    config_hash: str = "adhoc",
) -> EpisodeResult:
    """One episode: agent and simulated user alternate until the user stops
    or the turn cap is hit; then the judge scores the transcript."""
    started_at = datetime.now(UTC)
    # The Task authors the opening goal verbatim (seeded): turn 1 is part of the
    # controlled condition — identical across effort levels and models — so the
    # user-sim's behavior is the only treatment that varies.
    conversation = [
        Message(role="system", content=task.agent_system_prompt()),
        Message(role="user", content=task.initial_goal(seed)),
    ]
    turns: list[TurnRecord] = []

    for turn in range(max_turns):
        response = agent.next_turn(conversation)
        conversation.append(response.message)
        turns.append(
            TurnRecord(
                turn_index=len(turns), actor="agent", message=response.message, usage=response.usage
            )
        )
        if turn == max_turns - 1:
            break  # cap reached — don't spend a user-sim call the agent can't act on

        user_turn = user_sim.next_user_turn(task, conversation)
        # Recorded even when it's a stop signal: the stop decision is a real,
        # costed call — untracked calls would understate cost per episode.
        turns.append(
            TurnRecord(
                turn_index=len(turns),
                actor="user_sim",
                message=user_turn.message,
                usage=user_turn.usage,
            )
        )
        if user_turn.message is None:
            break  # the simulated user is satisfied
        conversation.append(user_turn.message)

    judge_score = judge.score(task, conversation)
    totals = sum((t.usage for t in turns), Usage.zero()) + judge_score.usage
    return EpisodeResult(
        run_name=run_name,
        episode_id=f"{task.name}__{agent.name}__{user_sim.effort}__s{seed}",
        task=task.name,
        model=agent.name,
        effort=user_sim.effort,
        seed=seed,
        transcript=conversation,
        turns=turns,
        totals=totals,
        judge=judge_score,
        started_at=started_at,
        config_hash=config_hash,
    )


def run_matrix(config: Config, output_dir: str | Path | None = None) -> list[EpisodeResult]:
    """Run every cell of the configured matrix; write JSONL (full records,
    streamed) and CSV (flat summary). `output_dir` overrides config for tests."""
    out_dir = Path(output_dir) if output_dir is not None else config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stamped into every record so a results file can always be traced back to
    # the exact experiment definition that produced it.
    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:12]

    judge_cls = _lookup(JUDGE_REGISTRY, config.judge.provider, "judge provider")
    judge = judge_cls(model=config.judge.model, rubric_version=config.judge.rubric_version)

    jsonl_path = out_dir / f"{config.run_name}.jsonl"
    results: list[EpisodeResult] = []
    with jsonl_path.open("w") as jsonl_file:
        for task_cfg in config.tasks:
            task = _lookup(TASK_REGISTRY, task_cfg.name, "task")()
            for model_cfg in config.models:
                for effort in config.user_sim.effort_levels:
                    for seed in config.seeds:
                        agent = _build_model(model_cfg, seed)
                        sim_model = _build_model(
                            ModelConfig(
                                provider=config.user_sim.provider, model=config.user_sim.model
                            ),
                            seed,
                        )
                        episode = run_episode(
                            task,
                            agent,
                            UserSimulator(model=sim_model, effort=effort),
                            judge,
                            seed=seed,
                            max_turns=config.max_turns,
                            run_name=config.run_name,
                            config_hash=config_hash,
                        )
                        jsonl_file.write(episode.model_dump_json() + "\n")
                        results.append(episode)

    _write_csv(out_dir / f"{config.run_name}.csv", results)
    return results


_CSV_COLUMNS = [
    "run_name",
    "episode_id",
    "task",
    "model",
    "effort",
    "seed",
    "n_agent_turns",
    "n_user_turns",
    "score",
    "rubric_version",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_s",
    "config_hash",
    "started_at",
]


def _episode_row(ep: EpisodeResult) -> dict[str, object]:
    return {
        "run_name": ep.run_name,
        "episode_id": ep.episode_id,
        "task": ep.task,
        "model": ep.model,
        "effort": ep.effort,
        "seed": ep.seed,
        "n_agent_turns": sum(1 for t in ep.turns if t.actor == "agent"),
        # A stop probe (message=None) is a costed call, not a conversational turn.
        "n_user_turns": sum(1 for t in ep.turns if t.actor == "user_sim" and t.message is not None),
        "score": ep.judge.score,
        "rubric_version": ep.judge.rubric_version,
        "input_tokens": ep.totals.input_tokens,
        "output_tokens": ep.totals.output_tokens,
        "cost_usd": ep.totals.cost_usd,
        "latency_s": ep.totals.latency_s,
        "config_hash": ep.config_hash,
        "started_at": ep.started_at.isoformat(),
    }


def _write_csv(path: Path, results: list[EpisodeResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_episode_row(ep) for ep in results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the collaborative-effort evaluation matrix.")
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to a YAML experiment config"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = run_matrix(config)

    out_dir = config.output_dir
    print(f"{len(results)} episodes -> {out_dir / config.run_name}.jsonl / .csv")
    for ep in results:
        print(
            f"  {ep.episode_id}: score={ep.judge.score:.3f} "
            f"cost=${ep.totals.cost_usd:.6f} latency={ep.totals.latency_s:.2f}s"
        )


if __name__ == "__main__":
    main()
