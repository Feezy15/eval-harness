"""Runner: orchestrates the (task x model x effort x seed) matrix.

Logs everything the analysis needs — transcript, per-call tokens/cost/latency,
judge score — as one JSONL line per episode (appended as each episode finishes,
so a crash mid-matrix loses nothing already run) plus a flat CSV summary
derived from the JSONL records.
"""

import argparse
import csv
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import Tracer, format_trace_id

from collab_eval import telemetry
from collab_eval.config import Config, ModelConfig, experiment_hash, load_config
from collab_eval.judge import Judge, build_judge
from collab_eval.models import MODEL_REGISTRY, AgentModel
from collab_eval.models.cache import CachedModel, ResponseCache
from collab_eval.models.pricing import PRICING_VERSION
from collab_eval.prompts import PROMPTS_FINGERPRINT, load_prompt
from collab_eval.tasks import TASK_REGISTRY, Task
from collab_eval.telemetry import build_tracer
from collab_eval.types import EpisodeFailure, EpisodeResult, Message, TurnRecord, Usage
from collab_eval.user_sim import UserSimulator

# The last in-loop agent turn is often a delta ("swapped Hotel X for Y"), not
# the plan — and that failure mode biases *against* high-effort episodes (more
# refinement rounds make the final turn more likely to be an edit rather than
# a restatement). So every episode gets one extra agent call, outside the
# max_turns cap, eliciting the complete artifact the judge actually scores.
FINAL_ARTIFACT_REQUEST = load_prompt("final_artifact_request")


def _lookup[T](registry: Mapping[str, T], key: str, kind: str) -> T:
    """Registry lookup with an error that says what *is* available."""
    try:
        return registry[key]
    except KeyError:
        raise ValueError(f"Unknown {kind} {key!r}; available: {sorted(registry)}") from None


def _build_model(cfg: ModelConfig, seed: int, cache: ResponseCache | None = None) -> AgentModel:
    cls = _lookup(MODEL_REGISTRY, cfg.provider, "model provider")
    model = cls(model=cfg.model, seed=seed, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if cache is None:
        return model
    # Key params come straight from config, not sniffed off the built model:
    # the config is the request identity we vouch for. Seed is included even
    # though remote APIs can't truly seed sampling — without it, replicate
    # cells with identical message prefixes would collide into one sample.
    return CachedModel(
        model,
        cache,
        key_params={
            "provider": cfg.provider,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "seed": seed,
        },
    )


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
    model_label: str | None = None,
    tracer: Tracer | None = None,
) -> EpisodeResult:
    """One episode: agent and simulated user alternate until the user stops
    or the turn cap is hit; then one final "write the whole plan out" agent
    call produces the artifact the judge scores (see `FINAL_ARTIFACT_REQUEST`).
    An episode therefore makes at most `max_turns + 1` agent calls, not
    `max_turns` — the consolidation call sits outside the cap so it can never
    shorten the collaboration itself.

    `model_label` is the agent's identity in the episode id and result rows —
    config entries that share a base model (e.g. a temperature ablation) pass
    distinct labels so their results stay distinguishable. Defaults to the
    agent's own name for standalone use.

    `tracer` defaults to a no-op (an episode called standalone, e.g. from a
    test, is untraced rather than silently becoming a root trace of its own
    tracer provider)."""
    if user_sim.user_context != task.user_context(seed):
        raise ValueError(
            f"user_sim.user_context does not match task.user_context(seed={seed}) "
            f"for task {task.name!r}: the episode would be judged against a "
            "different scenario than the sim was given"
        )
    # `is not None`, not truthiness: a falsy-but-present label must never silently
    # alias an episode to a different identity.
    label = model_label if model_label is not None else agent.name
    tracer = tracer if tracer is not None else trace.NoOpTracer()
    started_at = datetime.now(UTC)
    # The Task authors the opening goal verbatim (seeded): turn 1 is part of the
    # controlled condition — identical across effort levels and models — so the
    # user-sim's behavior is the only treatment that varies.
    conversation = [
        Message(role="system", content=task.agent_system_prompt()),
        Message(role="user", content=task.initial_goal(seed)),
    ]
    turns: list[TurnRecord] = []

    with tracer.start_as_current_span("episode") as episode_span:
        episode_id = f"{task.name}__{label}__{user_sim.effort}__s{seed}"
        episode_span.set_attribute(telemetry.ATTR_EPISODE_ID, episode_id)
        episode_span.set_attribute(telemetry.ATTR_TASK, task.name)
        episode_span.set_attribute(telemetry.ATTR_MODEL, label)
        episode_span.set_attribute(telemetry.ATTR_EFFORT, user_sim.effort)
        episode_span.set_attribute(telemetry.ATTR_SEED, seed)
        episode_span.set_attribute(telemetry.ATTR_CONFIG_HASH, config_hash)
        episode_span.set_attribute(telemetry.ATTR_PROMPTS_HASH, PROMPTS_FINGERPRINT)

        for turn in range(max_turns):
            turn_index = len(turns)
            with tracer.start_as_current_span("agent.turn") as agent_span:
                response = agent.next_turn(conversation)
                # Guarded: stub AgentModels used in tests predate `.model` and
                # don't define it, and there is nothing to attribute when the
                # span is discarded untraced anyway.
                if agent_span.is_recording():
                    agent_span.set_attribute(telemetry.ATTR_ACTOR, "agent")
                    agent_span.set_attribute(telemetry.ATTR_TURN_INDEX, turn_index)
                    agent_span.set_attribute(telemetry.ATTR_GENAI_MODEL, agent.model)
                    agent_span.set_attribute(telemetry.ATTR_GENAI_TEMPERATURE, agent.temperature)
                    agent_span.set_attribute(
                        telemetry.ATTR_GENAI_INPUT_TOKENS, response.usage.input_tokens
                    )
                    agent_span.set_attribute(
                        telemetry.ATTR_GENAI_OUTPUT_TOKENS, response.usage.output_tokens
                    )
                    agent_span.set_attribute(telemetry.ATTR_COST_USD, response.usage.cost_usd)
                    agent_span.set_attribute(telemetry.ATTR_LATENCY_S, response.usage.latency_s)
            conversation.append(response.message)
            turns.append(
                TurnRecord(
                    turn_index=turn_index,
                    actor="agent",
                    message=response.message,
                    usage=response.usage,
                )
            )
            if turn == max_turns - 1:
                break  # cap reached — don't spend a user-sim call the agent can't act on

            turn_index = len(turns)
            with tracer.start_as_current_span("user_sim.turn") as sim_span:
                user_turn = user_sim.next_user_turn(conversation)
                stopped = user_turn.message is None
                if sim_span.is_recording():
                    sim_span.set_attribute(telemetry.ATTR_ACTOR, "user_sim")
                    sim_span.set_attribute(telemetry.ATTR_TURN_INDEX, turn_index)
                    sim_span.set_attribute(telemetry.ATTR_GENAI_MODEL, user_sim.model.model)
                    sim_span.set_attribute(
                        telemetry.ATTR_GENAI_TEMPERATURE, user_sim.model.temperature
                    )
                    sim_span.set_attribute(
                        telemetry.ATTR_GENAI_INPUT_TOKENS, user_turn.usage.input_tokens
                    )
                    sim_span.set_attribute(
                        telemetry.ATTR_GENAI_OUTPUT_TOKENS, user_turn.usage.output_tokens
                    )
                    sim_span.set_attribute(telemetry.ATTR_COST_USD, user_turn.usage.cost_usd)
                    sim_span.set_attribute(telemetry.ATTR_LATENCY_S, user_turn.usage.latency_s)
                    # The stop probe is a real, costed call like any other — this
                    # flag is what lets a trace viewer show *where* the sim ended
                    # the episode, rather than that alone distinguishing the span.
                    sim_span.set_attribute(telemetry.ATTR_STOPPED, stopped)
            # Recorded even when it's a stop signal: the stop decision is a real,
            # costed call — untracked calls would understate cost per episode.
            turns.append(
                TurnRecord(
                    turn_index=turn_index,
                    actor="user_sim",
                    message=user_turn.message,
                    usage=user_turn.usage,
                )
            )
            if user_turn.message is None:
                break  # the simulated user is satisfied
            conversation.append(user_turn.message)

        # Consolidation turn: runs unconditionally, whether the loop ended via
        # a stop signal or the cap, so `transcript[-1]` is the full artifact
        # by construction — the contract the judge relies on — rather than
        # whatever the last in-loop message happened to be.
        conversation.append(Message(role="user", content=FINAL_ARTIFACT_REQUEST))
        turn_index = len(turns)
        with tracer.start_as_current_span("agent.turn") as consolidation_span:
            response = agent.next_turn(conversation)
            if consolidation_span.is_recording():
                consolidation_span.set_attribute(telemetry.ATTR_ACTOR, "agent")
                consolidation_span.set_attribute(telemetry.ATTR_TURN_INDEX, turn_index)
                consolidation_span.set_attribute(telemetry.ATTR_GENAI_MODEL, agent.model)
                consolidation_span.set_attribute(
                    telemetry.ATTR_GENAI_TEMPERATURE, agent.temperature
                )
                consolidation_span.set_attribute(
                    telemetry.ATTR_GENAI_INPUT_TOKENS, response.usage.input_tokens
                )
                consolidation_span.set_attribute(
                    telemetry.ATTR_GENAI_OUTPUT_TOKENS, response.usage.output_tokens
                )
                consolidation_span.set_attribute(telemetry.ATTR_COST_USD, response.usage.cost_usd)
                consolidation_span.set_attribute(telemetry.ATTR_LATENCY_S, response.usage.latency_s)
        conversation.append(response.message)
        turns.append(
            TurnRecord(
                turn_index=turn_index,
                actor="agent",
                message=response.message,
                usage=response.usage,
            )
        )

        with tracer.start_as_current_span("judge.score") as judge_span:
            judge_score = judge.score(task, conversation, seed)
            if judge_span.is_recording():
                judge_span.set_attribute(telemetry.ATTR_ACTOR, "judge")
                judge_span.set_attribute(telemetry.ATTR_GENAI_MODEL, judge.model)
                judge_span.set_attribute(
                    telemetry.ATTR_GENAI_INPUT_TOKENS, judge_score.usage.input_tokens
                )
                judge_span.set_attribute(
                    telemetry.ATTR_GENAI_OUTPUT_TOKENS, judge_score.usage.output_tokens
                )
                judge_span.set_attribute(telemetry.ATTR_COST_USD, judge_score.usage.cost_usd)
                judge_span.set_attribute(telemetry.ATTR_LATENCY_S, judge_score.usage.latency_s)
                judge_span.set_attribute(telemetry.ATTR_SCORE, judge_score.score)
                judge_span.set_attribute(telemetry.ATTR_RUBRIC_VERSION, judge_score.rubric_version)

        totals = sum((t.usage for t in turns), Usage.zero()) + judge_score.usage
        if episode_span.is_recording():
            episode_span.set_attribute(telemetry.ATTR_SCORE, judge_score.score)
            episode_span.set_attribute(telemetry.ATTR_TOTAL_INPUT_TOKENS, totals.input_tokens)
            episode_span.set_attribute(telemetry.ATTR_TOTAL_OUTPUT_TOKENS, totals.output_tokens)
            episode_span.set_attribute(telemetry.ATTR_TOTAL_COST_USD, totals.cost_usd)
            episode_span.set_attribute(telemetry.ATTR_TOTAL_LATENCY_S, totals.latency_s)

        span_context = episode_span.get_span_context()
        # An invalid context (the no-op tracer's case) has no trace to link to.
        trace_id = format_trace_id(span_context.trace_id) if span_context.is_valid else None

    return EpisodeResult(
        run_name=run_name,
        episode_id=episode_id,
        task=task.name,
        model=label,
        temperature=agent.temperature,
        effort=user_sim.effort,
        seed=seed,
        user_context=user_sim.user_context,
        transcript=conversation,
        turns=turns,
        totals=totals,
        judge=judge_score,
        started_at=started_at,
        config_hash=config_hash,
        # Measured from the prompt files (at import, not per episode), never
        # passed in: the caller can't claim an instrument identity other than
        # the one that ran, and a mid-run file edit can't either.
        prompts_hash=PROMPTS_FINGERPRINT,
        # Same principle as prompts_hash: the pricing snapshot is a property of
        # the code that ran, so it's stamped here, not passed in by the caller.
        pricing_version=PRICING_VERSION,
        trace_id=trace_id,
    )


def _load_resume_state(jsonl_path: Path, config_hash: str) -> tuple[list[EpisodeResult], set[str]]:
    """Validate an existing results file against the current experiment
    identity and return (episodes, completed episode ids) to resume from.

    Corrupt lines are a pydantic ValidationError (a ValueError subclass) we
    let propagate rather than skip: a half-written record is an instrument
    problem to triage, not silently drop. A results file from a different
    config or prompt text is a different experiment; appending to it would
    silently mix two conditions into one file, so that's also fatal.
    """
    episodes: list[EpisodeResult] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        record = EpisodeResult.model_validate_json(line)
        if record.config_hash != config_hash:
            raise ValueError(
                f"resume: {jsonl_path} contains results with config_hash "
                f"{record.config_hash!r}, but the current config hashes to "
                f"{config_hash!r} — these are different experiments"
            )
        if record.prompts_hash != PROMPTS_FINGERPRINT:
            raise ValueError(
                f"resume: {jsonl_path} contains results with prompts_hash "
                f"{record.prompts_hash!r}, but the current prompts hash to "
                f"{PROMPTS_FINGERPRINT!r} — the instrument text changed"
            )
        episodes.append(record)
    return episodes, {r.episode_id for r in episodes}


def _record_failure(path: Path, failure: EpisodeFailure) -> None:
    # Opened per failure, not held open for the whole matrix: failures are
    # rare, and this guarantees the line is on disk even if a later cell
    # crashes the process outright.
    with path.open("a") as f:
        f.write(failure.model_dump_json() + "\n")


def run_matrix(
    config: Config,
    output_dir: str | Path | None = None,
    tracer: Tracer | None = None,
    resume: bool = False,
) -> list[EpisodeResult]:
    """Run every cell of the configured matrix; write JSONL (full records,
    streamed) and CSV (flat summary). `output_dir` overrides config for tests;
    `tracer` likewise (tests inject an in-memory-exporting tracer instead of
    building one from `config.telemetry`).

    A failing cell is isolated (recorded to the `_failures.jsonl` sidecar,
    logged, skipped) rather than aborting the whole matrix — one bad episode
    shouldn't cost every other cell's spend. `resume=True` skips cells already
    present in an existing results file instead of re-running (and re-paying
    for) them; construction failures before a cell's `run_episode` call
    (unknown provider/task, cache/judge setup) are still fatal, since those
    indicate the run can't proceed at all, not that one cell is bad."""
    out_dir = Path(output_dir) if output_dir is not None else config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_hash = experiment_hash(config)

    cache = ResponseCache(config.cache.dir) if config.cache.enabled else None

    # Judge construction can fail (provider names resolve here at build time,
    # not at config-load time), so it happens before the tracer provider is
    # built: nothing may fail between building the provider and entering the
    # try/finally that guarantees its shutdown, or the provider would leak.
    judge = build_judge(config.judge, cache)

    # Only build (and later shut down) a provider we own; an injected tracer
    # belongs to its caller.
    provider = None
    if tracer is None:
        tracer, provider = build_tracer(config.telemetry)

    jsonl_path = out_dir / f"{config.run_name}.jsonl"
    failures_path = out_dir / f"{config.run_name}_failures.jsonl"
    # Every invocation starts from a clean slate for the sidecar: its absence
    # is the "nothing failed" signal a caller checks for, so a stale file from
    # a previous run must never survive into this run's report.
    failures_path.unlink(missing_ok=True)

    is_resuming = resume and jsonl_path.exists()
    if is_resuming:
        results, done_ids = _load_resume_state(jsonl_path, config_hash)
    else:
        results, done_ids = [], set()

    try:
        with tracer.start_as_current_span("run") as run_span:
            run_span.set_attribute(telemetry.ATTR_RUN_NAME, config.run_name)
            run_span.set_attribute(telemetry.ATTR_CONFIG_HASH, config_hash)
            run_span.set_attribute(telemetry.ATTR_PROMPTS_HASH, PROMPTS_FINGERPRINT)

            # "a" preserves already-validated lines byte-for-byte; a fresh or
            # non-resumed run truncates, matching the pre-resume behavior.
            with jsonl_path.open("a" if is_resuming else "w") as jsonl_file:
                for task_cfg in config.tasks:
                    task = _lookup(TASK_REGISTRY, task_cfg.name, "task")()
                    for model_cfg in config.models:
                        for effort in config.user_sim.effort_levels:
                            for seed in config.seeds:
                                agent = _build_model(model_cfg, seed, cache=cache)
                                label = (
                                    model_cfg.label if model_cfg.label is not None else agent.name
                                )
                                episode_id = f"{task.name}__{label}__{effort}__s{seed}"
                                if episode_id in done_ids:
                                    continue

                                sim_model = _build_model(
                                    ModelConfig(
                                        provider=config.user_sim.provider,
                                        model=config.user_sim.model,
                                        temperature=config.user_sim.temperature,
                                        max_tokens=config.user_sim.max_tokens,
                                    ),
                                    seed,
                                    cache=cache,
                                )
                                started_at = datetime.now(UTC)
                                try:
                                    episode = run_episode(
                                        task,
                                        agent,
                                        UserSimulator(
                                            model=sim_model,
                                            effort=effort,
                                            user_context=task.user_context(seed),
                                        ),
                                        judge,
                                        seed=seed,
                                        max_turns=config.max_turns,
                                        run_name=config.run_name,
                                        config_hash=config_hash,
                                        model_label=model_cfg.label,
                                        tracer=tracer,
                                    )
                                except (KeyboardInterrupt, SystemExit):
                                    raise
                                except Exception as exc:
                                    failure = EpisodeFailure(
                                        episode_id=episode_id,
                                        task=task.name,
                                        model=label,
                                        effort=effort,
                                        seed=seed,
                                        error_type=type(exc).__name__,
                                        error=str(exc),
                                        started_at=started_at,
                                    )
                                    _record_failure(failures_path, failure)
                                    print(
                                        f"episode {episode_id} failed: "
                                        f"{failure.error_type}: {failure.error}",
                                        file=sys.stderr,
                                    )
                                    continue
                                jsonl_file.write(episode.model_dump_json() + "\n")
                                results.append(episode)

            # Set once the matrix is actually done, so this reflects what ran
            # rather than a size computed in advance.
            run_span.set_attribute(telemetry.ATTR_N_EPISODES, len(results))
    finally:
        if provider is not None:
            provider.shutdown()

    _write_csv(out_dir / f"{config.run_name}.csv", results)
    return results


_CSV_COLUMNS = [
    "run_name",
    "episode_id",
    "task",
    "model",
    "temperature",
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
    "prompts_hash",
    "pricing_version",
    "started_at",
]


def _episode_row(ep: EpisodeResult) -> dict[str, object]:
    return {
        "run_name": ep.run_name,
        "episode_id": ep.episode_id,
        "task": ep.task,
        "model": ep.model,
        "temperature": ep.temperature,
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
        "prompts_hash": ep.prompts_hash,
        "pricing_version": ep.pricing_version,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes already present in an existing results file instead of rerunning them",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = run_matrix(config, resume=args.resume)

    out_dir = config.output_dir
    print(f"{len(results)} episodes -> {out_dir / config.run_name}.jsonl / .csv")
    for ep in results:
        print(
            f"  {ep.episode_id}: score={ep.judge.score:.3f} "
            f"cost=${ep.totals.cost_usd:.6f} latency={ep.totals.latency_s:.2f}s"
        )

    failures_path = out_dir / f"{config.run_name}_failures.jsonl"
    if failures_path.exists() and failures_path.stat().st_size > 0:
        failures = [
            EpisodeFailure.model_validate_json(line)
            for line in failures_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"{len(failures)} episodes failed:")
        for f in failures:
            print(f"  {f.episode_id}: {f.error_type}: {f.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
