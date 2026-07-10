"""Telemetry: the OTel span layer over the episode loop.

The span layer is *observational* instrumentation. These tests pin both halves
of that contract: it must describe a run exactly (a span tree mirroring
run -> episode -> LLM calls, with attributes matching the logged records), and
it must change nothing about the run (identical results with telemetry on,
off, or absent — and config identity that is blind to telemetry settings,
because tracing is operational, not experiment-defining).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_trace_id
from pydantic import ValidationError

from collab_eval.config import TelemetryConfig, experiment_hash, load_config
from collab_eval.judge import MockJudge
from collab_eval.models.mock import MockModel
from collab_eval.runner import run_episode, run_matrix
from collab_eval.tasks.toy import ToyTask
from collab_eval.telemetry import tracer_from_config
from collab_eval.types import EpisodeResult
from collab_eval.user_sim import UserSimulator
from conftest import EXPECTED_EPISODES, SMOKE_YAML, AlwaysStopsModel

# smoke.yaml: 1 task x 1 model x 3 effort levels x 2 seeds, max_turns=3, and the
# mock sim never stops -> per episode: 3 agent turns, 2 sim turns, 1 judge call.
TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _capturing_tracer():
    """A real SDK tracer wired to an in-memory exporter, so tests can read
    every finished span back — the global tracer provider is never touched
    (OTel allows setting it only once per process, which would couple tests)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _spans_by_name(spans, name):
    return [s for s in spans if s.name == name]


@dataclass(frozen=True)
class TracedRun:
    """One traced matrix run and the in-memory exporter that captured it."""

    results: list[EpisodeResult]
    exporter: InMemorySpanExporter
    output_dir: Path


@pytest.fixture(scope="module")
def traced_run(tmp_path_factory) -> TracedRun:
    """The smoke config run once with an explicit in-memory-exporting tracer,
    for every test that only *inspects* the resulting span tree.

    Contract: read-only, same as `smoke_run` — do not mutate `.results`, don't
    write into `.output_dir`, and don't call `.exporter.clear()`. A test that
    needs a different telemetry config (e.g. building the tracer from
    `config.telemetry` instead of injecting one, to check console/none
    exporter behavior) runs its own matrix rather than reusing this fixture.
    """
    output_dir = tmp_path_factory.mktemp("traced_run")
    cfg = load_config(SMOKE_YAML)
    tracer, exporter = _capturing_tracer()
    results = run_matrix(cfg, output_dir=output_dir, tracer=tracer)
    return TracedRun(results=results, exporter=exporter, output_dir=output_dir)


# --- config: telemetry is operational, never part of experiment identity ---


def test_telemetry_config_defaults_off_and_rejects_unknown_keys():
    assert TelemetryConfig().enabled is False
    with pytest.raises(ValidationError):
        TelemetryConfig(enabled=True, exporterr="console")
    with pytest.raises(ValidationError):
        TelemetryConfig(enabled=True, exporter="jaeger")


def test_runner_stamps_the_experiment_hash(smoke_run):
    # Holds on any untraced run of the unmodified smoke config, so the shared
    # fixture (rather than a fresh matrix) is enough to pin it.
    cfg = load_config(SMOKE_YAML)
    assert all(ep.config_hash == experiment_hash(cfg) for ep in smoke_run.results)


# --- disabled telemetry: no trace ids, no behavior change ---


def test_disabled_telemetry_stamps_no_trace_id_and_changes_nothing(tmp_path):
    cfg = load_config(SMOKE_YAML)
    untraced = run_matrix(cfg, output_dir=tmp_path / "off")

    assert all(ep.trace_id is None for ep in untraced)
    # The None survives the JSONL round trip.
    lines = (tmp_path / "off" / "smoke.jsonl").read_text().splitlines()
    assert all(EpisodeResult.model_validate_json(line).trace_id is None for line in lines)

    # Telemetry observes, never affects: a traced run produces byte-for-byte
    # the same transcripts, scores, and accounting.
    tracer, _ = _capturing_tracer()
    traced = run_matrix(cfg, output_dir=tmp_path / "on", tracer=tracer)
    for off, on in zip(untraced, traced, strict=True):
        assert off.transcript == on.transcript
        assert off.judge.score == on.judge.score
        assert off.totals == on.totals
        assert off.turns == on.turns


# --- the span tree mirrors the run structure ---


def test_span_tree_mirrors_run_episode_calls(traced_run):
    cfg = load_config(SMOKE_YAML)
    results = traced_run.results
    spans = traced_run.exporter.get_finished_spans()

    run_spans = _spans_by_name(spans, "run")
    episode_spans = _spans_by_name(spans, "episode")
    assert len(run_spans) == 1
    assert len(episode_spans) == EXPECTED_EPISODES
    assert len(_spans_by_name(spans, "agent.turn")) == EXPECTED_EPISODES * cfg.max_turns
    assert len(_spans_by_name(spans, "user_sim.turn")) == EXPECTED_EPISODES * (cfg.max_turns - 1)
    assert len(_spans_by_name(spans, "judge.score")) == EXPECTED_EPISODES

    # Parentage: episodes hang off the run; every call span hangs off an episode.
    run_ctx = run_spans[0].context
    assert run_spans[0].parent is None
    assert all(s.parent.span_id == run_ctx.span_id for s in episode_spans)
    episode_span_ids = {s.context.span_id for s in episode_spans}
    call_spans = [s for s in spans if s.name in {"agent.turn", "user_sim.turn", "judge.score"}]
    assert all(s.parent.span_id in episode_span_ids for s in call_spans)

    # The run span identifies the experiment and its size.
    attrs = run_spans[0].attributes
    assert attrs["collab_eval.run_name"] == cfg.run_name
    assert attrs["collab_eval.config_hash"] == experiment_hash(cfg)
    assert attrs["collab_eval.prompts_hash"] == results[0].prompts_hash
    assert attrs["collab_eval.n_episodes"] == EXPECTED_EPISODES


def test_episode_span_attributes_match_the_episode_record(traced_run):
    episode_spans = _spans_by_name(traced_run.exporter.get_finished_spans(), "episode")
    by_id = {s.attributes["collab_eval.episode_id"]: s for s in episode_spans}
    assert set(by_id) == {ep.episode_id for ep in traced_run.results}

    for ep in traced_run.results:
        attrs = by_id[ep.episode_id].attributes
        assert attrs["collab_eval.task"] == ep.task
        assert attrs["collab_eval.model"] == ep.model
        assert attrs["collab_eval.effort"] == ep.effort
        assert attrs["collab_eval.seed"] == ep.seed
        assert attrs["collab_eval.score"] == ep.judge.score
        assert attrs["collab_eval.total_input_tokens"] == ep.totals.input_tokens
        assert attrs["collab_eval.total_output_tokens"] == ep.totals.output_tokens
        assert attrs["collab_eval.total_cost_usd"] == ep.totals.cost_usd
        assert attrs["collab_eval.total_latency_s"] == ep.totals.latency_s
        assert attrs["collab_eval.config_hash"] == ep.config_hash
        assert attrs["collab_eval.prompts_hash"] == ep.prompts_hash


def test_call_span_attributes_match_turn_records(traced_run):
    cfg = load_config(SMOKE_YAML)
    spans = traced_run.exporter.get_finished_spans()

    episode_spans = _spans_by_name(spans, "episode")
    span_id_to_episode = {
        s.context.span_id: s.attributes["collab_eval.episode_id"] for s in episode_spans
    }
    ep = traced_run.results[0]

    turn_spans = sorted(
        (
            s
            for s in spans
            if s.name in {"agent.turn", "user_sim.turn"}
            and span_id_to_episode[s.parent.span_id] == ep.episode_id
        ),
        key=lambda s: s.attributes["collab_eval.turn_index"],
    )
    assert len(turn_spans) == len(ep.turns)
    for span, record in zip(turn_spans, ep.turns, strict=True):
        attrs = span.attributes
        assert span.name == f"{record.actor}.turn"
        assert attrs["collab_eval.actor"] == record.actor
        assert attrs["collab_eval.turn_index"] == record.turn_index
        # GenAI semconv names for the call itself; raw model name, not label.
        expected_model = "mock-agent" if record.actor == "agent" else "mock-user"
        expected_temp = 0.0 if record.actor == "agent" else cfg.user_sim.temperature
        assert attrs["gen_ai.request.model"] == expected_model
        assert attrs["gen_ai.request.temperature"] == expected_temp
        assert attrs["gen_ai.usage.input_tokens"] == record.usage.input_tokens
        assert attrs["gen_ai.usage.output_tokens"] == record.usage.output_tokens
        assert attrs["collab_eval.cost_usd"] == record.usage.cost_usd
        assert attrs["collab_eval.latency_s"] == record.usage.latency_s

    (judge_span,) = (
        s
        for s in _spans_by_name(spans, "judge.score")
        if span_id_to_episode[s.parent.span_id] == ep.episode_id
    )
    attrs = judge_span.attributes
    assert attrs["collab_eval.actor"] == "judge"
    assert attrs["gen_ai.request.model"] == cfg.judge.model
    assert attrs["gen_ai.usage.input_tokens"] == ep.judge.usage.input_tokens
    assert attrs["gen_ai.usage.output_tokens"] == ep.judge.usage.output_tokens
    assert attrs["collab_eval.score"] == ep.judge.score
    assert attrs["collab_eval.rubric_version"] == ep.judge.rubric_version


# --- trace ids link result records to their spans ---


def test_trace_id_links_jsonl_records_to_the_run_trace(traced_run):
    # One run = one trace: every episode record carries the run's trace id.
    (run_span,) = _spans_by_name(traced_run.exporter.get_finished_spans(), "run")
    run_trace_id = format_trace_id(run_span.context.trace_id)
    for ep in traced_run.results:
        assert ep.trace_id == run_trace_id
        assert TRACE_ID_RE.match(ep.trace_id)
    lines = (traced_run.output_dir / "smoke.jsonl").read_text().splitlines()
    assert all(EpisodeResult.model_validate_json(line).trace_id == run_trace_id for line in lines)


def test_enabled_config_without_console_stamps_trace_ids_quietly(tmp_path, capfd):
    # exporter "none": spans are recorded (so records get trace ids for later
    # OTLP export) but nothing is printed — usable in keyless CI.
    data = yaml.safe_load(SMOKE_YAML.read_text())
    data["telemetry"] = {"enabled": True, "exporter": "none"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    results = run_matrix(load_config(path), output_dir=tmp_path)

    assert all(ep.trace_id and TRACE_ID_RE.match(ep.trace_id) for ep in results)
    out, _ = capfd.readouterr()
    assert '"name": "episode"' not in out


def test_run_matrix_never_sets_the_global_tracer_provider(tmp_path):
    # Regression guard pinning already-correct behavior (so no red bar was
    # possible): OTel's global provider can be set once per process, and a
    # run_matrix that set it would couple every later run and test in the
    # process to whichever telemetry config ran first.
    data = yaml.safe_load(SMOKE_YAML.read_text())
    data["telemetry"] = {"enabled": True, "exporter": "none"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))

    before = trace.get_tracer_provider()
    run_matrix(load_config(path), output_dir=tmp_path)
    after = trace.get_tracer_provider()

    assert after is before
    # An SDK provider in the global slot means someone installed a real one;
    # the untouched default is the API's proxy placeholder.
    assert not isinstance(after, TracerProvider)


# --- tracer construction from config ---


def test_tracer_from_config_disabled_is_non_recording():
    tracer = tracer_from_config(TelemetryConfig())
    with tracer.start_as_current_span("probe") as span:
        assert not span.is_recording()


def test_tracer_from_config_console_exporter_prints_spans(capfd):
    tracer = tracer_from_config(TelemetryConfig(enabled=True, exporter="console"))
    with tracer.start_as_current_span("probe"):
        pass
    out, _ = capfd.readouterr()
    assert '"name": "probe"' in out


# --- standalone episodes and the stop-probe span ---


def test_stop_probe_is_a_flagged_span_and_standalone_episode_is_a_root_trace():
    task = ToyTask()
    agent = MockModel(model="mock-agent", seed=0)
    user_sim = UserSimulator(
        model=AlwaysStopsModel(), effort="passive", user_context=task.user_context(0)
    )
    judge = MockJudge(model="mock-judge", rubric_version="v0")
    tracer, exporter = _capturing_tracer()

    ep = run_episode(
        task=task, agent=agent, user_sim=user_sim, judge=judge, seed=0, max_turns=5, tracer=tracer
    )

    spans = exporter.get_finished_spans()
    (episode_span,) = _spans_by_name(spans, "episode")
    assert episode_span.parent is None  # no run span: the episode is its own trace
    assert ep.trace_id == format_trace_id(episode_span.context.trace_id)

    # The stop decision is a real, costed call: it gets a span like any other,
    # flagged so a trace viewer shows where the sim chose to end the episode.
    (stop_span,) = _spans_by_name(spans, "user_sim.turn")
    assert stop_span.attributes["collab_eval.stopped"] is True
    assert len(_spans_by_name(spans, "agent.turn")) == 1


def test_run_episode_without_tracer_stays_untraced():
    task = ToyTask()
    ep = run_episode(
        task=task,
        agent=MockModel(model="mock-agent", seed=0),
        user_sim=UserSimulator(
            model=AlwaysStopsModel(), effort="passive", user_context=task.user_context(0)
        ),
        judge=MockJudge(model="mock-judge", rubric_version="v0"),
        seed=0,
        max_turns=5,
    )
    assert ep.trace_id is None
