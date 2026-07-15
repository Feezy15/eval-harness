"""OTel span layer over the episode loop.

Purely observational: a run traced and a run untraced must produce identical
results (see `config.experiment_hash`, which excludes this module's config
block from experiment identity for the same reason). Spans are created at the
orchestration site in `runner.py` only — model/judge/task code is untouched —
and carry no message content (transcripts already live in the JSONL; the
GenAI semantic conventions treat content capture as opt-in).

We never call `trace.set_tracer_provider`: OTel's global provider can be set
exactly once per process, which would couple every test and every caller in
this process to whichever config ran first. Local providers, handed out per
call, keep runs independent.
"""

import sys

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Tracer

from collab_eval.config import TelemetryConfig

# GenAI semantic conventions (still incubating upstream) for the call itself;
# collab_eval.* for everything the semconv doesn't cover. Pinned as constants
# so a semconv rename is a one-place fix rather than a grep across runner.py.
ATTR_RUN_NAME = "collab_eval.run_name"
ATTR_CONFIG_HASH = "collab_eval.config_hash"
ATTR_PROMPTS_HASH = "collab_eval.prompts_hash"
ATTR_N_EPISODES = "collab_eval.n_episodes"
ATTR_N_EPISODES_RESUMED = "collab_eval.n_episodes_resumed"

ATTR_EPISODE_ID = "collab_eval.episode_id"
ATTR_TASK = "collab_eval.task"
ATTR_MODEL = "collab_eval.model"
ATTR_EFFORT = "collab_eval.effort"
ATTR_SEED = "collab_eval.seed"
ATTR_SCORE = "collab_eval.score"
ATTR_TOTAL_INPUT_TOKENS = "collab_eval.total_input_tokens"
ATTR_TOTAL_OUTPUT_TOKENS = "collab_eval.total_output_tokens"
ATTR_TOTAL_COST_USD = "collab_eval.total_cost_usd"
ATTR_TOTAL_LATENCY_S = "collab_eval.total_latency_s"

ATTR_ACTOR = "collab_eval.actor"
ATTR_TURN_INDEX = "collab_eval.turn_index"
ATTR_COST_USD = "collab_eval.cost_usd"
ATTR_LATENCY_S = "collab_eval.latency_s"
ATTR_STOPPED = "collab_eval.stopped"
ATTR_RUBRIC_VERSION = "collab_eval.rubric_version"

ATTR_GENAI_MODEL = "gen_ai.request.model"
ATTR_GENAI_TEMPERATURE = "gen_ai.request.temperature"
ATTR_GENAI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GENAI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


def build_tracer(cfg: TelemetryConfig) -> tuple[Tracer, TracerProvider | None]:
    """Tracer plus the provider it owns."""
    if not cfg.enabled:
        return trace.NoOpTracer(), None

    provider = TracerProvider()
    if cfg.exporter == "console":
        # Synchronous export (not batched): span order in the printed output
        # matches call order, which is what makes gate 4's eyeball check work.
        # `out=sys.stdout` is passed explicitly rather than left to the
        # exporter's own default: that default binds at class-definition time
        # (this module's import), so under a test runner that swaps sys.stdout
        # per test, the baked-in default goes stale and writes become
        # invisible to capfd/capsys. Looking it up here binds fresh, per call.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stdout)))
    # exporter == "none": no processor attached. Spans still get real trace
    # ids (id generation doesn't depend on processors) so records stay
    # traceable for later OTLP export, but nothing is exported now — this is
    # the keyless-CI path.
    return provider.get_tracer("collab_eval"), provider


def tracer_from_config(cfg: TelemetryConfig) -> Tracer:
    """The provider built here is dropped, never shut down — harmless for the
    console path (synchronous export buffers nothing), but callers needing
    flush-on-shutdown must use `build_tracer` and own the provider."""
    tracer, _ = build_tracer(cfg)
    return tracer
