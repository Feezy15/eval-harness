"""Experiment configuration: pydantic schema + YAML loader.

All experiment knobs live here.
Every model uses ``extra="forbid"`` so a typo'd key fails at load time instead of
silently running a different experiment — config bugs are the cheapest bugs to
catch and the most expensive to discover in a results plot.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from collab_eval.types import EffortLevel


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reject_duplicates(kind: str, values: Sequence) -> None:
    """Duplicate matrix entries re-run identical cells and write result rows with
    identical identity, which aggregation then silently pools into one condition."""
    duplicated = [v for v, n in Counter(values).items() if n > 1]
    if duplicated:
        raise ValueError(f"duplicate {kind}: {duplicated} — each matrix cell must run exactly once")


class TaskConfig(_StrictModel):
    name: str  # resolved against TASK_REGISTRY at runner build time


class ModelConfig(_StrictModel):
    provider: str  # resolved against MODEL_REGISTRY at runner build time
    model: str
    temperature: float = Field(default=0.0, ge=0)  # providers enforce their own upper bounds
    # Unset by default (mock and OpenAI tolerate that); Anthropic requires it
    # and fails loud at construction if it's still None for that provider.
    max_tokens: int | None = Field(default=None, ge=1)
    # This entry's identity in episode ids, result rows, and plots. Entries that
    # share a provider+model (e.g. a temperature ablation) need explicit distinct
    # labels, or their results would be indistinguishable downstream.
    label: str | None = None

    @model_validator(mode="after")
    def _default_label(self) -> "ModelConfig":
        if self.label is not None and not self.label.strip():
            # A blank label would fall back to the raw model name downstream,
            # reopening the identity collision labels exist to prevent.
            raise ValueError("label must be non-empty; omit it to default to provider:model")
        if self.label is None:
            self.label = f"{self.provider}:{self.model}"
        return self


class UserSimConfig(_StrictModel):
    provider: str
    model: str
    # One pinned value per run, not a matrix dimension: the sim is measurement
    # apparatus, not treatment. It lives in config (not code) so it feeds the
    # config hash — two runs with different sim sampling are different experiments.
    temperature: float = Field(default=0.0, ge=0)
    # Threaded into the ModelConfig the runner builds for the sim's backing
    # model -- an Anthropic-backed sim needs this set, same as any agent model.
    max_tokens: int | None = Field(default=None, ge=1)
    effort_levels: list[EffortLevel] = Field(min_length=1)

    @field_validator("effort_levels")
    @classmethod
    def _unique_effort_levels(cls, v: list[EffortLevel]) -> list[EffortLevel]:
        _reject_duplicates("effort_levels", v)
        return v


class JudgeConfig(_StrictModel):
    provider: str
    model: str
    rubric_version: str
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)


class TelemetryConfig(_StrictModel):
    """Observability knob, not an experiment knob — see `experiment_hash`.

    Disabled by default so existing configs and CI stay exactly as they were
    before this field existed (no telemetry block => this default applies).
    """

    enabled: bool = False
    # "console" for local/manual inspection; "none" for keyless CI and tests
    # that want real trace ids stamped without anything printing. OTLP/Jaeger
    # export is a later, optional exporter — not implemented yet.
    exporter: Literal["console", "none"] = "console"


class CacheConfig(_StrictModel):
    """Disk cache for real LLM responses, keyed on full request identity.

    Enabled by default: real spend should be cached unless explicitly opted
    out (repo guardrail) — configs that want zero filesystem footprint (e.g.
    the mock-model smoke config) turn it off explicitly instead.
    """

    enabled: bool = True
    dir: str = "llm_cache"


class Config(_StrictModel):
    run_name: str
    output_dir: Path
    seeds: list[int] = Field(min_length=1)
    max_turns: int = Field(
        ge=1
    )  # hard cap per episode — bounds cost even if the user-sim never stops
    tasks: list[TaskConfig] = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    user_sim: UserSimConfig
    judge: JudgeConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, v: list[int]) -> list[int]:
        _reject_duplicates("seeds", v)
        return v

    @field_validator("tasks")
    @classmethod
    def _unique_task_names(cls, v: list[TaskConfig]) -> list[TaskConfig]:
        _reject_duplicates("task names", [t.name for t in v])
        return v

    @field_validator("models")
    @classmethod
    def _unique_model_labels(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        # Default labels omit sampling params, so two entries varying only in
        # temperature collide here by design: naming the variants ("gpt-4o-t0",
        # "gpt-4o-t1") must be a conscious act, not something we auto-mangle.
        _reject_duplicates("model labels (set a unique `label` per entry)", [m.label for m in v])
        return v


def load_config(path: str | Path) -> Config:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def experiment_hash(config: Config) -> str:
    """Sha256 (first 12 hex chars) of the config, minus the `telemetry` and
    `cache` blocks.

    Stamped into every result record (as `config_hash`) so it can always be
    traced back to the experiment definition that produced it.
    Both excluded blocks are *operational*, not experiment-defining: a traced
    or cached run and an untraced/uncached run of the same matrix are the same
    experiment and must hash the same, or turning on observability/caching
    would silently split identical results across two experiment identities.
    The cache key already covers the full request identity (including seed),
    so a hit can only ever replay what the identical experiment call produced
    earlier -- caching is a rerun-cost optimization, not a new experiment.
    Excluding both blocks also keeps this hash stable for configs written
    before they existed — they dump identical JSON either way.
    """
    return hashlib.sha256(
        config.model_dump_json(exclude={"telemetry", "cache"}).encode()
    ).hexdigest()[:12]
