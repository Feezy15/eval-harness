"""Experiment configuration: pydantic schema + YAML loader.

All experiment knobs live here (PROJECT.md: "No hardcoded params in code").
Every model uses ``extra="forbid"`` so a typo'd key fails at load time instead of
silently running a different experiment — config bugs are the cheapest bugs to
catch and the most expensive to discover in a results plot.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from collab_eval.types import EffortLevel


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskConfig(_StrictModel):
    name: str  # resolved against TASK_REGISTRY at runner build time


class ModelConfig(_StrictModel):
    provider: str  # resolved against MODEL_REGISTRY at runner build time
    model: str
    temperature: float = 0.0


class UserSimConfig(_StrictModel):
    provider: str
    model: str
    effort_levels: list[EffortLevel] = Field(min_length=1)


class JudgeConfig(_StrictModel):
    provider: str
    model: str
    rubric_version: str


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


def load_config(path: str | Path) -> Config:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
