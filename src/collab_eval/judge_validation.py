"""Golden-set validation: the judge is a measuring instrument, and an
instrument gets calibrated against known-good readings before it's trusted to
score a real experiment. A small set of hand-labeled artifacts (with known
per-criterion boolean labels) is scored through the judge's real call path;
any disagreement fails the gate.

CLI: `python -m collab_eval.judge_validation --config <experiment.yaml> --golden <golden.yaml>`.
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from collab_eval.config import load_config
from collab_eval.judge import Judge, build_judge
from collab_eval.models.cache import ResponseCache
from collab_eval.tasks import TASK_REGISTRY, Task
from collab_eval.types import Message


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldenArtifact(_StrictModel):
    name: str
    seed: int
    kind: Literal["gold", "flawed", "degenerate", "adversarial"]
    # One boolean per `task.judge_criteria(seed)` entry, in checklist order —
    # the hand-labeled ground truth the judge's per-criterion verdicts are
    # compared against.
    expected: list[bool]
    # Required whenever any criterion is labeled unmet: an unexplained "False"
    # is unfalsifiable when a judge disagreement shows up later and someone
    # has to decide whether the judge or the fixture is wrong.
    note: str = ""
    artifact: str

    @model_validator(mode="after")
    def _note_explains_unmet(self) -> "GoldenArtifact":
        if any(not e for e in self.expected) and not self.note.strip():
            raise ValueError(
                f"artifact {self.name!r} has an unmet expected criterion but no note explaining why"
            )
        return self


class GoldenSet(_StrictModel):
    task: str
    rubric_version: str
    artifacts: list[GoldenArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_artifact_names(self) -> "GoldenSet":
        counts = Counter(a.name for a in self.artifacts)
        duplicates = [name for name, n in counts.items() if n > 1]
        if duplicates:
            raise ValueError(f"duplicate artifact names: {sorted(duplicates)}")
        return self


def load_golden_set(path: Path, task: Task) -> GoldenSet:
    """Load and validate a golden set against the task it claims to target.

    Fails loud on a task-name mismatch (wrong fixture file for this task) and
    on a per-artifact criterion-count mismatch (fixture written against a
    stale `judge_criteria`) — both are fixture bugs that must never silently
    validate the wrong thing.
    """
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    golden = GoldenSet.model_validate(raw)

    if golden.task != task.name:
        raise ValueError(
            f"golden set targets task {golden.task!r}, but was loaded against task {task.name!r}"
        )

    for artifact in golden.artifacts:
        n_criteria = len(task.judge_criteria(artifact.seed))
        if len(artifact.expected) != n_criteria:
            raise ValueError(
                f"artifact {artifact.name!r} has {len(artifact.expected)} expected verdicts; "
                f"task.judge_criteria(seed={artifact.seed}) has {n_criteria}"
            )

    return golden


@dataclass
class Disagreement:
    artifact: str
    kind: str
    criterion_index: int  # 1-based, matching the checklist numbering the judge sees
    criterion: str
    expected: bool
    got: bool
    reasoning: str  # the judge's stated reasoning for this criterion


@dataclass
class ArtifactAgreement:
    """Per-artifact roll-up: how many of its criteria the judge agreed on,
    and the expected vs. judged met-fraction (comparable to `JudgeScore.score`
    on a real episode)."""

    name: str
    kind: str
    n_criteria: int
    n_agree: int
    expected_fraction: float
    judged_fraction: float


@dataclass
class ValidationReport:
    disagreements: list[Disagreement]
    artifacts: list[ArtifactAgreement] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.disagreements

    def render(self) -> str:
        lines = ["Golden-set judge validation", ""]
        for a in self.artifacts:
            flag = "  [ADVERSARIAL]" if a.kind == "adversarial" else ""
            lines.append(
                f"- {a.name} ({a.kind}){flag}: expected {a.expected_fraction:.2f} met, "
                f"judged {a.judged_fraction:.2f} met "
                f"({a.n_agree}/{a.n_criteria} criteria agree)"
            )

        if self.disagreements:
            lines.append("")
            lines.append("Disagreements:")
            for d in self.disagreements:
                lines.append(
                    f"- {d.artifact} [{d.kind}] criterion {d.criterion_index}: {d.criterion!r} "
                    f"expected={d.expected} got={d.got} — judge reasoning: {d.reasoning}"
                )

        lines.append("")
        lines.append(
            "PASS: judge agrees with every golden label"
            if self.passed
            else f"FAIL: {len(self.disagreements)} disagreement(s) — see above"
        )
        return "\n".join(lines)


def validate_judge(golden: GoldenSet, judge: Judge, task: Task) -> ValidationReport:
    """Score every golden artifact through the judge's real call path
    (`Judge.score`, not some parsing shortcut) and compare its per-criterion
    verdicts to the hand-labeled `expected` list.
    """
    if judge.rubric_version != golden.rubric_version:
        raise ValueError(
            f"judge rubric_version {judge.rubric_version!r} does not match golden set's "
            f"rubric_version {golden.rubric_version!r}"
        )

    disagreements: list[Disagreement] = []
    artifact_rows: list[ArtifactAgreement] = []

    for a in golden.artifacts:
        score = judge.score(task, [Message(role="assistant", content=a.artifact)], a.seed)
        if score.criteria is None:
            raise ValueError(
                f"judge produced no per-criterion verdicts for artifact {a.name!r}; "
                "a judge without per-criterion verdicts can't be golden-set validated"
            )

        criteria_text = task.judge_criteria(a.seed)
        got = [c.met for c in score.criteria]
        n_agree = sum(1 for e, g in zip(a.expected, got, strict=True) if e == g)
        artifact_rows.append(
            ArtifactAgreement(
                name=a.name,
                kind=a.kind,
                n_criteria=len(a.expected),
                n_agree=n_agree,
                expected_fraction=sum(a.expected) / len(a.expected),
                judged_fraction=sum(got) / len(got),
            )
        )

        for i, (expected, verdict, criterion) in enumerate(
            zip(a.expected, score.criteria, criteria_text, strict=True), start=1
        ):
            if verdict.met != expected:
                disagreements.append(
                    Disagreement(
                        artifact=a.name,
                        kind=a.kind,
                        criterion_index=i,
                        criterion=criterion,
                        expected=expected,
                        got=verdict.met,
                        reasoning=verdict.reasoning,
                    )
                )

    return ValidationReport(disagreements=disagreements, artifacts=artifact_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate a judge against a hand-labeled golden set before a real run."
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML config")
    parser.add_argument("--golden", required=True, type=Path, help="Golden-set YAML fixture")
    args = parser.parse_args(argv)

    with args.golden.open() as f:
        raw = yaml.safe_load(f)
    task_name = raw.get("task") if isinstance(raw, dict) else None
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task {task_name!r}; available: {sorted(TASK_REGISTRY)}")
    task = TASK_REGISTRY[task_name]()

    golden = load_golden_set(args.golden, task)

    config = load_config(args.config)
    cache = ResponseCache(config.cache.dir) if config.cache.enabled else None
    judge = build_judge(config.judge, cache)

    report = validate_judge(golden, judge, task)
    print(report.render())
    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
