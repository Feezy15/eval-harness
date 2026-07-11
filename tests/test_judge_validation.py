"""judge_validation: golden-set fixture loading and judge-agreement checking.

Exercised over stub judges (no network): the loader's fail-loud contract,
per-criterion agreement/disagreement detection, and the CLI's exit-code
behavior. The real-model validation run is an operational gate before the
experiment matrix, not a test.
"""

import pytest
import yaml

from collab_eval.judge import LLMJudge, MockJudge
from collab_eval.judge_validation import load_golden_set, main, validate_judge
from collab_eval.models import MODEL_REGISTRY
from collab_eval.tasks.toy import ToyTask
from conftest import FakeProviderModel, ScriptedModel, canned_checklist, smoke_dict, write_yaml

# ToyTask.judge_criteria has 3 items — every fixture below is sized to that.


def _golden(artifacts: list[dict]) -> dict:
    return {"task": "toy", "rubric_version": "v1", "artifacts": artifacts}


def _artifact(**overrides) -> dict:
    base = {
        "name": "gold_0",
        "seed": 0,
        "kind": "gold",
        "expected": [True, True, True],
        "artifact": "a plan that satisfies everything",
    }
    base.update(overrides)
    return base


def _write_golden(tmp_path, data: dict, filename: str = "golden.yaml"):
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_golden_set_roundtrip(tmp_path):
    path = _write_golden(
        tmp_path,
        _golden(
            [
                _artifact(),
                _artifact(
                    name="flawed_0",
                    kind="flawed",
                    expected=[True, False, True],
                    note="criterion 2: number stated in the plan is wrong",
                ),
            ]
        ),
    )

    golden = load_golden_set(path, ToyTask())

    assert golden.task == "toy"
    assert golden.rubric_version == "v1"
    assert [a.name for a in golden.artifacts] == ["gold_0", "flawed_0"]
    assert golden.artifacts[1].expected == [True, False, True]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda g: g.update(bogus=1), "bogus"),
        (lambda g: g.update(task="trip_planning"), "task"),
        (lambda g: g["artifacts"][0].update(expected=[True, True]), "expected"),
        (lambda g: g["artifacts"].append(_artifact()), "duplicate"),
        (
            lambda g: g["artifacts"][0].update(kind="flawed", expected=[True, False, True]),
            "note",
        ),
    ],
    ids=["unknown_key", "task_mismatch", "wrong_expected_length", "duplicate_names", "no_note"],
)
def test_load_golden_set_fails_loud(tmp_path, mutate, match):
    data = _golden([_artifact()])
    mutate(data)

    with pytest.raises(ValueError, match=match):
        load_golden_set(_write_golden(tmp_path, data), ToyTask())


def test_validate_judge_detects_disagreements(tmp_path):
    task = ToyTask()
    # The stub judge always answers [True, True, False]: first artifact's
    # labels agree with it, second's don't at criterion 3.
    judge = LLMJudge(
        model=ScriptedModel(canned_checklist([True, True, False])), rubric_version="v1"
    )
    golden = load_golden_set(
        _write_golden(
            tmp_path,
            _golden(
                [
                    _artifact(
                        name="agrees",
                        kind="flawed",
                        expected=[True, True, False],
                        note="criterion 3: plan omits the requirement",
                    ),
                    _artifact(name="disagrees", expected=[True, True, True]),
                ]
            ),
        ),
        task,
    )

    report = validate_judge(golden, judge, task)

    assert not report.passed
    (d,) = report.disagreements
    assert d.artifact == "disagrees"
    assert d.criterion_index == 3
    assert d.expected is True and d.got is False
    assert d.reasoning
    assert "disagrees" in report.render()

    agreeing_only = load_golden_set(
        _write_golden(
            tmp_path,
            _golden(
                [
                    _artifact(
                        name="agrees",
                        kind="flawed",
                        expected=[True, True, False],
                        note="criterion 3: plan omits the requirement",
                    )
                ]
            ),
            filename="agreeing.yaml",
        ),
        task,
    )
    assert validate_judge(agreeing_only, judge, task).passed


def test_validate_judge_requires_per_criterion_verdicts(tmp_path):
    golden = load_golden_set(_write_golden(tmp_path, _golden([_artifact()])), ToyTask())
    judge = MockJudge(model="mock-judge", rubric_version="v1")

    with pytest.raises(ValueError, match="verdict"):
        validate_judge(golden, judge, ToyTask())


def test_validate_judge_rejects_rubric_version_mismatch(tmp_path):
    golden = load_golden_set(_write_golden(tmp_path, _golden([_artifact()])), ToyTask())
    judge = LLMJudge(model=ScriptedModel(canned_checklist([True] * 3)), rubric_version="v2")

    with pytest.raises(ValueError, match="rubric"):
        validate_judge(golden, judge, ToyTask())


def test_main_exit_codes(monkeypatch, tmp_path):
    monkeypatch.setitem(MODEL_REGISTRY, "fake", FakeProviderModel)
    cfg = smoke_dict()
    cfg["judge"] = {"provider": "fake", "model": "fake-model", "rubric_version": "v1"}
    cfg["cache"] = {"enabled": False}
    config_path = write_yaml(tmp_path, cfg)

    # FakeProviderModel judges every criterion met, so all-met labels pass.
    passing = _write_golden(tmp_path, _golden([_artifact()]), filename="passing.yaml")
    assert main(["--config", str(config_path), "--golden", str(passing)]) is None

    failing = _write_golden(
        tmp_path,
        _golden(
            [
                _artifact(
                    kind="flawed", expected=[True, False, True], note="criterion 2: wrong number"
                )
            ]
        ),
        filename="failing.yaml",
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--golden", str(failing)])
    assert excinfo.value.code == 1
