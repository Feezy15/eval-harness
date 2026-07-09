"""Prompt files are the instrument definition: loadable, and fingerprinted so
results are traceable to the exact prompt text that produced them.

A version string alone can lie (edit the text, forget the bump); a content hash
cannot. Every episode record carries the fingerprint.
"""

import csv
from typing import get_args

from collab_eval.prompts import load_prompt, prompts_fingerprint
from collab_eval.types import EffortLevel

FINGERPRINT_HEX_LEN = 12


def test_all_instrument_prompts_load():
    names = [f"effort_{level}" for level in get_args(EffortLevel)] + ["rubric_v1"]
    for name in names:
        text = load_prompt(name)
        assert isinstance(text, str)
        assert text.strip(), f"prompt {name!r} is empty"


def test_unknown_prompt_fails_loud_listing_available():
    try:
        load_prompt("no_such_prompt")
    except ValueError as err:
        assert "available" in str(err)
    else:
        raise AssertionError("expected ValueError for unknown prompt name")


def test_fingerprint_shape_and_determinism():
    fp = prompts_fingerprint()
    assert len(fp) == FINGERPRINT_HEX_LEN
    int(fp, 16)  # valid hex
    assert fp == prompts_fingerprint()


def test_fingerprint_changes_when_content_changes(tmp_path):
    (tmp_path / "a.md").write_text("original text")
    before = prompts_fingerprint(tmp_path)
    (tmp_path / "a.md").write_text("edited text")
    assert prompts_fingerprint(tmp_path) != before


def test_fingerprint_changes_when_file_renamed(tmp_path):
    # The prompt's *name* is part of its identity: effort_passive.md renamed to
    # effort_moderate.md is a different instrument even with identical text.
    (tmp_path / "a.md").write_text("same text")
    before = prompts_fingerprint(tmp_path)
    (tmp_path / "a.md").rename(tmp_path / "b.md")
    assert prompts_fingerprint(tmp_path) != before


def test_effort_prompts_come_from_the_files():
    from collab_eval.user_sim import EFFORT_PROMPTS

    assert set(EFFORT_PROMPTS) == set(get_args(EffortLevel))
    for level, text in EFFORT_PROMPTS.items():
        assert text == load_prompt(f"effort_{level}")


def test_episode_records_carry_the_fingerprint(smoke_run):
    fp = prompts_fingerprint()
    assert all(ep.prompts_hash == fp for ep in smoke_run.results)

    with (smoke_run.output_dir / "smoke.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(row["prompts_hash"] == fp for row in rows)
