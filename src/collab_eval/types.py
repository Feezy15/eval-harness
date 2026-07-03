"""Shared data model for the harness.

Kept in one module so every component (tasks, models, user-sim, judge, runner)
speaks the same types — the interfaces stay small and swappable.
"""

from typing import Literal

# The independent variable of the whole study: how involved the simulated user is.
# A Literal (not an Enum) so it reads/writes as a plain string in YAML configs and JSONL logs.
EffortLevel = Literal["passive", "moderate", "active_steering"]
