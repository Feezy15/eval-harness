"""TripPlanningTask: the first real (non-toy) task.

One seeded scenario table feeds three views — `initial_goal` (what the agent
sees: deliberately underspecified), `user_context` (what the simulated user's
private system prompt sees: full requirements), `judge_context` (what the
judge scores against: ground truth). Pins seed-stability, seed-indexing
(`seed % len`, matching `ToyTask._SCENARIOS`), and that the three views agree
on which scenario they're describing.
"""

import pytest
from collab_eval.tasks.trip_planning import _SCENARIOS, TripPlanningTask

from collab_eval.tasks import TASK_REGISTRY

N_SCENARIOS = len(_SCENARIOS)


def test_registered_under_trip_planning():
    assert TASK_REGISTRY["trip_planning"] is TripPlanningTask
    assert TripPlanningTask.name == "trip_planning"


@pytest.mark.parametrize("seed", range(N_SCENARIOS))
def test_seed_is_stable_across_repeated_calls(seed):
    task = TripPlanningTask()
    assert task.initial_goal(seed) == task.initial_goal(seed)
    assert task.user_context(seed) == task.user_context(seed)
    assert task.judge_context(seed) == task.judge_context(seed)


def test_different_seeds_yield_different_scenarios():
    # All table-sized seeds are distinct scenarios; a collision would mean two
    # cells of the (task x seed) matrix silently measure the same condition.
    task = TripPlanningTask()
    seeds = range(N_SCENARIOS)
    assert len({task.initial_goal(s) for s in seeds}) == N_SCENARIOS
    assert len({task.user_context(s) for s in seeds}) == N_SCENARIOS
    assert len({task.judge_context(s) for s in seeds}) == N_SCENARIOS


def test_seed_indexes_modulo_table_size():
    # Seed-indexed deterministically, matching ToyTask._SCENARIOS' `seed % len`
    # convention — a seed larger than the table must wrap, not error or repeat
    # a different mapping.
    task = TripPlanningTask()
    assert task.initial_goal(0) == task.initial_goal(N_SCENARIOS)
    assert task.user_context(0) == task.user_context(N_SCENARIOS)
    assert task.judge_context(0) == task.judge_context(N_SCENARIOS)


@pytest.mark.parametrize("seed", range(N_SCENARIOS))
def test_destination_is_consistent_across_all_three_views(seed):
    # The same scenario must feed all three views: a destination mentioned in
    # the underspecified goal has to be the same one the user's hidden
    # requirements and the judge's ground truth are about.
    task = TripPlanningTask()
    destination = _SCENARIOS[seed % N_SCENARIOS]["destination"]
    assert destination in task.initial_goal(seed)
    assert destination in task.user_context(seed)
    assert destination in task.judge_context(seed)
