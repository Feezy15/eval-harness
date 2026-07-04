"""A trivial deterministic task so the episode loop has something to chew on.

Exists for smoke runs and CI — real tasks (trip planning, CSV cleaning)
implement the same three methods.
"""

from collab_eval.tasks.base import Task

# A few canned scenario flavors; the seed picks one, so different seeds produce
# genuinely different conversations without any randomness at run time.
_SCENARIOS = [
    "a small birthday gathering",
    "a two-day team offsite",
    "a rainy-Saturday activity list",
    "a beginner home-cooking menu",
]


class ToyTask(Task):
    name = "toy"

    def agent_system_prompt(self) -> str:
        return (
            "You are a helpful assistant collaborating with a user on a simple "
            "planning exercise. Work iteratively: propose, then refine based on feedback."
        )

    def initial_goal(self, seed: int) -> str:
        scenario = _SCENARIOS[seed % len(_SCENARIOS)]
        return f"Help me plan {scenario}. I haven't thought through the details yet."

    def judge_context(self) -> str:
        return (
            "Score how complete, concrete, and responsive-to-feedback the final "
            "plan is for the user's stated scenario."
        )