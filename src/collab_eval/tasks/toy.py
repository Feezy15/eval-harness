"""A trivial deterministic task so the episode loop has something to chew on.

Exists for smoke runs and CI — real tasks (trip planning, CSV cleaning)
implement the same four methods.
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

    def user_context(self, seed: int) -> str:
        scenario = _SCENARIOS[seed % len(_SCENARIOS)]
        return f"You're planning {scenario}. Answer follow-up questions if the assistant asks."

    def judge_context(self, seed: int) -> str:
        return (
            "Score how complete, concrete, and responsive-to-feedback the final "
            "plan is for the user's stated scenario."
        )

    def judge_criteria(self, seed: int) -> list[str]:
        return [
            "A concrete plan exists for the stated scenario.",
            "The plan is specific (names, times, quantities) rather than generic filler.",
            "Feedback the user gave earlier in the conversation is incorporated.",
        ]
