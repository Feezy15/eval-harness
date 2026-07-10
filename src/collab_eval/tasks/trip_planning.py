"""
Trip planning

One seeded scenario feeds three views of the same trip:

- `initial_goal`   — what the agent sees: destination + rough intent only.
- `user_context`   — what the simulated user's private system prompt sees:
                      the full requirements a real traveler would actually have.
- `judge_context`  — what the judge scores against: the same ground truth,
                      plus scoring guidance.

The gap between `initial_goal` and `user_context` *is* the experiment: a
passive simulated user won't volunteer the hidden requirements, so the
agent's plan is scored (via `judge_context`) against requirements it may
never have been told. That gap, and how it narrows as effort increases, is
what the utility-vs-effort curve measures.
"""

from collab_eval.tasks.base import Task

# Seed-indexed like ToyTask._SCENARIOS (`seed % len`), so a seed larger than
# the table wraps deterministically instead of erroring. Each scenario is a
# dict, not a formatted string, so `initial_goal` can cherry-pick the public
# subset (destination + intent) while `user_context`/`judge_context` render
# everything — the fields that make a plan actually fit are what a passive
# user would never think to volunteer up front.
_SCENARIOS = [
    {
        "destination": "Lisbon",
        "intent": "a long weekend trip",
        "dates": "May 8-11",
        "budget": "$1,200 total for two people",
        "party": "two adults, one with a peanut allergy",
        "constraints": [
            "no red-eye or overnight flights — hard to sleep on planes",
            "at least one hotel with a pool",
            "avoid neighborhoods with a lot of nightlife noise",
        ],
    },
    {
        "destination": "Kyoto",
        "intent": "a first-time visit",
        "dates": "the first two weeks of November",
        "budget": "$3,500 total for one person",
        "party": "one adult, traveling solo",
        "constraints": [
            "vegetarian meals only",
            "wants to avoid the most crowded tourist sites during peak leaf season",
            "prefers walkable neighborhoods over needing taxis",
        ],
    },
    {
        "destination": "Denver",
        "intent": "a family ski trip",
        "dates": "the week between Christmas and New Year's",
        "budget": "$5,000 total for a family of four",
        "party": "two adults and two kids (ages 6 and 9)",
        "constraints": [
            "one of the kids is a beginner skier and needs lessons",
            "lodging must be ski-in/ski-out or have a free shuttle",
            "no flights connecting through Denver itself — direct only",
        ],
    },
    {
        "destination": "Mexico City",
        "intent": "a food-focused trip",
        "dates": "the second week of March",
        "budget": "$1,800 total for two people",
        "party": "two adults, one with limited mobility (uses a cane)",
        "constraints": [
            "lodging and restaurants must be step-free or have elevator access",
            "wants a mix of street food and sit-down restaurants",
            "no day trips requiring more than 2 hours of travel each way",
        ],
    },
]


def _render_requirements(scenario: dict) -> str:
    constraints = "\n".join(f"- {c}" for c in scenario["constraints"])
    return (
        f"Destination: {scenario['destination']}\n"
        f"Dates: {scenario['dates']}\n"
        f"Budget: {scenario['budget']}\n"
        f"Party: {scenario['party']}\n"
        f"Constraints:\n{constraints}"
    )


class TripPlanningTask(Task):
    name = "trip_planning"

    def agent_system_prompt(self) -> str:
        return (
            "You are a helpful travel-planning assistant working with a user to "
            "put together a trip itinerary. Work iteratively: propose a plan, "
            "then refine it based on feedback. Ask clarifying questions when the "
            "user's request is underspecified rather than guessing."
        )

    def initial_goal(self, seed: int) -> str:
        scenario = _SCENARIOS[seed % len(_SCENARIOS)]
        return (
            f"I want to plan {scenario['intent']} to {scenario['destination']}. "
            "Can you help me put together an itinerary?"
        )

    def user_context(self, seed: int) -> str:
        scenario = _SCENARIOS[seed % len(_SCENARIOS)]
        return (
            "These are your actual requirements for this trip — the assistant "
            "does not know them unless you tell it. Share whatever the "
            "conversation calls for, at whatever level of detail fits how "
            "involved a traveler you're playing:\n\n" + _render_requirements(scenario)
        )

    def judge_context(self, seed: int) -> str:
        scenario = _SCENARIOS[seed % len(_SCENARIOS)]
        return (
            "Ground truth for this trip, which the user may or may not have "
            "fully disclosed during the conversation:\n\n"
            f"{_render_requirements(scenario)}\n\n"
            "Score the final plan's coverage of ALL of the above — budget, "
            "dates, party composition, and every listed constraint — not just "
            "the subset the user happened to mention. A plan that looks polished "
            "but silently violates an undisclosed constraint (e.g. books a "
            "red-eye despite the no-red-eye requirement) should score low on "
            "requirement coverage even if the user never objected to it."
        )
