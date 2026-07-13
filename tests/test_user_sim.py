"""Stop-sentinel semantics: the sim stops the episode only when it *signals*
completion, not whenever the sentinel string appears somewhere in its reply.

A substring check would end the episode on a mere mention ("I'll say <<DONE>>
once the plan is final") — a false stop that truncates the episode and biases
the utility measurement downward.
"""

import pytest

from collab_eval.types import Message
from collab_eval.user_sim import STOP_SENTINEL, UserSimulator
from conftest import ScriptedModel


def _sim_turn(reply: str, user_context: str = ""):
    sim = UserSimulator(model=ScriptedModel(reply), effort="passive", user_context=user_context)
    conversation = [
        Message(role="system", content="framing"),
        Message(role="user", content="goal"),
        Message(role="assistant", content="a proposal"),
    ]
    return sim.next_user_turn(conversation)


@pytest.mark.parametrize(
    "reply",
    [
        STOP_SENTINEL,
        f"  {STOP_SENTINEL}  ",
        f"Looks good, thanks. {STOP_SENTINEL}",
        f"{STOP_SENTINEL} Thanks for the help!",
        # Models habitually wrap a final token in closing punctuation, quotes,
        # or markdown emphasis — still a stop signal, not conversation.
        f"Sounds great, {STOP_SENTINEL}!",
        f'"{STOP_SENTINEL}"',
        f"**{STOP_SENTINEL}**",
    ],
)
def test_sentinel_as_signal_stops(reply):
    assert _sim_turn(reply).message is None


@pytest.mark.parametrize(
    "reply",
    [
        f"I'll reply with {STOP_SENTINEL} once the plan covers the budget.",
        f"Does {STOP_SENTINEL} mean we're finished? Anyway, please add a museum day.",
    ],
)
def test_sentinel_mention_mid_reply_does_not_stop(reply):
    turn = _sim_turn(reply)
    assert turn.message is not None
    assert turn.message.role == "user"


def test_user_context_appears_in_sim_system_prompt():
    # user_context is the hidden-requirements channel the sim's private system
    # prompt must carry — the whole point of separating it from the agent-visible
    # goal is that the sim (and only the sim) sees it.
    backend = ScriptedModel()
    canary = "CANARY_USER_CONTEXT_abc123"
    sim = UserSimulator(model=backend, effort="passive", user_context=canary)
    conversation = [
        Message(role="system", content="framing"),
        Message(role="user", content="goal"),
    ]
    sim.next_user_turn(conversation)
    (call,) = backend.calls
    system_msg = next(m for m in call if m.role == "system")
    assert canary in system_msg.content
