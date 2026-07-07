"""Stop-sentinel semantics: the sim stops the episode only when it *signals*
completion, not whenever the sentinel string appears somewhere in its reply.

A substring check would end the episode on a mere mention ("I'll say <<DONE>>
once the plan is final") — a false stop that truncates the episode and biases
the utility measurement downward.
"""

from collections.abc import Sequence

import pytest

from collab_eval.models.base import AgentModel
from collab_eval.tasks.toy import ToyTask
from collab_eval.types import Message, ModelResponse, Usage
from collab_eval.user_sim import STOP_SENTINEL, UserSimulator


class _FixedReplyModel(AgentModel):
    """Sim backend that replies with exactly the text under test."""

    name = "stub:fixed-reply"
    temperature = 0.0

    def __init__(self, reply: str):
        self._reply = reply

    def next_turn(self, conversation: Sequence[Message]) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=self._reply),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=1e-6, latency_s=0.01),
        )


def _sim_turn(reply: str):
    sim = UserSimulator(model=_FixedReplyModel(reply), effort="passive")
    conversation = [
        Message(role="system", content="framing"),
        Message(role="user", content="goal"),
        Message(role="assistant", content="a proposal"),
    ]
    return sim.next_user_turn(ToyTask(), conversation)


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
