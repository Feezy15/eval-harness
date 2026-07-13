"""Pricing table: cost math must be exact and unknown pairs must fail loud.

A silent $0 for an untracked (provider, model) pair would defeat the <$30
spend guardrail this whole harness runs under — every real call must be
attributable to a rate we've actually vouched for.
"""

import pytest

from collab_eval.models.pricing import PRICING_VERSION, cost_usd


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        ("anthropic", "claude-haiku-4-5", 6.00),
        ("openai", "gpt-5.4-mini", 5.25),
        ("openai", "gpt-5.4-nano", 1.45),
    ],
    ids=["anthropic-haiku", "openai-gpt-5.4-mini", "openai-gpt-5.4-nano"],
)
def test_cost_matches_table(provider, model, expected):
    assert cost_usd(provider, model, 1_000_000, 1_000_000) == pytest.approx(expected)


def test_anthropic_haiku_dated_id_prices_the_same_as_the_bare_alias():
    assert cost_usd(
        "anthropic", "claude-haiku-4-5-20251001", 1_000_000, 1_000_000
    ) == pytest.approx(6.00)


def test_unknown_pair_raises_naming_the_pair_and_pricing_version():
    with pytest.raises(Exception, match="unknown-provider") as exc_info:
        cost_usd("unknown-provider", "unknown-model", 100, 100)
    message = str(exc_info.value)
    assert "unknown-model" in message
    assert PRICING_VERSION in message


def test_table_param_allows_injection_for_tests():
    fake_table = {("fake", "m"): (10.0, 20.0)}
    assert cost_usd("fake", "m", 1_000_000, 1_000_000, table=fake_table) == pytest.approx(30.0)
    # And the real table is untouched by the injected one.
    with pytest.raises(KeyError):
        cost_usd("fake", "m", 1_000_000, 1_000_000)
