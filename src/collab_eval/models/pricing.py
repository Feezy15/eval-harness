"""Per-token pricing for real LLM providers.

A versioned, hand-maintained table rather than a provider API lookup: pricing
pages change on their own schedule, and this project's <$30 spend cap depends
on cost being computed from a rate we've actually checked, not guessed.

Freshness is procedural, not detected: before any paid run, re-verify the
pages cited below and bump PRICING_VERSION if anything moved. Recorded costs
always mean "list price as of PRICING_VERSION" — stale is auditable, silent
drift is not.
"""

PRICING_VERSION = "2026-07-08"  # snapshot date of the source price pages below

# (provider, model) -> ($ per 1M input tokens, $ per 1M output tokens).
# Verified as of PRICING_VERSION against:
#   https://platform.claude.com/docs/en/docs/about-claude/pricing
#   https://developers.openai.com/api/docs/pricing
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),
    ("anthropic", "claude-haiku-4-5-20251001"): (1.00, 5.00),  # dated ID, same model
    ("openai", "gpt-5.4-mini"): (0.75, 4.50),
    ("openai", "gpt-5.4-nano"): (0.20, 1.25),
}


def cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    table: dict[tuple[str, str], tuple[float, float]] = PRICING,
) -> float:
    """Dollar cost of one call. Unknown (provider, model) fails loud: a silent
    $0 for an untracked pair would defeat the spend cap this table exists to
    enforce. `table` is overridable so tests can inject a fake rate without
    depending on (or polluting) the real pricing data."""
    try:
        in_per_mtok, out_per_mtok = table[(provider, model)]
    except KeyError:
        raise KeyError(
            f"no pricing entry for (provider={provider!r}, model={model!r}) as of "
            f"PRICING_VERSION {PRICING_VERSION!r} -- add it to pricing.PRICING before "
            "spending against this model"
        ) from None
    return (input_tokens * in_per_mtok + output_tokens * out_per_mtok) / 1e6
