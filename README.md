# Collaborative-Effort Evaluation Harness

*Python 3.12 · MIT License*

A small, reproducible harness that measures **how an LLM agent's usefulness scales with user involvement**
across multi-turn tasks. It reproduces the core finding of *Completion ≠ Collaboration* (Wu et al., 2025) and
extends it with a **cost/latency-vs-utility** analysis the paper doesn't cover.

## Motivation

Most agent evaluation is one-shot and single-turn, which misses that real goals are underspecified and evolve
through back-and-forth. This project targets the under-tooled space of **multi-turn, collaborative** evaluation:
*how much more useful does an agent become as the user gets more involved — and what does that cost?*

## Key idea: collaborative effort scaling

Instead of scoring an agent only on its final output, we vary **how involved a (simulated) user is** across turns
and measure how agent utility changes. An "agent" LLM works an iterative task; a second LLM plays the user at
several effort levels (`passive` → `active_steering`); the harness logs quality, tokens, cost, and latency at each
level and plots the curves across models. Concept from
[*Completion ≠ Collaboration*](https://arxiv.org/abs/2510.25744) (Wu et al., 2025).

## What it does

- Pluggable **Task / AgentModel / UserSimulator / Judge / Runner** interfaces — add a new task, model, or user
  type via config.
- Logs utility (LLM-as-judge), tokens, **cost ($)**, and **latency (s)** per run.
- Produces **utility-vs-effort** and **cost/latency-vs-utility** curves across models.
- Runs end-to-end on a **mock model** with zero API cost for development and CI.

## Results

_Coming soon — utility-vs-effort and cost-vs-utility plots + headline takeaways will be added here after the
first full run._

## Quickstart

```bash
git clone <repo-url> && cd eval-harness
uv sync   # installs the pinned Python 3.12 + exact locked deps (get uv: https://docs.astral.sh/uv/)

# No-cost smoke run on the mock model
uv run python -m collab_eval.runner --config configs/smoke.yaml

# Tests + lint (no keys, no network)
uv run pytest && uv run ruff check .

# Real run (coming in M1 — set OPENAI_API_KEY / ANTHROPIC_API_KEY first)
uv run python -m collab_eval.runner --config configs/experiment.yaml

# Render plots from a results file (coming in M1)
uv run python -m collab_eval.analysis --results results/<run>.jsonl
```

## Configuration

Everything is config-driven — no hardcoded parameters. Define the experiment matrix in `configs/*.yaml`
(tasks, models, effort levels, seeds, judge rubric).

## Architecture

| Module | Role |
|---|---|
| `tasks/` | Iterative tasks + scoring (`trip_planning`, `csv_cleaning`) |
| `models/` | LLM wrappers behind one interface (+ `mock.py`) |
| `user_sim.py` | Simulated user at configurable effort levels |
| `judge.py` | LLM-as-judge with a versioned rubric |
| `runner.py` | Orchestrates the matrix; logs transcript, tokens, cost, latency |
| `analysis.py` | Computes curves and renders plots |

## Extending it

Add a `Task` or `AgentModel` implementation, register it (one line in the module's registry), and
reference it by name in config. The user simulator is deliberately a single concrete class — its
backing model, temperature, and effort levels are all config, not subclasses.
See [`docs/architecture.md`](docs/architecture.md) for the system design, interfaces, and decision log,
and [`docs/PROJECT.md`](docs/PROJECT.md) for the research spec and milestones.

## Limitations

Small sample sizes, LLM-as-judge bias, the realism of the simulated user, and prompt sensitivity all affect the
results. Treat the curves as directional, not definitive. See the writeup for details.

## References

- Wu et al., *Completion ≠ Collaboration: Scaling Collaborative Effort with Agents*, arXiv:2510.25744 (2025).
- Lee, Liang, Yang, *CoAuthor* (CHI 2022).
- Lee et al., *Evaluating Human-Language Model Interaction (HALIE)* (TMLR 2023).

## License

MIT
