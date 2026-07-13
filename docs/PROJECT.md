# PROJECT.md — Collaborative-Effort Evaluation Harness

> A reusable, config-driven harness that measures how an LLM agent's usefulness scales with user
> involvement across multi-turn tasks — reproducing the core finding of *Completion ≠ Collaboration*
> (Wu et al., 2025) and extending it with a **cost/latency-vs-utility** dimension the paper doesn't cover.

---

## 1. Motivation & design goals

Most agent evaluation is **one-shot and single-turn**, which misses that real goals are underspecified and
evolve through interaction. This project targets the under-tooled space of **multi-turn, collaborative**
evaluation: how much more useful does an agent become as the user gets more involved?

Goals:
- **Reproduce a recent finding** — the "collaborative effort scaling" idea from *Completion ≠ Collaboration*.
- **Extend it** with a dimension the paper doesn't cover: **cost/latency-vs-utility** (does more collaboration
  pay off per dollar and per second?).
- **Ship reusable tooling, not a one-off script** — pluggable Task / Model / UserSimulator / Judge interfaces,
  config-driven, tested, containerized, so anyone can add a new task, model, or user type.
- **Reproducibility as a first-class feature** — pinned deps, fixed seeds, one-command runs.

**Definition of "done":** a public repo where one command reproduces a utility-vs-effort curve for 2+ models on
2 tasks, plus a cost/latency-vs-utility curve, with a clean README and honest limitations.

---

## 2. The idea

**Source paper:** *Completion ≠ Collaboration: Scaling Collaborative Effort with Agents* — Shen, Chen, …,
Tongshuang Wu, David Sontag. arXiv:2510.25744 (2025). https://arxiv.org/abs/2510.25744

**Core concept — "collaborative effort scaling":** rather than scoring agents on one-shot *completion*, measure
how an agent's utility grows with **increasing user involvement** across multi-turn interaction. The paper finds
that state-of-the-art agents often **underperform in multi-turn scenarios** — the missing ingredient is the
ability to **sustain engagement and scaffold user understanding**.

**This harness** operationalizes that: an "agent" LLM works an iterative task over several turns; a second LLM
plays the **user** at configurable involvement/effort levels; the harness logs output quality (and cost/latency)
at each level and plots the resulting curves across models.

**What's novel here (beyond reproduction):**
1. **Cost/latency-vs-utility** curves — not in the original paper.
2. **A reusable framework** — pluggable interfaces so new tasks/models/user-types drop in via config.
3. (Stretch) **Robustness** — sensitivity of the finding to the user-simulator prompt and judge model.

---

## 3. Architecture

Pluggable interfaces (small and swappable):

- **`Task`** — an iterative task: initial user goal/state and how to score a candidate output (task metric or
  LLM-judged rubric). Start with `trip_planning`, `csv_cleaning`.
- **`AgentModel`** — wraps an LLM behind one interface (`next_turn(conversation) -> message`). Config: provider,
  model, temperature. Providers: OpenAI + Anthropic to start.
- **`UserSimulator`** — an LLM prompted to play the user at a configurable **effort level**
  (`passive`, `moderate`, `active_steering`); produces user turns and decides when to stop.
- **`Judge`** — scores final (and optionally intermediate) output quality; LLM-as-judge with a versioned rubric.
- **`Runner`** — orchestrates the matrix (task × model × effort × seed) and logs everything: transcript,
  per-turn tokens, **cost ($), latency (s)**, and scores → JSONL + CSV.
- **`analysis`** — computes utility-vs-effort and utility-per-dollar / per-second curves; renders plots.
- **`config`** — YAML defining the experiment matrix and all knobs. No hardcoded params in code.

Suggested layout:
```
collab-effort-eval/
  README.md
  pyproject.toml / requirements.txt
  Dockerfile
  configs/           # smoke.yaml (mock), experiment.yaml (real)
  src/collab_eval/
    tasks/           # trip_planning.py, csv_cleaning.py, base.py
    models/          # openai.py, anthropic.py, base.py, mock.py
    user_sim.py
    judge.py
    runner.py
    analysis.py
  results/           # JSONL + CSV + PNG (gitignored except samples)
  tests/
  .github/workflows/ci.yml
```

**Stack:** Python 3.12, OpenAI + Anthropic SDKs, pydantic (config), pandas + matplotlib (analysis), pytest,
Docker, GitHub Actions CI. Optional: vLLM (open-weight models), Weights & Biases (run tracking).

---

## 4. Milestones

Each milestone has a **Definition of Done (DoD)**. Status markers: ✅ done · 🔜 next · ⬜ not started.

### ✅ M0 — Scaffold & plumbing
Repo skeleton, config loading, interfaces defined, a **mock model** so the loop runs with no API cost.
- **DoD:** `python -m collab_eval.runner --config configs/smoke.yaml` runs end-to-end on the mock model and
  writes results JSONL/CSV; `pytest` passes.
- **Status (2026-07-05):** DoD met and review-hardened — `uv run python -m collab_eval.runner --config
  configs/smoke.yaml` writes 6 mock episodes to JSONL + CSV; 28 tests green; config rejects duplicate
  matrix cells and unique model labels are episode identity; keyless CI pulled forward from M3.
  See `docs/architecture.md` for what was built and why.

### ✅ M1 — MVP (reproduce the core idea)
1 real task (`trip_planning`), 2 models, 3 effort levels, LLM-as-judge rubric; produce the **utility-vs-effort** plot.
- **DoD:** a plot showing utility rising with involvement, and a note on where agents plateau/underperform.
- **Status (2026-07-12): DoD met** — see the README's "First results" for the figure and note.
  24 real episodes (`claude-haiku-4-5` + `gpt-5.4-mini` agents, `gpt-5.4-mini` sim,
  `claude-haiku-4-5` checklist judge, $1.80): utility rises passive → moderate for both models
  (+0.16/+0.24 met-fraction) then plateaus — extra effort past moderate bought no coverage.
  Manipulation check confirms behaviorally distinct effort levels (23/141/286 words per user
  message); the judge passed a 21-artifact human-labeled golden set 148/148 (incl. injection
  probes) before any matrix spend. Infrastructure landed across chunks 1–7: prompt fingerprints,
  OTel span layer, real provider wrappers + versioned pricing + response cache, the
  `trip_planning` task with isolated context channels, the checklist `LLMJudge` with
  consolidation turn and bounded repair-retry, `analysis.py` (figure + manipulation summary),
  and `judge_validation` as a pre-run CLI gate. 137 tests green; decisions 13–22 in
  `docs/architecture.md`.

### 🔜 M2 — The extension (cost/latency-vs-utility)
Add token/cost/latency logging and render **utility-per-dollar** and **utility-per-second** curves; add a 2nd
task and/or an open-weight model.
- **DoD:** a cost-vs-utility plot (not in the paper) plus a short written takeaway.

### ⬜ M3 — Reproducibility & engineering polish
Dockerfile, pinned deps, fixed seeds, config-driven matrix, unit tests + GitHub Actions CI (mock model, no
keys — landed early, in M0), response caching.
- **DoD:** `docker run … --config configs/experiment.yaml` reproduces results from a clean machine; CI green;
  adding a new task/model is documented and small.

### ⬜ M4 — Writeup & release
README with question, method, findings (plots), **honest limitations**, and a "how to add a task/model/user type"
section.
- **DoD:** public repo; README skimmable in two minutes.

---

## 5. Design principles & non-goals

- **Not a general eval framework.** Don't reinvent lm-eval-harness / HELM. Stay focused on multi-turn
  collaborative eval — the thin-tooling area.
- **Keep the matrix small.** 2 tasks × 2–3 models × 3 effort levels × a few seeds.
- **Reproducibility is a feature, not an afterthought.**
- **Measure cost/latency from day one** — it's cheap to log and it's the differentiator.
- **Write honest limitations** — small samples, LLM-judge bias, and user-simulator realism are real caveats.
- **Budget guardrails:** small/cheap models for iteration; cache responses; cap total spend (e.g., < $30).

---

## 6. Stretch ideas (post-v1)

- Robustness: vary the user-simulator prompt and the judge model; report sensitivity.
- New user types: `novice`, `expert`, `adversarial`.
- Open-weight models via vLLM (does effort-scaling hold for small models?).
- A small human spot-check to sanity-check the LLM judge.
- Contribute an improvement upstream to a related open-source project (e.g., Sotopia / HAICOSYSTEM).

---

## 7. References

- Wu et al., *Completion ≠ Collaboration: Scaling Collaborative Effort with Agents*, arXiv:2510.25744 (2025).
- Lee, Liang, Yang, *CoAuthor* (CHI 2022). https://coauthor.stanford.edu
- Lee et al., *Evaluating Human-Language Model Interaction (HALIE)* (TMLR 2023). https://arxiv.org/abs/2212.09746
