# Defects found, fixed, and still open

> Record of what testing the agent actually turned up, 2026-08-30/31. Written so that
> someone who was not present can tell what is safe to rely on and what is not.
>
> Everything below was found by *running* the system, not by reading it. Three of the
> most serious defects were invisible to inspection and only appeared under a live run.

## Summary

| | Count |
|---|---:|
| Defects fixed | 12 |
| Context/documentation corrections | 9 |
| **Open — not fixed** | **8** |
| Of the open items, blocking a measured run | 0 |
| Of the open items, materially affecting score | 2 |

---

# Part 1 — Fixed

## F1. Guard rejected every training-loop change *(critical)*

**Symptom.** Nine consecutive LLM iterations across three measured runs were rejected
before execution. Zero experiments ran. Two runs designated the baseline at a 0.0000
delta. Cost of the symptom: ~$3.49.

**Cause.** `harness/guards.py::scan_diff` flattened the added lines of *every* file in a
diff into one blob and scanned it as `pipeline/<diff>`. `scan_source` applies the
feature-builder label rule to anything whose path ends in `<diff>`. Since `agent/coder.py`
emits whole files, every line of `pipeline/train.py` reappears as an added line — and
`train.py` legitimately reads `IDX['label']` on lines 26, 32 and 42 to build training
targets and to hand ground truth to `evaluate()`.

**Proof.** The guard rejected the **pristine, unmodified `train.py`** when re-emitted
whole. No change to the training loop could ever have passed: loss function, batching,
early stopping, model selection were all sealed off — including the organizers' headroom
idea 1, which they rate most likely to help.

**Fix.** `_added_by_file()` attributes added lines to their source file, so the
feature-builder rule judges only `features.py`.

**Verified.** Pristine `train.py` accepted; direct label read in `features.py` still
rejected; diff touching `kit/` still rejected. Regression test
`test_guard_scopes_label_rule_to_features_file`. After the fix, all three iterations of
the next run executed and scored.

## F2. Identity-linked API key had no workspace header *(blocking)*

**Symptom.** `400 invalid_request_error: anthropic-workspace-id is required when
authenticating with an identity-linked API key`. No measured run could start at all.

**Cause.** `agent/llm.py` called `anthropic.Anthropic()` with no headers.

**Fix.** Sends `anthropic-workspace-id` from `ANTHROPIC_WORKSPACE_ID` (falling back to
`ANTHROPIC_AWS_WORKSPACE_ID`). Absent when unset, so a classic key is unaffected.

**Verified.** With a deliberately invalid id the error changes to "must be a valid
workspace ID", proving the header transmits; with the real id, calls succeed.

## F3. No legal route to prior-history features

**Symptom.** None observed in a run — see D2 in Part 4 for how this was misdiagnosed.

**Cause.** The rules permit an outcome column as an input when aggregated from strictly
earlier rows (`AGENT_RULES.md` §3), but nothing made it possible: `same_row()` refuses
`label` by name, the guard rejects any other `IDX['label']` read in `features.py`, and no
helper existed.

**Fix.** `harness/history.py` — `prior_stats` / `prior_rate` / `prior_count` /
`bucketize`. Lives on the harness side, so the guard needed no weakening.

**Verified** against the real 1.44M rows:

| Property | Result |
|---|---|
| Brute-force recomputation, 400 sampled valid rows | exact match |
| 3,665 validation tie groups | 0 where members saw each other |
| Flip **every** test label and recompute | every output byte-identical |
| Row includes itself | never |

Plus `test_history_is_strictly_causal` and `test_history_excludes_test_outcomes`. Used
successfully by iteration 2 of run `20260830T155011Z`.

## F4. Divide-by-zero in the history helper

`prior_weight=0` (a legitimate request for the raw unsmoothed rate) on a row with no
history produced `0/0`. Fixed with `_smooth()`, which falls back to the train global mean
when the denominator is empty. Found by the regression tests written for F3, before it
could reach a run.

## F5. Single-seed accept/reject decision

**Symptom.** Every measured hypothesis ran on exactly one seed. The agent's own
falsification criteria repeatedly demanded more — *"3-seed paired delta"*, *"at matched
seeds"*, *"fails to reproduce on matched seeds"* — and were never once honoured.

**Cause.** `agent/controller.py::_plan` returned a hardcoded `[0]` on the LLM path, and
the proposer schema has no `seeds` field.

**Why it mattered.** With `ACCEPT = 0.0014` and single-seed paired σ ≈ 0.0005–0.0007,
rejection was safe (observed deltas were 4–5σ) but **acceptance was near a coin flip** —
a true improvement and a true zero sit ~2σ apart. Worse, a false negative does not merely
lose an iteration: convergence fires after 3 non-improving iterations, so one unlucky seed
burns one of the ~3 shots the run actually gets.

**Fix.** `CONFIRM_SEEDS = [0, 1, 2]`, `CONFIRM_FLOOR = -0.0007`. A single-seed delta below
the floor is rejected outright (cheap — all observed rejections qualify). Anything else is
re-measured on matched seeds, and the decision uses the **mean per-seed primary**. `KEEP`
additionally requires no individual seed to fall below the parent, and an unconfirmed
single-seed gain is never kept.

**Important subtlety.** `run_node` *rank-averages* multiple seeds into an ensemble, and
that ensemble score is systematically higher than any single seed. Using it as the
confirmation metric would flatter every candidate, so the fix reads `per_seed_primary` and
takes the mean instead.

**Verified free, in dry run.** Iteration 1's ensemble read **+0.00099** — most of the way
to the bar — while the honest mean per-seed delta was **+0.00031** with one seed *below*
parent. Rejected on both grounds. That gap is the exact failure the fix exists to prevent.

## F6. Confirmation was untestable without paying

The first draft of F5 gated confirmation behind `not self.dry_run`, which would have made
it impossible to verify without a live LLM run. Caught before shipping; the gate was
removed, and F5 was then verified for $0.

## F7. `build_packet.py` silently regenerated human-reviewed files

`context/constraints.md` and `context/references.md` were held as Python string literals
inside `build_packet.py` and **rewritten on every run**. Copying the reviewed evidence in
would have been reverted by the next `python context/build_packet.py`.

**Fix.** The script now *reads* both from disk via `_read()` and never writes them; its
output distinguishes `wrote` from `read … (human-reviewed, not regenerated)`.

## F8. Proposer could not see the reviewed data profile

`agent/proposer.py` has no file-read tool — it sees `packet.md` and the journal digest,
nothing else. `research/data_profile.md` was not in the packet, so measurements demoted
there (see D2) were invisible to the agent. `build_packet.py` now reads it into the packet.

## F9. `AGENT_RULES.md` pointed at guard-forbidden raw CSV reads

§5 item 2 instructed the agent to read `log_standard_*.csv` directly from `features.py`.
The guard rejects exactly that (`open(...log_standard...)`), because that file spans
validation *and* test and carries `long_view`. Superseded; now points at
`harness.adapter` and `harness.history`.

## F10. Cache TTL and its pricing

The packet block used the default 5-minute ephemeral cache. Node training of 30–95s,
tripled by confirmation runs, pushes consecutive calls past that window — run
`20260830T153555Z` paid for **two** writes of the ~33k-token packet.

**Fix.** `ttl: '1h'`, and the meter's cache-write multiplier moved 1.25x → **2.0x** to
match 1-hour pricing. Without the price change the budget ceiling and the Feasibility
figures reported to judges would have been understated.

**Verified live, definitively.** A controlled probe on a small unique block returned
`cache_creation: {ephemeral_1h_input_tokens: 7214, ephemeral_5m_input_tokens: 0}` — the
write lands in the 1-hour bucket, so the meter's 2.0x price is correct and the Feasibility
figures are honest. A follow-up read also hit the packet's entry long after a run had
ended, which a 5-minute entry could not have survived.

Note the first verification I ran was *not* sufficient: a write-then-read pattern cannot
distinguish a 5-minute cache from a 1-hour one, and I initially reported it as confirmed
on that basis. The `cache_creation` breakdown is the only field that settles it.

**Unexplained, and now instrumented — see O8.** Run `20260830T164210Z` still recorded two
packet writes (66,014 = 2 x 33,007) across six calls despite the 1-hour TTL.

## F11. Token estimate understated by ~60%

`build_packet.py` printed `chars // 4`. The API's real count for this packet is 33,007
against an estimate of 20,610. Recalibrated to `/2.5`; now reports ~33,657 against 33,007,
within 2%.

## F12. Missing `anthropic` SDK

Not installed. Installed 1.2.0. Environmental, but it blocked every measured run.

---

# Part 2 — Open. Not fixed.

## O1. The parent is still a single-seed measurement *(affects score)*

F5 confirms the **candidate** on 3 seeds but compares it against a parent measured on one.
Comparison σ drops from ~0.0007 to ~0.00035 — better, not zero.

**Risk.** A candidate within ~0.0004 of the bar can still be decided by the parent's seed
luck.

**Cost to fix.** Re-measure the parent on the same seeds at every comparison, roughly
doubling training time per confirmed iteration. Training is not the binding constraint
(100 baseline iterations ≈ 28 min of one CPU core), so this is affordable if wanted.

## O2. The run effectively gets ~3 attempts, not 50 *(affects score)*

`_converged()` runs on the **best-so-far** trajectory: if three consecutive iterations fail
to improve, best-so-far is flat, the difference is 0 ≤ ε, and the run stops and designates
the validation-best checkpoint.

This is the organizers' rule implemented correctly, and `HARNESS.md` defends it — so it is
**not a bug**. But it is the single largest strategic risk: the 50-iteration cap is not the
binding constraint, the convergence rule is. Run `20260830T155011Z` tried three sound
ideas, lost all three by 0.0023–0.0028, and stopped.

**No fix proposed.** Changing it would break compliance with the scoring rule.

## O3. `coder.py` emits whole files — the dominant token cost

Output is ~66% of spend ($1.158 of $1.7616 in run `20260830T153555Z`). The coder produces
2.4x the proposer's output *despite* already running at `effort='medium'`, because it
re-emits entire files rather than patches.

**Not fixed:** changing to patch-based editing risks correctness in the component whose
correctness matters most. Flagged as the largest remaining cost lever.

## O4. `harness/history.py` recomputes from scratch on every call

Each `prior_stats()` call re-reads `time_ms` through the adapter (~8s) and walks all
1.27M train+valid rows — ~29s total. Iteration 2 of run `20260830T155011Z` took 94s
against a 56s baseline, largely from this.

**Risk.** An agent building several aggregates (user, video, author keys) pays it once per
call. Four keys ≈ two extra minutes per iteration.

**Fix would be:** memoise on `(signal, key, prior_weight)` within a process, or hoist the
`time_ms` read. Straightforward; not done.

## O5. Nothing loads `.env`

`agent/llm.py` reads environment variables only. The key and workspace id must be exported
by hand; `run_measured.sh` does it, but a bare `python -m agent.controller` will fail with
`LLMUnavailable` even though `.env` holds a valid key.

**Not fixed** to avoid adding a dependency or silent credential loading.

## O6. Noise-floor figure is inconsistent across the repo

`agent/controller.py` comments σ ≈ 0.0007; `HARNESS.md` and `context/baseline.json` say
paired σ ≈ 0.0005; the reviewed evidence uses the published 0.0008 as the safe reference.
`ACCEPT = 0.0014` is described as "2 sigma" but is 2σ only under the 0.0007 reading.

Nothing is *wrong* — they are different measurements — but a reader cannot tell which
governs. Unresolved.

## O8. Two packet cache writes per run, cause unknown

Run `20260830T164210Z` recorded `cache_write = 66,014`, exactly twice the 33,007-token
packet, across six calls — with the 1-hour TTL confirmed active (F10) and the run lasting
only 651s. One write is expected; the second is not.

Ruled out by controlled probe: **effort level does not partition the cache.** Calls at
`effort=high`, `effort=medium` and with no effort config all read the same entry. That was
my first hypothesis and it is wrong.

Still possible: a propagation window between near-simultaneous early calls, an entry
expiring between runs rather than within one, or something in how the role instruction
block interacts with the breakpoint. Not established.

**Cost of the symptom:** one extra write per run, $0.33 at 1-hour pricing. Small now,
worth understanding before a 6-hour run.

**Instrumented rather than guessed at.** `Meter.totals()['by_role']` now carries
`cache_write` and `cache_read` per role, so the next run identifies which call wrote
without further probing. No spend required to diagnose it.

## O7. The proposer cannot request seeds

Deliberately not added. F5 makes confirmation automatic and unconditional; making it
something the LLM must remember to ask for would reintroduce the unreliability being
fixed. Recorded here because it was proposed and then rejected on purpose, not forgotten.

---

# Part 3 — Context-package corrections

Found by auditing the six context files against the official problem statement
(last updated 27 Aug 2026). Detail in `context/CONTEXT_UPDATE_REPORT.md`.

| # | Correction |
|---|---|
| C1 | **`RULES.md` had no "no external training data" rule** — the statement's single hard resource rule was entirely absent |
| C2 | **Agent was under-permitted** — any open-source framework (PyTorch, RecBole, TorchRec, LightGBM), papers, and pretrained weights are in scope; `references.md` had cautioned "the official kit is numpy-only" |
| C3 | **Scoring formula absent** — `delta(m) = score_agent(m) − score_baseline(m)`, averaged over metrics |
| C4 | **Feasibility described wrongly** — wall-clock *replaces* GPU-hours as the scored measure; grading is three coarse tiers; iteration count is not itself scored |
| C5 | **"Converged result, not the peak"** and the validation-best-checkpoint rule were missing |
| C6 | **Oracle 0.8645 was unlabelled** — that is the *test* oracle; validation is 0.8484. Development progress judged against the wrong ceiling |
| C7 | **Seed std 0.0008 unscoped** — it is the organizers' *test* std over 5 seeds |
| C8 | **170,588 mislabelled** as an official published row count; it is a reproduced date-only count |
| C9 | **Stale artifact identified and not used** — `research/experiment_results/video_feature_inventory.csv` profiles 7,545 video rows, not 7,583, and reports `like_cnt` mean 231.84 rather than the verified 230.75 |

---

# Part 4 — Errors I made during this work

Recorded because they cost real money and would otherwise be invisible.

**D1. I spent $3.49 confirming a fix for the wrong problem.** After the first failed run I
inferred the cause from a one-line console message instead of opening the journal, which
was sitting on disk and named the real cause. Two measured runs were spent before I read it.

**D2. I misdiagnosed F1 as F3.** I told the user the blocker was the missing history
helper. The journal shows all six LLM iterations proposed a within-user listwise softmax,
and every rejected line was pre-existing `train.py` code. The history helper is a real gap
and is now used by the agent, but it fixed nothing that was failing.

**D3. I claimed the proposer was a monoculture.** It was not — that was an artifact of the
blocked runs, where the proposer kept re-attempting the one idea the guard killed.
Unblocked, it produced three distinct hypotheses covering headroom ideas 1, 2 and BPR.

**D4. I wrote a false claim into `AGENT_RULES.md` §0b** — that a measured run "spent all
three iterations proposing history features". Corrected, and §0c added to document the
actual defect.

**D5. I over-promised on packet trimming.** I pushed two trims as meaningful; together they
recovered ~1,250 tokens (~4%), and the first was under 1% of the packet. Output tokens are
66% of cost — packet size was never the lever.

**D6. I asserted both LLM roles ran at `effort='high'`.** The coder was already at
`medium`. I read the default in `llm.py`'s signature instead of the call sites.

---

# Part 5 — Cost ledger

| Item | Cost |
|---|---:|
| Dry run, 3 iterations | $0.0000 |
| Measured run 1 (`20260830T151005Z`) — 0 iterations, guard bug | $1.7291 |
| Measured run 2 (`20260830T153555Z`) — 0 iterations, same bug | $1.7616 |
| Measured run 3 (`20260830T155011Z`) — 3 iterations, working | $1.7667 |
| Dry run, confirmation path | $0.0000 |
| API smoke test | $0.2051 |
| Cache-TTL probe | $0.3470 |
| Measured run 4 (`20260830T164210Z`) — 2 iterations, new code | $1.6262 |
| TTL-bucket diagnostic probes | $0.0353 |
| **Total** | **$7.4710** |

Injection recovery tests (`leak`, `syntax`, `timeout`, `nan`) run in dry-run mode and cost
nothing. All four recovered with `orphan_free=True`.

---

# Part 6 — Test coverage

`python tests/test_harness.py` — **18/18 passing**. Four added during this work:

- `test_history_is_strictly_causal`
- `test_history_excludes_test_outcomes`
- `test_guard_allows_history_helper`
- `test_guard_scopes_label_rule_to_features_file`

The last one is the regression test for F1 and is the most important test in the file: it
asserts that the shipped `train.py` passes its own guard.
