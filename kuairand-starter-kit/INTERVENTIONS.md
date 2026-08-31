# Manual-intervention summary

> Required deliverable (`context/PROBLEM.md` §9). Impact & Relevance is scored
> "primarily by the number of manual interventions required to reach convergence", so
> this document states the number, the evidence behind it, and — deliberately — the
> interventions that happened outside the scored run as well.
>
> Every figure here was read out of the hash-chained journals under `runlogs/`. Where a
> claim rests on absence of evidence rather than positive evidence, it says so.

## 1. The taxonomy

Defined in `harness/journal.py:35-42`. L0 and L1 are recorded but do **not** count
against autonomy; L2-L5 do.

| Class | Meaning | Counts |
|---|---|:--:|
| `L0_setup` | Configuration before `RUN_START`, bounded by the workspace hash | no |
| `L1_observe` | Reading logs; no effect on the run | no |
| `L2_env_repair` | Restart with no code or state change | **yes** |
| `L3_code_edit` | Editing inside the hashed tree while a run is live | **yes** |
| `L4_steer` | Injecting an idea, changing config, killing a branch | **yes** |
| `L5_select` | A human chose the final submission | **yes** |

## 2. The scored run

**`runlogs/run_20260831T011354Z` — 0 counted interventions (L2-L5).**

| | |
|---|---|
| Started | 2026-08-31T01:13:56Z |
| Stop reason | `converged` (eps=0.002, N=3) |
| Wall-clock | 6,259.6 s (1 h 44 m) |
| Designated | iteration 12, validation primary 0.607243 |
| Submission | `submissions/FINAL_submission_agent_best.csv`, sha256 `475d168d…` |

What is actually proven, and how:

- The journal is a SHA-256 hash chain (`harness/journal.py`). `python verify.py --chain
  --order runlogs/run_20260831T011354Z/journal.jsonl` returns **OVERALL: PASS**, so no
  event was inserted, removed or edited after the fact.
- `FINAL_DESIGNATION` precedes any `TEST_OPEN`, which is what makes "selected on
  validation only" checkable rather than asserted.
- No `human_intervention` event appears between `RUN_START` and `FINAL_DESIGNATION`.
- The run recovered from six failures without a human: one node crash, three static-guard
  rejections and two coder role failures (see §5).

**The honest limit of that claim.** `workspace_hash` is written once at `RUN_START` and
never re-verified at designation, and the intervention API (`Journal.intervention`) has
no production caller — its only caller is `tests/test_harness.py`. So the zero is
"nothing was recorded", not "a tamper-evident check found nothing". A human editing the
tree mid-run (an L3) would not have been caught automatically. We are stating this
because a judge who greps for the writer will find only the test, and an unexplained zero
from uninstrumented tooling is worth less than a smaller number that is honestly derived.

## 3. Campaign ledger — every measured run

14 measured runs; 10 reached `FINAL_DESIGNATION`, 4 were aborted.

| Run | Model | Workspace | Cap | Outcome | USD |
|---|---|---|--:|---|--:|
| 20260830T151005Z | opus-5 | cc008273 | 3 | iteration_cap | 1.73 |
| 20260830T153555Z | opus-5 | cc008273 | 3 | iteration_cap | 1.76 |
| 20260830T155011Z | opus-5 | cc008273 | 3 | converged | 1.77 |
| 20260830T164210Z | opus-5 | cc008273 | 2 | iteration_cap | 1.63 |
| 20260830T222356Z | sonnet-5 | cc008273 | 50 | converged | 1.25 |
| 20260830T230438Z | sonnet-5 | 77eff95e | 50 | converged | 1.14 |
| 20260830T234014Z | sonnet-5 | 77eff95e | 50 | **aborted — crash** | n/r |
| 20260830T235541Z | sonnet-5 | 77eff95e | 50 | converged | 1.14 |
| 20260831T003537Z | sonnet-5 | 77eff95e | 50 | **aborted — killed** | n/r |
| 20260831T004013Z | sonnet-5 | 5492542a | 50 | converged | 1.06 |
| 20260831T005426Z | sonnet-5 | 5492542a | 50 | **aborted — killed** | n/r |
| 20260831T005603Z | sonnet-5 | 5492542a | 50 | converged | 2.00 |
| 20260831T011156Z | sonnet-5 | 5492542a | 50 | **aborted — killed** | n/r |
| **20260831T011354Z** | **opus-4-8** | 5492542a | 50 | **converged — SCORED** | **8.66** |

Campaign spend across the 10 designated runs: **$22.13**. Aborted runs report `n/r`
because resources are written only inside `FINAL_DESIGNATION`, so a killed run accounts
for nothing even though it made paid calls — a known reporting gap, recorded in
`BUGS.md`. The campaign total is therefore a **lower bound**.

## 4. Counted interventions at campaign level

These all happened **between** runs, never inside the scored one. `context/PROBLEM.md`
§10 explicitly permits humans to supply task definition, rules, data, codebase, evidence,
references and budget; configuring and launching a run is that. They are listed anyway,
because they are visible in the journals and silence about them is worth less than
disclosure.

| Class | Count | What |
|---|--:|---|
| `L2_env_repair` | 1 | `234014Z` crashed on an unhandled `ValueError` escaping through `main()` (traceback in `runlogs/full_run_config_fixed.log`). Fixed in `agent/llm.py`, relaunched 11 minutes later as `235541Z`. |
| `L4_steer` | 3 | `003537Z`, `005426Z` and `011156Z` were each killed by hand roughly 50 s after the iteration-0 baseline, before any hypothesis was proposed. Matching 3-line logs: `full_run_confirmed.log`, `full_run_r2.log`, `full_run_r3.log`. |
| `L5_select` | 1 | See §6 — a human banked an artifact the agent did not designate. **This one is being reversed.** |
| `L0_setup` | 14 | Per-run configuration: model, iteration cap, budget ceiling. Not counted. |

### Configuration selection

Three model configurations were tried across the campaign: `claude-opus-5` (4 runs),
`claude-sonnet-5` (9), `claude-opus-4-8` (1 — the scored run). Four distinct workspace
hashes reflect harness fixes landing between runs.

The switch that produced the winner is visible and we are not going to leave it to be
discovered: `011156Z` started at 01:12:10Z on sonnet-5, logged its baseline at 01:13:02Z
and was killed; `011354Z` started **54 seconds later** at 01:13:56Z on `claude-opus-4-8`
with the *identical* workspace hash `5492542a`. A human changed the model between two
runs and the second one won.

That is an L0/L4 at the campaign level, not an intervention inside the scored run — the
code was byte-identical (same workspace hash) and the agent proposed every hypothesis
itself. But it is why the scored run is reported as *one* run rather than as the
campaign: choosing the best of 14 runs is a human selection, and the honest framing is
that the human chose the *configuration*, and the agent, unaided, produced the result.

## 5. Self-recovery inside the scored run

Six failures, none of which required a human, all recorded in the journal:

| Iteration | Failure | Handling |
|--:|---|---|
| 2 | node crash (`contract`) | pruned, reverted, reflected on, loop continued |
| 7, 8, 11 | static guard rejected the diff (`evaluate()` on test) | change refused **before execution**, iteration abandoned |
| 9, 10 | coder returned unparseable JSON twice | role failure logged, iteration skipped, streak counter reset on the next good node |

This is the Technical Execution robustness exhibit: the run neither crashed, stalled nor
diverged, and the leak guard refused three separate attempts to score the test split.

**What these six cost.** Every one was *lost*, not recovered — the iteration was
discarded. The retry paths that hand a guard finding or a traceback back to the coder
(`BUGS.md` F17/F18) were written **after** this run and have not yet been exercised in a
measured run. No journal in this repository yet contains a recovered iteration. We are
saying so rather than letting `HARNESS.md`'s "Fixed" imply otherwise.

## 6. The one intervention we are reversing

`RESULTS.md` (as first written) banked `submissions/verified_listwise_3seed_ensemble.csv`
at validation primary 0.604051. That file's sha256 matches **no** `FINAL_DESIGNATION` in
any of the 20 journals, and no script in the repository produces it: it was built by hand.
Choosing it over the agent's own designated artifact is an `L5_select`.

Reversing it does two things at once, which is why it is the single highest-value change
in the project:

1. **Technical Execution** — the agent's designation scores mean-of-deltas **+0.005693**
   against the banked file's **+0.002501**. Shipping the hand-built file forfeits 56% of
   the improvement actually earned.
2. **Impact & Relevance** — with the reversal, the submitted artifact is byte-identical
   to what the agent designated, and the counted intervention total for the delivered
   result returns to **zero**.

The listwise artifact is retained as a documented runner-up, not as the submission.

## 7. Summary

| | |
|---|--:|
| Counted interventions inside the scored run (L2-L5) | **0** |
| Counted interventions across the whole campaign | **4** (1 × L2, 3 × L4) |
| L5 selections in the delivered result | **0**, after the §6 reversal |
| Unattended wall-clock of the scored run | 1 h 44 m |
| Failures the run handled without a human | 6 |
