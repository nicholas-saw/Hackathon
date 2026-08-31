# Results — KuaiRand-Pure autonomous ML research agent

> Every number here is validation-only. The hidden test split was never scored during
> development, and the ordering is checkable: `FINAL_DESIGNATION` precedes any `TEST_OPEN`
> in the hash-chained journal, and `verify.py` enforces it.
>
> All figures come from `runlogs/run_20260831T193724Z/journal.jsonl` unless stated.
>
> This supersedes run `20260831T011354Z` (+0.005693), whose figures are kept in
> section 7 for continuity. The two are directly comparable: same harness, same
> frozen context, same selection rule.

## 1. Final submission

| | |
|---|---|
| File | `submissions/FINAL_submission_agent_best.csv` |
| SHA-256 | `e3f55ed563cfcb6a1b0ccb70dea0e3a57aae46a7cd35dbe16e5159efc0d746de` |
| Rows | 170,588 — `row_id` contract validated by the frozen `kit/submit.py` checker |
| Provenance | Byte-identical to the `FINAL_DESIGNATION` of run `20260831T193724Z`. The agent designated it; **0 human interventions** were logged. |
| Method | Within-user listwise softmax on `long_view` (`fm_listwise_pure`) + strictly-prior user-side aggregate fields, plus a **second listwise head on `is_click`** sharing the FM embedding table. Rank-averaged over 3 seeds. |
| Config | `{"model": "fm_listwise_pure", "k": 16, "lr": 0.001, "bs": 8192, "patience": 4}` on top of the kept feature and multi-task changes |

The pipeline that produced this is reconstructable from the journal — the controller
restores a pristine pipeline when a run ends, so the winning code lives in the run record
rather than the working tree:

```bash
python replay.py runlogs/run_20260831T193724Z/journal.jsonl -o patches/
git apply --directory=kuairand-starter-kit patches/iter001.patch   # objective + prior features
git apply --directory=kuairand-starter-kit patches/iter004.patch   # listwise learning rate
git apply --directory=kuairand-starter-kit patches/iter005.patch   # the is_click auxiliary head
```

The `fm_listwise_pure` objective that iteration 1 builds on is in the tree — see
`pipeline/train.py`. It is **not** `fm_listwise`, which is a different formulation.
## 2. Results table

Validation-best at convergence, against the official published baseline.

| Metric | Official baseline | This submission | Absolute delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | **0.683803** | **+0.016403** |
| nDCG@5 | 0.5357 | **0.544587** | **+0.008887** |
| primary | 0.6016 | **0.614195** | +0.012645 |

Against this run's own reproduced 3-seed baseline (0.602186), the deltas are +0.015702
GAUC and +0.008317 nDCG@5, a mean of **+0.012009**. That is the more conservative
comparison and the one the selection machinery used.

The judged quantity is `delta(m) = score_agent(m) − score_baseline(m)` averaged over
`m ∈ {GAUC, nDCG@5}`:

**mean-of-deltas = (+0.016403 + 0.008887) / 2 = +0.012645**

(That is 0.00005 above the delta in `primary` only because the organizer rounds
(0.6674 + 0.5357)/2 = 0.60155 to 0.6016. The two are otherwise algebraically identical.)

Against the validation oracle ceiling of 0.8484, the baseline captures ~31% of the range
above random; this adds roughly a further 5.1% of what is attainable.

### Requirement 1 — reproducing the official baseline

Iteration 0 of every measured run re-runs the untouched baseline. The **single-seed**
reproduction is bit-identical across all 15 measured runs:

| Metric | Official | Reproduced | Delta | vs published seed sd (0.0008) |
|---|---:|---:|---:|---:|
| GAUC | 0.6674 | 0.667133 | −0.000267 | −0.33 σ |
| nDCG@5 | 0.5357 | 0.535806 | +0.000106 | +0.13 σ |
| primary | 0.6016 | 0.601470 | −0.000130 | −0.16 σ |

Every metric lands inside a third of one seed standard deviation, from 15 independent
journalled runs. That is the reproduction claim.

> The scored run's iteration 0 reports **0.602186**, not 0.601470, because
> `SEEDS_PER_NODE = 3`: its baseline node is a rank-average of seeds 0, 1 and 2, and that
> ensemble sits about +0.0007 above any single seed through variance reduction alone. The
> table above is the single-seed comparison against the organizer's published single-model
> figure. Every delta reported for the submission is measured against the 3-seed baseline,
> which is the stricter of the two.

### Why this candidate, and not the largest number available

Selecting the best node of a run on a metric whose seed noise is ±0.0008 is exactly the
winner's curse that `harness/selection.py` exists to prevent. Unlike the previous run,
this one **ran that gate itself, before designating**, so the `FINAL_DESIGNATION` carries
its own `selection` block rather than an after-the-fact argument.

Five candidates were stability-tested against the reproduced 3-seed baseline over five
disjoint user folds, needing 4 of 5 folds to count as stable:

| Candidate | Pooled delta | Folds won | Fold mean / sd | Stable |
|---|---:|---:|---:|:--:|
| **iteration 5 (chosen)** | **+0.01203** | **5 / 5** | +0.01207 / 0.00329 | **yes** |
| iteration 4 | +0.01193 | 5 / 5 | +0.01197 / 0.00306 | yes |
| iteration 1 | +0.01125 | 5 / 5 | +0.01131 / 0.00415 | yes |
| iteration 3 (harness ensemble) | +0.00655 | 5 / 5 | +0.00658 / 0.00196 | yes |
| iteration 0 (baseline) | 0.00000 | 0 / 5 | 0.00000 / 0.00000 | no |

Three things make this a decision rather than an argmax:

- **The gate rejected a tempting option.** A diverse 4-member ensemble of every stable
  candidate was built and measured, and it *lost* to the single winner: pooled −0.00044,
  winning 1 fold of 5. It was discarded. A selector that only maximises would have kept
  the ensemble on the strength of its members.
- **It clears the previously banked floor on every fold.** Against
  `verified_listwise_3seed_ensemble.csv`, the pooled delta is **+0.01145**, 5 folds of 5,
  fold mean +0.01149 (sd 0.00284).
- **It is the agent's own designation, unedited.** The submitted bytes hash to
  `e3f55ed5…`, identical to the `submission_sha256` sealed in the journal before any test
  row was touched, and `verify.py` confirms `FINAL_DESIGNATION` precedes any `TEST_OPEN`.
  `INTERVENTIONS.md` counts **0** interventions in this run.

The margin over the runner-up (iteration 4) is only +0.00010, well inside noise — the two
are not meaningfully different, and either would have been defensible. What matters is
that both sit ~+0.012 above baseline and every fold agrees.

**Where the gain came from.** The lineage is three steps, and the first is the largest:

| Iter | Change | Paired mean | Worst seed |
|---|---|---:|---:|
| 1 | `fm_listwise_pure` objective **+** strictly-prior user-side aggregate fields | +0.01097 | +0.01068 |
| 4 | lower learning rate for the listwise arm | +0.00112 | +0.00092 |
| 5 | second listwise head on `is_click`, sharing the FM embeddings | +0.00012 | −0.00009 |

Iteration 1 is superadditive: the objective alone measures +0.00162 and the user-side
aggregates alone +0.00281 (`constraints.md` C25, C28), which would predict ~+0.0044
together, not +0.0110. The plausible reading is that the two are complementary rather
than additive — under pointwise BCE a prior-rate field mostly encodes absolute
propensity, which the metric is invariant to, while a within-user softmax can exploit the
same field's *variation inside the list*. That is a hypothesis about the interaction, not
a measured decomposition; isolating it would need a 2x2 ablation that this run did not do.

Iterations 4 and 5 are both below the 0.0014 accept bar and were kept by the unanimous
gate (`UNANIMOUS_ACCEPT`, BUGS.md F19), which is calibrated for a matched-seed mean rather
than a single draw. Iteration 5's worst seed is −0.00009, i.e. essentially flat; it wins
the designation on pooled fold performance, not on its own increment.

## 3. Resource consumption

The scored run — the one that reached the submitted result. These are the two figures
Feasibility is graded on.

| | |
|---|---:|
| LLM calls | 15 |
| Input tokens | 124,241 |
| Output tokens | 185,756 |
| Cache write / read | 34,669 / 480,749 |
| **Total tokens** | **825,415** |
| **Agent wall-clock to convergence** | **3,667.1 s (1 h 1 m)** |
| Cost | $5.85 of a $14.00 ceiling |
| GPU-hours | 0.0 (CPU only) |
| Iterations against the 50 cap | 5 attempted, 4 scored |
| Stop reason | `converged` (eps = 0.002, N = 3) |

Per role: proposer 5 calls / $1.30 · coder 6 calls / $4.34 · reflector 4 calls / $0.21.
Model: `claude-opus-4-8` throughout. Prompt caching with a 1-hour TTL; `cache_working:
true`, 480,749 tokens served from cache against 34,669 written.

> `iterations_used` in the journal payload reports **scored nodes** (4), not iterations
> consumed against the cap (5). One iteration was lost: iteration 2 failed the pipeline
> contract twice, the second time after the traceback had been returned to the coder for
> a retry (BUGS.md F18). Iteration 3 was the harness's own zero-token ensembler.

This run is roughly **half the cost and 40% less wall-clock than the superseded run**
(§7: $8.66, 1 h 44 m) while more than doubling the result, mostly because it reached its
best node on the first iteration rather than the tenth.

**Whole campaign, for completeness.** 15 measured runs, of which 11 reached a
designation: **$27.98**. Aborted runs record no resources at all (they are written only
inside `FINAL_DESIGNATION`), so the campaign total is a lower bound. The single scored run
above is the figure that corresponds to the submitted result.

## 4. What the agent found

All hypotheses below were the agent's own, formed from the evidence packet and the
persistent direction registry. Deltas are against the running parent.

| Iteration | Hypothesis | Result |
|--:|---|---:|
| 1 | **`fm_listwise_pure` objective + strictly-prior user-side aggregate fields, together** | **KEEP +0.01120** |
| 2 | Widen the prior aggregates to item-side quantile buckets | failed the pipeline contract twice |
| 3 | *(harness ensembler, zero tokens)* rank-average of iterations 1 and 0 | −0.00475 |
| 4 | **The listwise arm is under-trained: lower its learning rate** | **KEEP +0.00112** |
| 5 | **Second listwise softmax head on `is_click`, sharing the FM embeddings** | **designated, +0.00012** |

The trajectory is short because the first iteration was right.

**It combined two things nobody had combined.** The registry recorded a within-user
listwise objective verified at +0.00162 and user-side prior aggregates verified at
+0.00281, as separate results from separate runs. The agent's opening move was to put
them together, and the pair measured **+0.01097 on matched seeds, worst seed +0.01068** —
roughly 2.5x what adding the two known effects would predict. Whatever the mechanism, it
is not a lucky seed: all three paired seeds moved together and every one of five disjoint
user folds prefers it.

**It reversed the previous run's conclusion.** The superseded run (§7) reported that the
listwise objective "does not help *on top of* the prior-aggregate features". This run
shows it helps enormously. The two are not contradictory measurements of one thing — they
are measurements of *different implementations sharing a name*. The earlier run's listwise
mixed a pointwise BCE term into the loss, capped user groups, and admitted any group
holding a positive; `fm_listwise_pure` does none of those. Nine implementations of
"within-user listwise softmax" span −0.00318 to +0.00162 across this project
(`constraints.md` C25), a spread 5.2x wider than seed noise. The label was never the
experiment.

**Its last idea came from the metric's own geometry.** Iteration 5 added a second listwise
head on `is_click`, sharing the FM embedding table, with an explicit rationale: the pure
`long_view` listwise loss skips uniform-label groups because they carry no ordering
signal, and ~42% of validation users are uniform-label (`constraints.md` C9). An auxiliary
head on a *different* signal supplies ranking gradient for exactly the users the primary
loss cannot reach. That is the organizers' multi-task direction, arrived at from a
constraint rather than from the ranked list.

Its measured increment is small (+0.00012, worst seed −0.00009) and it wins the
designation on pooled fold performance rather than on that increment. The idea deserves a
cleaner test than this run gave it.

**What this run did not do.** It did not decompose the iteration-1 interaction, so
"objective and features are complementary" remains a hypothesis rather than a measurement.
It did not revisit `grouped_batching_cost` — the finding that the user-grouped batching
these losses require costs −0.00239 on its own
(`research/objective_ablation/FINDINGS.md`), which remains the largest identified and
unclaimed gain on the table.

## 5. Reproduction

```bash
set PYTHONUTF8=1
cd kuairand-starter-kit

python -m harness.cache                    # build the encoded cache (~22 s)
python context/verify_context.py           # the frozen context package is unaltered
python tests/test_harness.py               # 42 invariants
python -m harness.guards                   # static leak scan

python -m agent.controller --iterations 50 --budget 14
python verify.py --chain --order runlogs/<run_id>/journal.jsonl
python report.py runlogs/<run_id>/journal.jsonl -o report.html
```

Requires `ANTHROPIC_API_KEY`, and `ANTHROPIC_WORKSPACE_ID` for identity-linked keys. See
the [root README](../README.md) for setup.

Rendered report for the scored run:
[`runlogs/run_20260831T193724Z/report.html`](runlogs/run_20260831T193724Z/report.html).

## 6. Honest limitations

- **The gain is still small in absolute terms.** +0.0126 mean-of-deltas against a 0.8484
  oracle ceiling — about 5% of the range above random that remains attainable.
- **Validation-to-test transfer is unverified**, by design: test is scored once. Expect the
  test delta to be smaller. One specific reason to expect that here: `harness/history.py`
  gives a validation row an expanding window that includes *earlier validation* rows,
  while a test row receives train+valid state and no earlier test rows. The feature is
  therefore slightly better-informed on validation than it can be on test, and this run
  leans on it heavily.
- **The headline interaction is unexplained.** Iteration 1 measured +0.01097 where the two
  known component effects sum to ~+0.0044. The complementarity story in §4 is a hypothesis;
  the 2x2 ablation that would test it was not run.
- **The designated increment is not the gain.** Iteration 5 wins on pooled fold
  performance with an increment of +0.00012 (worst seed −0.00009). Iteration 4 is +0.00010
  behind it and statistically indistinguishable. Essentially all of the result is
  iteration 1.
- **Convergence bounds the search, not the cap.** Three consecutive non-improving
  iterations end a run, so this run used 5 of 50 iterations.
- **One iteration was still lost.** Iteration 2 failed the pipeline contract, was handed
  its traceback, and failed again. The retry paths (`BUGS.md` F17/F18) were exercised in a
  measured run for the first time here and did not save it — they bounded the damage to
  one iteration rather than ending the run.
- **Most iterations lost across the project went to harness defects, not bad science.**
  Nineteen are written up in `BUGS.md`, several of which were silently producing false
  negatives.
- **A human chose the model configuration between runs**, not inside one, and fixed a
  client-side SDK error before this run could make a single model call (`BUGS.md`; the
  run it aborted, `20260831T193129Z`, spent $0.00). The full accounting is in
  `INTERVENTIONS.md`, which counts 0 interventions inside the scored run.

**One previously-listed limitation no longer holds.** Earlier versions of this document
said the direction registry "is not persisted, so separate runs re-proposed and re-refuted
the same listwise hypothesis repeatedly". `context/directions.json` now persists across
runs, and this run is the evidence that it matters: its first hypothesis explicitly
combined two results that *earlier separate runs had each found and then forgotten*, and
that single iteration is essentially the whole submitted gain.

## 7. Superseded run — `20260831T011354Z`

Kept for continuity. Same harness, same frozen context, same selection rule; it simply
scored lower, and its submission is retained in git history.

| Metric | Official baseline | That submission | Absolute delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | 0.675137 | +0.007737 |
| nDCG@5 | 0.5357 | 0.539348 | +0.003648 |
| primary | 0.6016 | 0.607243 | +0.005643 |

mean-of-deltas **+0.005693**. Method: strictly-prior user-level behavioural aggregates as
bucketized FM fields, rank-averaged over 5 independently-seeded models
(`{"model": "fm", "n_models": 5}`). Designated node: iteration 12 of that run.

Re-measured on 2026-09-01 through `replay.py` and `selection.compare`, it reproduced
0.6072425246238708 — identical to its journal designation to the last digit — and passed
the 5-fold gate 5/5 with a pooled delta of +0.00510 against the 3-seed baseline. It is a
real result. The current submission is 2.1x larger on the judged quantity.
