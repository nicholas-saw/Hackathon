# Controlled objective ablation — pointwise vs pairwise vs listwise

**Date:** 2026-08-31 · **Command:** `python research/objective_ablation/ablate.py --seeds 0,1,2`
**Raw:** `results.json` · **Log:** `run.log` · **Scope:** validation only; no test label was read.

## The question

Nine independently written "within-user listwise softmax" implementations scored between
−0.00318 and +0.00162 (`constraints.md` C25). Seven lost, one won, and the winner became
the banked submission. The obvious next question is *not* "shall we try a tenth listwise
implementation" — it is **which property of the winner is doing the work.**

## Design

One model. One batching scheme. Matched seeds. Exactly one thing varies: the per-batch
update rule.

Held fixed: encoded features and split, FM architecture and `k`, initialisation, Adam +
learning rate + L2, the epoch budget, early stopping on validation primary with
`patience=4`, the seed, and — critically — `_build_grouped_batches`, transcribed verbatim
from the winning implementation's journal diff. Every objective sees identical batches in
identical order for a given seed.

Varied: `FM.step` (pointwise BCE) vs `step_pairwise` (within-group BPR over all
positive×negative pairs) vs `step_listwise` (within-group softmax CE — the winner).

The pairwise arm deliberately mirrors the winner's scaffolding — same eligible groups
(mixed-label only), same normalisation discipline — rather than the historical BPR runs,
so any difference is attributable to the loss and not to sampling or scaling.

## Result

| Objective | mean primary | min | max | vs pointwise (paired) | vs official baseline |
|---|---:|---:|---:|---:|---:|
| pointwise | 0.599047 | 0.598020 | 0.600190 | — | **−0.00239** |
| pairwise | 0.602633 | 0.602546 | 0.602745 | **+0.00359** (3/3 up) | **+0.00119** |
| listwise | 0.603059 | 0.602730 | 0.603445 | **+0.00401** (3/3 up) | **+0.00162** |

Official pointwise baseline = 0.601440 (`constraints.md` C8, 3 matched seeds, random
row batching). Accept bar 0.0014; convergence epsilon 0.002; reference seed noise 0.0008.

**Validity check.** The listwise arm lands at +0.00162 against the official baseline —
reproducing the banked submission's independently hand-verified +0.00162 (`RESULTS.md` §2)
to five decimals. The reconstruction is measuring the right thing.

## What it says

**1. A ranking objective is real, and large.** Against pointwise on identical batches,
listwise gains +0.00401 and pairwise +0.00359, both with every seed up — roughly 3x the
accept bar. This is not noise.

**2. "Listwise" is not the active ingredient.** Listwise beats pairwise by **+0.00043**,
worst seed **−0.00001**, only 2/3 seeds up. That is inside the seed reference (0.0008) and
far below the accept bar (0.0014). The two are indistinguishable here. The gain belongs to
*not being pointwise*, not to the listwise formulation specifically.

**3. The seven "failed listwise" runs did not fail because they were listwise**, and the
five "failed BPR" runs did not fail because they were pairwise. Both objectives work under
this scaffolding. Those runs failed on implementation.

**4. The batching is eating most of the gain.** The grouping that ranking losses require
costs −0.00239 on its own, measured with the loss held pointwise. So:

```
official pointwise baseline                    0.601440
  + user-grouped batching (loss unchanged)     -0.00239
  + ranking objective instead of pointwise     +0.00401
  = the banked result                          +0.00162
```

The shipped +0.00162 is a small residual of a large positive and a large negative.

## What to do next

Recovering the −0.00239 is worth more than any further ranking-loss variant: it is 1.5x
the size of the entire banked gain, and the ablation says the objective side is already
close to saturated (pairwise and listwise agree within noise).

Registered as direction `grouped_batching_cost`. Untested candidates: a larger
`target_bs` so each batch spans more users; restoring a pointwise term on ungrouped rows
alongside the ranking term; finer-grained shuffling; and revisiting early stopping — the
grouped pointwise arm ran to epochs 8/11/10 while listwise stopped at 3/4/4, so the
grouped arms may simply be converging differently rather than worse.

## What it does not establish

Nothing about the test split. Nothing about other model families — this is the official
five-field FM throughout. It does not identify *which* of the winner's four distinguishing
traits (pure softmax vs mixed BCE, mixed-label-only groups, normalisation by total
positives, uncapped lists) matters, only that the choice of ranking loss among the two
tested is not what separates it from pointwise. Three seeds is enough to see a +0.004
effect cleanly; it is not enough to resolve the +0.00043 listwise-over-pairwise gap, which
would need substantially more seeds to call either way.
