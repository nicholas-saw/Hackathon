# Results — KuaiRand-Pure autonomous ML research agent

> Every number here is validation-only. The hidden test split was never scored during
> development, and the ordering is checkable: `FINAL_DESIGNATION` precedes any `TEST_OPEN`
> in the hash-chained journal, and `verify.py` enforces it.
>
> All figures come from `runlogs/run_20260831T011354Z/journal.jsonl` unless stated.

## 1. Final submission

| | |
|---|---|
| File | `submissions/FINAL_submission_agent_best.csv` |
| SHA-256 | `475d168d1865c5dc815964d00243ff989aaf10549b61bfc38275161adbc7fe10` |
| Rows | 170,588 — `row_id` contract validated by the frozen `kit/submit.py` checker |
| Provenance | Byte-identical to the `FINAL_DESIGNATION` of run `20260831T011354Z`. The agent designated it; no human chose it. |
| Method | Strictly-prior user-level behavioural aggregates as bucketized FM fields, predictions rank-averaged over 5 independently-seeded models |
| Config | `{"model": "fm", "n_models": 5}` on top of the kept feature change |

The pipeline that produced this is reconstructable from the journal — the controller
restores a pristine pipeline when a run ends, so the winning code lives in the run record
rather than the working tree:

```bash
python replay.py runlogs/run_20260831T011354Z/journal.jsonl -o patches/
git apply --directory=kuairand-starter-kit patches/iter004.patch
git apply --directory=kuairand-starter-kit patches/iter012.patch
```

## 2. Results table

Validation-best at convergence, against the official published baseline.

| Metric | Official baseline | This submission | Absolute delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | **0.675137** | **+0.007737** |
| nDCG@5 | 0.5357 | **0.539348** | **+0.003648** |
| primary | 0.6016 | **0.607243** | +0.005643 |

The judged quantity is `delta(m) = score_agent(m) − score_baseline(m)` averaged over
`m ∈ {GAUC, nDCG@5}`:

**mean-of-deltas = (+0.007737 + 0.003648) / 2 = +0.005693**

(That is 0.00005 above the delta in `primary` only because the organizer rounds
(0.6674 + 0.5357)/2 = 0.60155 to 0.6016. The two are otherwise algebraically identical.)

Against the validation oracle ceiling of 0.8484, the baseline captures ~31% of the range
above random; this adds roughly a further 2.3% of what is attainable.

### Requirement 1 — reproducing the official baseline

Iteration 0 of every measured run re-runs the untouched baseline. It is **bit-identical
across all 14 measured runs**:

| Metric | Official | Reproduced | Delta | vs published seed sd (0.0008) |
|---|---:|---:|---:|---:|
| GAUC | 0.6674 | 0.667133 | −0.000267 | −0.33 σ |
| nDCG@5 | 0.5357 | 0.535806 | +0.000106 | +0.13 σ |
| primary | 0.6016 | 0.601470 | −0.000130 | −0.16 σ |

Every metric lands inside a third of one seed standard deviation, from 14 independent
journalled runs. That is the reproduction claim.

### Why this candidate, and not the largest number available

Selecting the best of 14 runs on a metric whose seed noise is ±0.0008 is exactly the
winner's curse that `harness/selection.py` exists to prevent, so the choice needs an
argument rather than an argmax:

- **Both contributing steps were confirmed on matched seeds, before selection.** Iteration
  4 measured a paired mean of **+0.00281 with a worst seed of +0.00236**; iteration 12
  measured **+0.00112 with a worst seed of +0.00090**. In both, every seed moved the same
  way — not a lucky draw.
- **The margin dwarfs the selection inflation.** The gap to the runner-up candidate is
  0.0032, about 4.0 σ against the published 0.0008 seed sd, while best-of-14 inflation on
  that noise scale is ~1.7 σ.
- **It is the agent's own designation.** Shipping it means the submitted bytes are what
  the run produced, with no human selection step in between (see `INTERVENTIONS.md` §6).

**Runner-up, retained but not submitted.** A within-user listwise softmax measured
+0.00162 mean over three matched seeds (worst +0.00097) and was hand-rebuilt as a 3-seed
ensemble at validation primary 0.604051, mean-of-deltas +0.002501
(`submissions/verified_listwise_3seed_ensemble.csv`). It is a real effect and a good
result. It is not the submission, for two reasons: it scores 2.3× lower, and its bytes
match no `FINAL_DESIGNATION` in any journal because it was assembled by hand — which
would make the delivered artifact a human selection rather than an agent one.

## 3. Resource consumption

The scored run — the one that reached the submitted result. These are the two figures
Feasibility is graded on.

| | |
|---|---:|
| LLM calls | 35 |
| Input tokens | 221,161 |
| Output tokens | 266,693 |
| Cache write / read | 33,007 / 1,122,238 |
| **Total tokens** | **1,643,099** |
| **Agent wall-clock to convergence** | **6,259.6 s (1 h 44 m)** |
| Cost | $8.66 |
| GPU-hours | 0.0 (CPU only) |
| Iterations against the 50 cap | 12 attempted, 6 scored |
| Stop reason | `converged` (eps = 0.002, N = 3) |

Per role: proposer 13 calls / $2.69 · coder 15 calls / $5.60 · reflector 7 calls / $0.37.
Models: proposer `claude-opus-5`; coder and reflector `claude-opus-4-8`. Prompt caching
with a 1-hour TTL; `cache_working: true`.

> `iterations_used` in the journal payload reports **scored nodes** (6), not iterations
> consumed against the cap (12). Six attempts were lost to a node crash, three static-guard
> rejections and two coder role failures — see `INTERVENTIONS.md` §5. The honest figure
> against the 50-iteration cap is 12.

**Whole campaign, for completeness.** 14 measured runs, of which 10 reached a
designation: **$22.13**. Four aborted runs record no resources at all (they are written
only inside `FINAL_DESIGNATION`), so the campaign total is a lower bound. The single
scored run above is the figure that corresponds to the submitted result.

## 4. What the agent found

All hypotheses below were the agent's own, formed from the evidence packet. Deltas are
against the running parent, from the scored run's journal.

| Iteration | Hypothesis | Result |
|--:|---|---:|
| 1 | Replace pointwise logloss with a within-user listwise softmax | REVERT −0.00233 |
| 2 | Hybrid objective: pointwise BCE + weighted listwise softmax | failed to run |
| 3 | Add a within-user pairwise (BPR) term on top of BCE | REVERT −0.00139 |
| 4 | **Strictly-prior user-level behavioural aggregates as bucketized FM fields** | **KEEP +0.00281** |
| 5 | Extend the same idea to item-side and user×item aggregates | REVERT −0.00320 |
| 6 | Add a user×tab conditional prior long_view rate | REVERT −0.00333 |
| 12 | **Rank-average 5 independently-seeded FMs on the kept feature set** | **KEEP +0.00112** |

The shape of that trajectory is the point.

The organizers rank objective/loss mismatch as the most promising direction. The agent
tried it **first and three times** — listwise softmax, a BCE-listwise hybrid, and a BPR
term — and **refuted it on this dataset**, at −0.00233 and −0.00139. It then proposed
something the packet did not point at: strictly-prior behavioural aggregates, routed
through the causal `harness.history` helper so no feature can see the row it predicts.
That is the change that paid.

It then did the thing that distinguishes research from search: it probed the boundary of
its own finding. Iterations 5 and 6 extended the winning idea to item-side, user×item and
user×tab aggregates, and both came back clearly negative. The effect is specific to
user-level priors, and the agent established that by trying to break it.

Only after the feature direction was exhausted did it consolidate with variance reduction.

Two caveats worth stating. Iteration 12's contribution is ensembling — a real and
permitted gain, but variance reduction rather than a modelling insight, and the journal
records it as such. And the listwise objective is not universally refuted: it measured
+0.00162 on matched seeds from the *unmodified* baseline in a different run. What the
scored run establishes is that it does not help *on top of* the prior-aggregate features.

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
[`runlogs/run_20260831T011354Z/report.html`](runlogs/run_20260831T011354Z/report.html).

## 6. Honest limitations

- **The gain is small.** +0.0057 mean-of-deltas against a 0.8484 oracle ceiling.
- **Validation-to-test transfer is unverified**, by design — test is scored once. Expect
  the test delta to be smaller than the validation delta.
- **Convergence bounds the search, not the cap.** Three consecutive non-improving
  iterations end a run, so runs explored 3–12 hypotheses, never 50.
- **Nothing carries between runs.** `harness/knowledge.py` records refuted directions
  within a run, but the registry is not persisted, so separate runs re-proposed and
  re-refuted the same listwise hypothesis repeatedly. This is the largest single waste in
  the project.
- **Six of twelve iterations in the scored run were lost, and none was recovered.** The
  retry paths that hand a guard finding or a traceback back to the coder are implemented
  (`BUGS.md` F17/F18) but were written after this run and have not yet been exercised in a
  measured run.
- **Most iterations lost across the project went to harness defects, not bad science.**
  Nineteen are written up in `BUGS.md`, several of which were silently producing false
  negatives.
- **A human chose the model configuration between runs.** Not inside one. The full
  accounting is in `INTERVENTIONS.md`.
