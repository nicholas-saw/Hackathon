# Results — KuaiRand-Pure autonomous ML research agent

> Deliverable summary. Numbers are validation-only; the hidden test set was never scored
> during development. Provisional pending the final run in progress at 2026-08-31 08:40
> MPST — the banked submission below stands regardless.

## 1. Final submission

| | |
|---|---|
| File | `submissions/verified_listwise_3seed_ensemble.csv` |
| Rows | 170,588 — `row_id` contract validated by the frozen `kit/submit.py` checker |
| Method | Within-user **listwise softmax** objective replacing pointwise BCE, rank-averaged over 3 seeds |
| Config | `{"model": "fm_listwise", "k": 16, "lr": 0.001, "epochs": 40, "bs": 8192, "patience": 4}` |
| Provenance | Agent hypothesis (iteration 1 of run `20260830T235541Z`), verified on matched seeds |

## 2. Results table

Validation-best, against the official published baseline.

| Metric | Official baseline (valid) | This submission (valid) | Absolute delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | **0.670511** | **+0.003111** |
| nDCG@5 | 0.5357 | **0.537590** | **+0.001890** |
| **primary** | **0.6016** | **0.604051** | **+0.002451** |

Scoring formula per the problem statement: `delta(m) = score_agent(m) - score_baseline(m)`,
averaged over metrics — **+0.00250** on validation.

### Matched-seed verification of the underlying method

The single-seed result was verified by re-measuring baseline *and* method on the same
three seeds, because a single seed cannot distinguish a real gain from seed luck.

| Seed | Baseline | Listwise | Paired delta |
|---|---:|---:|---:|
| 0 | 0.601470 | 0.603445 | +0.00197 |
| 1 | 0.601761 | 0.602730 | +0.00097 |
| 2 | 0.601090 | 0.603003 | +0.00191 |
| **mean** | **0.601440** | **0.603059** | **+0.00162** (sd 0.00046) |

All three seeds positive; ~3.5 sigma; worst seed still +0.00097. The honest expectation
for the method is **+0.00162**, not the +0.00197 that seed 0 alone suggested — the
submitted artifact is the 3-seed rank average, which scores higher than any single seed
through variance reduction.

**Sanity check:** the measured 3-seed baseline mean of 0.601440 matches the independently
audited figure in `context/constraints.md` C8 (`0.601440 ± 0.000275`) to six decimals,
confirming the pipeline reproduces the official baseline exactly.

## 3. Resource consumption

Across 9 completed measured runs (`runlogs/run_*/journal.jsonl`):

| | |
|---|---:|
| LLM calls | 62 |
| Input tokens | 265,539 |
| Output tokens | 355,325 |
| Cache write / read | 327,072 / 1,708,228 |
| **Total tokens** | **2,656,164** |
| **Total cost** | **$10.41** |
| Agent wall-clock | 6,765 s (1 h 53 m) |
| GPU-hours | 0.0 (CPU only) |
| Iterations | 3 per run typical; convergence (eps=0.002, N=3) fired before the 50 cap in every case |

Models: proposer `claude-opus-5`, coder and reflector `claude-sonnet-5`. Prompt caching
with a 1-hour TTL; `cache_working: true` on every run.

## 4. What the agent found

Explored across runs, all as its own hypotheses from the evidence packet:

- **Within-user listwise softmax** — the winner. GAUC carried the gain (+0.0031) while
  nDCG@5 moved less (+0.0019), exactly the mechanism predicted for an objective aligned
  to within-user ranking.
- Listwise with fixed group sizes — smaller gain (+0.00062).
- BCE + listwise hybrid — negative (−0.00289).
- Pairwise BPR, and BPR blended with BCE — negative.
- Strictly-prior history aggregates (user-level, item-level, user-author affinity) via the
  causal `harness.history` helper — negative (−0.0011 to −0.0012).

The organizers rank loss/objective mismatch as the most promising direction; the agent
reached the same conclusion from the evidence and it is the one that paid.

## 5. Reproduction

```bash
set PYTHONUTF8=1
python -m harness.cache                    # build the encoded cache (~22s)
python context/build_packet.py             # regenerate the research packet
python tests/test_harness.py               # 40 invariants
python -m agent.controller --iterations 50 --budget 14
python verify.py --chain --order runlogs/<run_id>/journal.jsonl
python report.py runlogs/<run_id>/journal.jsonl -o report.html
```

Requires `ANTHROPIC_API_KEY` and, for identity-linked keys, `ANTHROPIC_WORKSPACE_ID`.

## 6. Honest limitations

- **The gain is small.** +0.0025 validation primary against a 0.8484 oracle ceiling. The
  baseline already captures ~31% of the attainable range; this adds roughly one further
  percent of it.
- **Validation-to-test transfer is unverified**, by design — test is scored once. Expect
  the test delta to be smaller than the validation delta: the candidate was selected as
  the validation maximum across several, which inflates the selected value.
- **Convergence bounds the search.** Three consecutive non-improving iterations end a run,
  so each run explored 3–5 hypotheses rather than 50. This is the organizers' rule applied
  literally, not a defect.
- **Most iterations across the project were lost to harness defects**, not to bad science —
  see `BUGS.md`. Sixteen were found and fixed, several of which had been silently
  producing false negatives.
