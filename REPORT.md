# Building an autonomous ML research agent for KuaiRand-Pure

A written report for TikTok TechJam 2026, Track 2. The track allows a ~3-minute video or,
in its absence, a detailed report; this is the report. It is meant to be read on its own,
so it repeats the headline numbers rather than making you chase them.

**Result: mean-of-deltas +0.005693 on validation** (GAUC +0.007737, nDCG@5 +0.003648),
reached in a single unattended 1 h 44 m run with zero human interventions between
`RUN_START` and `FINAL_DESIGNATION`.

---

## 1. What the problem actually is

The surface description — "improve a recommender baseline" — hides the two constraints
that shaped every design decision.

**The metric barely moves.** The task is within-user ranking over logged impressions with
label `long_view`, scored as the mean of GAUC and nDCG@5. Users average about five
impressions each, and the metric only responds to within-user order. A user whose labels
are all 0 or all 1 has an nDCG pinned at 0 or 1 and is excluded from GAUC entirely — no
model can move them. That is why the validation oracle, scoring with the true labels, is
**0.8484 and not 1.0**, and why 42% of validation users are metric-invariant. The
organizer's FM baseline sits at 0.6016. The real headroom above random is 0.2468, and
published seed noise is ±0.0008. So a genuine improvement is a few thousandths, and it
lives inside three seed standard deviations of nothing at all.

**Convergence, not the iteration cap, is the budget.** The rule is: converged when
validation primary has not improved by more than ε = 0.002 over N = 3 consecutive
iterations, and *the score locks at convergence*. The 50-iteration cap is a red herring —
in practice a run gets three consecutive non-improving iterations and stops. Our runs
explored 3 to 12 hypotheses, never 50.

Put together: the agent gets on the order of ten attempts, each attempt must clear a bar
close to the measurement noise, and a wasted iteration costs one of the three that trigger
convergence. **Iterations are the scarce resource — not tokens, not compute.** That single
observation drove the architecture.

## 2. Architecture: put the judgement in the model, everything else in code

The system is a deterministic harness with three LLM roles inside it. Almost everything
that protects the result costs no tokens at all.

```
proposer  →  coder  →  [static guards]  →  executor  →  scorer  →  reflector
   ↑                                                                    │
   └──────────────── journal (hash-chained) ────────────────────────────┘
```

**Three roles, deliberately starved.** The *proposer* sees the research packet and the
journal digest, emits three hypotheses with mechanism, cited evidence and a falsifier
each, ranks them and picks one. The *coder* sees only the chosen hypothesis, the three
editable files and the rules — not the packet, not the other candidates, because that
judgement already happened upstream and re-sending it per node is most of the bill. The
*reflector* sees what was predicted before it is shown what happened, so the journal
records a prediction, then a miss, then a revised belief.

**Everything else is plain code.** The parts that decide whether a result is real:

- `harness/guards.py` — static analysis that runs *before* generated code executes. During
  the scored run it refused three separate attempts to call the evaluator on the test
  split. The leak firewall is code, not a promise.
- `harness/run_node.py` — blanks the hidden-test label array before the agent's pipeline is
  even imported. `fit_predict` needs test *features* to produce a submission vector and
  never needs test *labels*, so the array the agent could read is zeros. This is the
  difference between a firewall that is textually discouraged and one that is structurally
  impossible; see §5.
- `harness/journal.py` — every event is `sha256(prev_hash ‖ payload)`. Editing a past
  record breaks the chain. `verify.py` checks the chain *and* the ordering constraint that
  `FINAL_DESIGNATION` precedes any `TEST_OPEN`.
- `harness/selection.py` — the winner's-curse defence, discussed below.
- `harness/history.py` — a causal prior-aggregate helper. Any feature built from a user's
  past must exclude the row being predicted, including ties. Hand-rolled aggregates get
  this wrong in a way that inflates validation and evaporates on test.

**The measurement discipline.** Every node trains three seeds and rank-averages them, and
the parent carries its own per-seed scores forward, so a candidate is compared seed-to-seed
against its parent — a matched, paired test at no extra training cost. A change is kept
only if the shipped artifact clears the accept bar *and* no individual seed regressed.
Getting this wrong is not hypothetical: an earlier version measured candidates on one seed
against a three-seed baseline, which handed the baseline ~0.0007 of pure variance reduction
and charged it to every candidate as a regression — half the accept bar. Three ideas that
looked like losses were break-even once the comparison was matched.

## 3. What the agent found

All hypotheses were the agent's own, formed from an evidence packet that contains measured
facts and no ranked recommendations. Deltas are against the running parent.

| Iteration | Hypothesis | Result |
|--:|---|---:|
| 1 | Within-user listwise softmax replacing pointwise logloss | REVERT −0.00233 |
| 2 | Hybrid: pointwise BCE + weighted listwise softmax | failed to run |
| 3 | Within-user pairwise (BPR) term on top of BCE | REVERT −0.00139 |
| 4 | **Strictly-prior user-level behavioural aggregates as bucketized FM fields** | **KEEP +0.00281** |
| 5 | The same idea extended to item-side and user×item aggregates | REVERT −0.00320 |
| 6 | A user×tab conditional prior long_view rate | REVERT −0.00333 |
| 12 | **Rank-average 5 independently-seeded FMs on the kept feature set** | **KEEP +0.00112** |

The shape of that trajectory is the interesting part, and it is worth being precise about
what it shows.

The organizers rank objective/loss mismatch as the most promising direction — it is the
obvious move, given that the model trains pointwise and is scored on within-user order.
The agent tried it **first, and three times**: a listwise softmax, a BCE-listwise hybrid,
and a BPR term. It **refuted all three** on this dataset. Only then did it propose
something the packet did not point at — strictly-prior behavioural aggregates, routed
through the causal history helper — and that is the change that paid, at +0.00281 with
every one of three matched seeds positive.

Then it did the thing that separates research from search: it attacked the boundary of its
own finding. Iterations 5 and 6 extended the winning idea to item-side, user×item, and
user×tab aggregates. Both came back clearly negative, −0.00320 and −0.00333. The effect is
specific to *user-level* priors, and the agent established that by trying to break it
rather than by assuming it generalised.

Only when the feature direction was exhausted did it consolidate with variance reduction.

Two honest caveats. Iteration 12's contribution is ensembling — a real and permitted gain,
but variance reduction rather than a modelling insight, and the journal labels it as such.
And the listwise objective is not universally refuted: measured from the *unmodified*
baseline in a different run it was +0.00162 over three matched seeds, every seed positive.
What the scored run establishes is narrower and more interesting — that it does not help
*on top of* the prior-aggregate features.

## 4. Selection: why the biggest number is not automatically the answer

With seed noise at ±0.0008 and effects in the low thousandths, picking the maximum of N
candidates does not select the best model; it selects the luckiest draw. The project's own
earlier evidence is blunt about this: within one model family, the rank correlation between
validation and test was **0.000**, and the validation-argmax of 21 like-for-like runs was a
model *below* the official baseline on test.

So `harness/selection.py` does not take an argmax. Candidates are compared to the incumbent
across five user-level folds, and a candidate is designated only if it wins the pooled score
*and* at least four of five folds. If nothing is stable, the incumbent stays. A floor
tripwire refuses to ship anything worse than an already-banked submission.

For the delivered artifact the argument is explicit rather than assumed. Both contributing
steps were confirmed on matched seeds before selection — iteration 4 at mean +0.00281 with
worst seed +0.00236, iteration 12 at +0.00112 with worst seed +0.00090, every seed positive
in both. The gap to the runner-up candidate is ~4.0 σ against the published 0.0008 seed sd,
while best-of-14 inflation on that noise scale is ~1.7 σ. And it is the agent's own
designation: the submitted bytes are what the run produced, with no human selection step in
between.

## 5. What went wrong, and what it taught us

Nineteen defects are written up in `BUGS.md`. Almost all were invisible to reading and only
appeared under a live run. Four are worth repeating because each is a general lesson.

**A guard that rejects without explaining trains the agent to guess.** `guards.py` says
exactly that in its own docstring, every finding carries a `fix` field written for it, and
`coder.code()` has always accepted a `last_error`. None of it was wired: the controller's
guard branch was three lines and a `continue`. In the scored run the coder tripped the same
evaluate-on-test rule at iterations 7, 8 and 11 — writing the mirror image of the legitimate
`valid` line next to it — and was never told. Three nodes, and the delay of its own best
idea, for one correctable misunderstanding. The mechanism existed end-to-end and nothing
connected it.

**A bar calibrated for one measurement, applied to a mean.** The accept bar was 2 σ on a
single measurement, but every node is a three-seed mean whose standard error is σ/√3.
Holding the mean to the single-draw bar is ~3.5 σ. Iteration 12 measured +0.00112 with all
three seeds positive and was reverted out of the lineage for missing it — the highest
scoring node of the entire project.

**A regex is not a firewall.** The static guard matched the literal `enc['test'][1]`. It
does not match `Xt, yt, ut = enc['test']`, or `enc.get('test')[1]`, or a variable key. Nothing
leaked in fact, but the guarantee rested on spelling. The fix was structural: blank the test
labels in `run_node.py` before the agent's pipeline is imported. The pipeline needs test
features and never test labels, so the change costs nothing and converts a bypassable
pattern match into an impossibility.

**A stale fingerprint is worse than none.** `FROZEN.md` publishes SHA-256 hashes for the
frozen context package, and its own verification command failed on seven of eight files on a
Windows checkout — because git converts LF to CRLF and the hashes were over raw bytes. Every
byte delta equalled the file's CRLF count exactly; nothing had been altered. But an integrity
artifact that fails on untouched files discredits the claim it exists to support. Hashes are
now taken over normalised bytes, with a script that ships alongside them.

The through-line: **most of these were mechanisms that had been built and never connected**,
and every one was found by running the system rather than by reading it.

## 6. Autonomy, counted honestly

Impact & Relevance is scored primarily on manual interventions. The scored run,
`run_20260831T011354Z`, has **zero counted interventions** (classes L2–L5) between
`RUN_START` and `FINAL_DESIGNATION`: a hash-chained journal that verifies, no
`human_intervention` event, 1 h 44 m unattended, and six failures handled without a human.

We also state what that zero does *not* prove. `workspace_hash` is written once at
`RUN_START` and never re-verified at designation, and the intervention API has no production
caller — its only caller is a unit test. So the zero means "nothing was recorded", not "a
tamper-evident check found nothing".

At campaign level there were four counted interventions across 14 measured runs — one crash
restart and three runs killed by hand shortly after their baseline — plus a model
configuration chosen by a human *between* runs. The full ledger, including the four aborted
runs that record no spend, is in `INTERVENTIONS.md`. All of it is visible in the journals,
so disclosing it costs nothing and concealing it would cost a great deal.

## 7. Limitations and what we would do next

**Nothing carries between runs.** This is the biggest single waste. `harness/knowledge.py`
records refuted directions *within* a run, but the registry is not persisted, so nine
separate runs opened with the same listwise hypothesis and re-refuted it. Persisting it
would turn 14 independent searches into one cumulative one — and on a task that allows about
ten attempts per run, that is close to a tenfold increase in effective depth.

**The retry paths are new and unexercised.** Handing a guard finding or a traceback back to
the coder is implemented, but no journal in this repository yet contains a recovered
iteration. Until one does, the claim is code, not evidence.

**Diagnostics went unused.** The agent can request a cheap analysis instead of spending an
iteration on a training run. It never did, in any real run. An unused capability is not an
exhibit, and the proposer's instructions should push harder toward it.

**The coder re-emits whole files**, which is 65% of the token spend and the source of two
truncation failures. Patch-based editing would cut it, at some risk to the component whose
correctness matters most.

**Validation-to-test transfer is unverified by design.** Test is scored once. Given that the
candidate was selected as a validation maximum, the test delta should be expected to come in
smaller than +0.005693.

---

## Appendix: reproducing this

```bash
set PYTHONUTF8=1
cd kuairand-starter-kit
pip install -r requirements.txt

python -m harness.cache                  # encoded split cache (~22 s)
python context/verify_context.py         # frozen context unaltered
python tests/test_harness.py             # 42 invariants
python -m harness.guards                 # static leak scan

python -m agent.controller --iterations 50 --budget 14
python verify.py --chain --order runlogs/<run_id>/journal.jsonl
python report.py runlogs/<run_id>/journal.jsonl -o report.html
```

The winning pipeline is reconstructable from the run record, because the controller
restores a pristine pipeline when a run ends:

```bash
python replay.py runlogs/run_20260831T011354Z/journal.jsonl -o patches/
git apply --directory=kuairand-starter-kit patches/iter004.patch
git apply --directory=kuairand-starter-kit patches/iter012.patch
```

Full numbers in [`RESULTS.md`](kuairand-starter-kit/RESULTS.md); design rationale in
[`HARNESS.md`](kuairand-starter-kit/HARNESS.md); the defect record in
[`BUGS.md`](kuairand-starter-kit/BUGS.md); intervention accounting in
[`INTERVENTIONS.md`](kuairand-starter-kit/INTERVENTIONS.md).
