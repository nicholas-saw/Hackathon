# Harness and agent — what was built, and how to run it

The starter kit shipped a working FM pipeline and a frozen scoring kit. This adds the
orchestration layer around it: the loop, the guards, the journal, the evidence.

## Layout

```
kit/          FROZEN — filesystem read-only. A write raises PermissionError.
pipeline/     the agent's editable surface: features.py, model.py, train.py
harness/      deterministic infrastructure. No LLM calls, no tokens.
  cache.py       parsed splits + encoded arrays, invalidated by features.py's hash
  adapter.py     the ONLY route to raw columns; drops test rows during parsing
  guards.py      static leak / edit-surface scan, run before any generated code executes
  score.py       wraps the frozen evaluator; test is sealed behind FINAL_DESIGNATION
  executor.py    subprocess + timeout + Windows process-tree kill, psutil-verified
  journal.py     hash-chained append-only JSONL
  submission.py  builds submissions THROUGH pipeline/, validates via the frozen checker
  run_node.py    one experiment, in its own process
  diagnostics.py whitelisted analyses the proposer can request before spending a node
  knowledge.py   persistent registry of directions: tried, refuted, provably no-op
agent/        three LLM roles, three deterministic ones
  llm.py         Anthropic client, prompt caching, token + cost meter, budget ceiling
  proposer.py    3 hypotheses, ranked, one chosen — a single call
  coder.py       whole-file changes; the harness computes the diff
  reflector.py   verdict against the prediction, plus a deterministic fallback
  controller.py  the loop: search policy, convergence, checkpointing, designation
context/      the research packet (built by context/build_packet.py)
report.py     journal.jsonl -> a self-contained report.html
verify.py     hash chain + FINAL_DESIGNATION precedes TEST_OPEN
tests/        fast invariants, no training
```

## Run it

```bash
set PYTHONUTF8=1                       # required: kit/submit.py crashes on cp1252

python -m harness.cache                # build the cache once (~28s cold, 0.43s warm)
python context/build_packet.py         # regenerate the research packet
python -m harness.submission --floor --seeds 5    # bank a floor submission

python tests/test_harness.py           # 14 invariants, seconds
python -m agent.controller --inject leak      # each of: leak syntax timeout nan
python -m agent.controller --dry-run --iterations 3   # no LLM, proves the plumbing

export ANTHROPIC_API_KEY=...           # required for a measured run
python -m agent.controller --iterations 50 --budget 14

python verify.py --chain --order runlogs/<run_id>/journal.jsonl
python report.py runlogs/<run_id>/journal.jsonl -o report.html
```

## Four defects this layer fixes

**1. `kit/submit.py --make` never touches `pipeline/`.** It imports `encode` from
`kit/data.py` and `FM` from `kit/baseline.py`, so a submission built that way contains
none of the agent's work — fifty iterations of improvement would score a 0.0000 delta.
`harness/submission.py` builds from `pipeline.train.fit_predict` instead and uses the
frozen checker only for validation. Verified: changing `lr` in the pipeline moves the
submission.

**2. `kit/` was not actually read-only.** Both `AGENT_RULES.md` and `AGENT_SETUP.md`
claimed a Windows read-only attribute enforced the boundary; `attrib` showed `A` on all
five files. It is set now, and a write raises `PermissionError`.

**3. Raw CSV access collided with test isolation.** `AGENT_RULES.md` forbids opening the
raw CSVs in section 1 and permits it in section 5 — and
`log_standard_4_22_to_5_08_pure.csv` spans validation *and* test and carries `long_view`.
`harness/adapter.py` resolves it: raw columns are available, test-period rows are dropped
during parsing. Verified: the newest date it will serve is 20220428.

**4. The documented join key is not a key.** `(user_id, video_id)` repeats on 3.06% of
evaluation rows, up to 12 times, so joining on it mis-attributes another impression's
outcome onto the current row. The adapter aligns positionally against `kit.data.load()`'s
exact file order and date filter.

`AGENT_RULES.md` section 0a records all four, and supersedes the text below it.

## Design decisions worth knowing

**Convergence is a hard stop.** The official rule scores "the validation-best checkpoint
at that point", so the score locks at convergence and running longer cannot raise it —
while wall-clock is scored under Feasibility. The controller stops.

**The accept bar is not epsilon.** Measured paired noise on validation is sigma ~ 0.0005
(the widely-quoted 0.0008 is the *test* seed std; five identity seeds on validation give
0.00032). A node joins the lineage only on a delta above 0.0014, roughly 2 sigma.

**Three hypotheses per iteration, one call.** Convergence fires after three consecutive
non-improving iterations, so each iteration should be the best of several ideas rather
than the first idea. This costs one request, not three.

**Only three roles spend tokens.** Controller, guards, executor, cache, scorer, journal
and the submission builder are plain code. `agent/llm.py` meters every call by role and
refuses a new node past the budget ceiling.

**Prompt caching is load-bearing.** The packet is ~6,300 tokens sent on every call. Cached
it is ~$9 for a 20-iteration run plus rehearsals; uncached the same run is ~$26, over
budget. `Meter.cache_working()` reports whether cache reads are actually happening.

## How the proposer is kept honest

Three gates, all enforced in `agent/proposer.py:validate()` before a node is spent:

**It cannot re-derive a closed direction.** `harness/knowledge.py` carries every
direction with a status, seeded with the organizers' two published dead ends (static
features 0.5940 vs 0.5950; capacity k=8/16/32 → 0.5895/0.5902/0.5887) and the two classes
that are provably zero under within-user ranking (pure user-side first-order terms, and
per-user monotone transforms). Proposing one is rejected unless the candidate sets
`extends_refuted` explaining what is different — so it is a speed bump, not a wall.

**It must predict above the resolution floor.** Paired validation noise is sigma ~ 0.0005,
so `expected_delta_primary` below 0.0015 is rejected: unmeasurable here even if real.
This kills hyperparameter tinkering, which is what a naive proposer reaches for first.

**It must cite evidence and a falsifier.** No `evidence` list or no `invalid_if`, no node.

And it can decline to experiment. `REQUEST_ANALYSIS` runs one whitelisted diagnostic and
feeds the result back without consuming an iteration — `no_op_screen` proves a column is
constant within users (and therefore useless) in about a second, versus a minute and one
of roughly ten iterations to discover the same thing by training.

## What makes a real improvement detectable

Three settings decide whether a genuine gain can be seen at all, and the first version got
all three wrong.

**Every node is a 3-seed rank-average.** A single-seed node carries sigma ~ 0.0006, which
is most of the 0.0014 accept bar, so a real +0.001 improvement was indistinguishable from
noise and got reverted. Three seeds cut the measurement noise to ~0.00035 and make the bar
mean something. It costs 3x wall-clock per node, which the 6-hour ceiling absorbs easily.

The count is one number, `self.seeds_per_node`, and it governs the **baseline as well as
every candidate**. That is not a detail: a rank-average scores systematically above any of
its members — measured here, 0.60219 over three seeds against 0.60147 over one — so
measuring candidates on fewer seeds than the baseline charges them ~0.0007 of pure
variance reduction as if it were a regression, half the accept bar. A dry run uses
`DRY_SEEDS_PER_NODE = 2` to stay quick, but it uses 2 for everything.

**Accept needs every paired seed to improve, not just the average.** Because the parent
carries its own per-seed primaries, `paired_confirmation()` compares seed *s* against
seed *s* — a matched test that removes the seed as a source of variance at no extra
training cost. A change is kept only if the shipped artifact clears the accept bar *and*
no individual seed regressed. Under the null, 3/3 paired seeds agreeing is p = 0.125 on
its own; with the bar on top, a noise iteration essentially never enters the lineage.

**The controller injects a free ensemble node every third iteration.** It trains nothing
and calls no model — it rank-averages the best distinct candidates so far — so it costs
zero tokens and about a second. Combining diverse candidates is the only move measured to
clear the noise floor reliably on this task, and waiting for the proposer to think of it
wasted iterations that the convergence rule does not have spare.

**Ensemble members are not filtered by individual strength.** What an ensemble buys is
decorrelation, not strength: on this dataset the listwise objective scores -0.0020 alone
and still adds +0.0012 in combination. The earlier gate only ensembled individually-stable
candidates, which removed exactly the diversity that makes it work. Now the ensemble is
built from the top few regardless and the *ensemble* is stability-tested. On the real
dry-run candidates this finds a stable +0.00055 where the old gate found nothing.

## Token economics, measured

Two paid calls against the live API established the real numbers, and two settings follow
from them:

- The packet is **10,192 tokens** and caches cleanly (call 1 wrote 10,187, call 2 read
  10,187 back).
- Role instructions are ~2.5k tokens, identical per role, and were sitting *outside* the
  cached prefix — paying full input price every call. They now have their own breakpoint.
- The cache TTL is **1 hour, not the 5-minute default**. Nodes take ~3 minutes, so
  consecutive calls for a role fall outside a 5-minute window and silently pay the write
  cost every iteration. Measured on the real prefix: rewriting each of 12 iterations is
  $0.76, versus $0.16 for one 1-hour write plus 12 reads.

Observed cost is ~$0.13 per proposer call after the first, dominated by output tokens
(2,900-4,300 per call, three candidates plus adaptive thinking).

## Why the run cannot ship a regression

No mechanism makes every hypothesis improve the score — a hypothesis that cannot fail is
not research. What `harness/selection.py` guarantees is the weaker, useful thing: the
*submitted* result never goes backwards.

The gap it closes is the winner's curse. `max(nodes, key=validation_primary)` does not
select the best model, it selects the luckiest draw; on a metric with paired sigma ~
0.0005, best-of-10 inflates the apparent score by roughly 1.5 sigma, and that inflation is
precisely what fails to transfer. Measured on this dataset: the validation-argmax of 21
statistically indistinguishable runs scored **below** the official baseline on test.

So a candidate is designated only if it beats the incumbent on the pooled score **and** on
4 of 5 independent user folds. Folds split by user, never by row, because the metric is
computed within users. Preference order is ensemble-of-stable > best stable single >
baseline, and a floor tripwire refuses to ship anything worse than the banked floor.

On the real dry-run candidates this changes the outcome: iteration 3 had a positive pooled
delta (+0.00069) but won only 3 of 5 folds, so it is rejected as a lucky draw, and the run
ships the floor at 0.60274 instead of iteration 3 at 0.60219.

Three other regression paths are closed upstream:

- **Fitting on the split you select on.** Bucket edges, vocabularies or target encodings
  computed from `splits['valid']` raise the validation score and lower the hidden-test
  score. The guard rejects it in `features.py` while still allowing `train.py` to use
  `enc['valid']` for early stopping, which is correct.
- **Sub-floor proposals.** A candidate predicting under 0.0015 is unmeasurable here and is
  rejected before it costs a node.
- **Re-deriving a closed direction.** The registry carries the published dead ends.

## Why there is no vector database

The corpus is a few dozen method cards plus at most fifty journal entries — under 10k
tokens, already riding in the cached system prompt at about $0.0005 a call. Embedding and
retrieving it would add a dependency, a build step and per-call latency to solve a recall
problem that does not exist at this scale.

It would also answer the wrong question. What the proposer needs is not "which directions
are *similar* to this one" but "is this one already closed" — that is set membership, and
similarity search returns near-misses precisely where an exact answer matters. So
`knowledge.py` is a structured registry with exact status lookup, not RAG. If the corpus
ever grew past a few hundred entries the trade would flip.

One cache detail worth preserving: the registry is passed in the **user message**, not the
cached system packet. It changes every iteration, and putting it in the prefix would
invalidate the prompt cache on every call and roughly triple the run's cost.

## Status

Built and verified: cache, adapter, guards, score, executor, journal, submission,
run_node, all three roles, controller, report, verify, tests, research packet.

Not done: the measured autonomous run (needs `ANTHROPIC_API_KEY`), and the Devpost
writeup. The floor submission is banked at validation primary 0.60274.
