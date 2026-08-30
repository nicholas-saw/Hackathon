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

## Status

Built and verified: cache, adapter, guards, score, executor, journal, submission,
run_node, all three roles, controller, report, verify, tests, research packet.

Not done: the measured autonomous run (needs `ANTHROPIC_API_KEY`), and the Devpost
writeup. The floor submission is banked at validation primary 0.60274.
