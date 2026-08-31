# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An agent that runs the machine-learning research loop by
itself: it reads the evidence, forms its own hypotheses, writes the code, runs the
experiment, decides what the result means, and stops when it has converged — with a
hash-chained journal that lets anyone check every decision after the fact.

**Result.** Starting from the organizer's Factorization Machine baseline, the agent
reached **validation primary 0.607243** in a single unattended 1 h 44 m run, with zero
human interventions between `RUN_START` and `FINAL_DESIGNATION`.

| Metric | Official baseline | Agent | Absolute delta |
|---|---:|---:|---:|
| GAUC | 0.6674 | **0.675137** | **+0.007737** |
| nDCG@5 | 0.5357 | **0.539348** | **+0.003648** |
| **Rubric score** — mean over metrics of `score_agent(m) − score_baseline(m)` | | | **+0.005693** |

Validation only. The hidden test split was never scored during development, and
`verify.py` proves the ordering. The winning pipeline was reconstructed from its own
journal and re-measured: it reproduces to the last digit, and wins **5 of 5** disjoint
user folds against the baseline.

---

## What is actually here

The agent is not a prompt loop. It is a deterministic harness with three LLM roles inside
it, and almost everything that protects the result is plain code that costs no tokens.

```
kuairand-starter-kit/
  agent/         proposer · coder · reflector · controller · llm      the three LLM roles
  harness/       guards · selection · journal · executor · adapter    deterministic, 0 tokens
                 history · knowledge · diagnostics · submission
  pipeline/      features.py · model.py · train.py                    the agent's edit surface
  kit/           FROZEN organizer code — data, evaluate, submit       read-only
  context/       PROBLEM.md · constraints.md · references.md          the research packet
  runlogs/       run_*/journal.jsonl                                  hash-chained evidence
  submissions/   the designated CSV
```

The parts that matter most:

- **`harness/guards.py`** — static analysis that runs *before* generated code executes.
  It refused three separate attempts by the agent to score the test split during the
  scored run. The leak firewall is code, not a promise.
- **`harness/journal.py`** — every event is `sha256(prev_hash ‖ payload)`. Editing any
  past record breaks the chain, and `verify.py` checks it.
- **`harness/selection.py`** — the winner's-curse defence. Picking the best of N runs on a
  metric whose seed noise is ±0.0008 selects the luckiest draw, so candidates are
  stability-tested across 5 user-level folds instead of taken by argmax.
- **`agent/proposer.py`** — the agent proposes three hypotheses, ranks them and picks one,
  each with a mechanism, cited evidence and a falsifier it must commit to in advance.

## Where to look

| Document | What it answers |
|---|---|
| [`REPORT.md`](REPORT.md) | **The detailed written report** — problem, architecture, what the agent found, what went wrong |
| [`RESULTS.md`](kuairand-starter-kit/RESULTS.md) | The numbers, the submission, resource consumption |
| [`INTERVENTIONS.md`](kuairand-starter-kit/INTERVENTIONS.md) | Manual-intervention summary — how autonomous it really was |
| [`HARNESS.md`](kuairand-starter-kit/HARNESS.md) | Why the harness is built the way it is |
| [`BUGS.md`](kuairand-starter-kit/BUGS.md) | Every defect found by running the system, fixed and still open |
| [`context/PROBLEM.md`](kuairand-starter-kit/context/PROBLEM.md) | The authoritative task definition |
| `runlogs/run_20260831T011354Z/` | The scored run: journal, and `report.html` |

`Research/` holds three earlier, superseded exploration tracks. Nothing in the delivered
system depends on it; it is kept for provenance only.

---

## Setup

Python 3.9+ (developed on 3.14). Two third-party packages; everything else is standard
library.

```bash
pip install -r kuairand-starter-kit/requirements.txt     # numpy, anthropic
```

Put the dataset where the kit expects it — `kuairand-starter-kit/data/`, unpacked from
the official KuaiRand-Pure archive. Then:

```bash
# Windows consoles default to cp1252 and the kit prints UTF-8. Set this or you will get
# a UnicodeEncodeError from a script that otherwise succeeded.
set PYTHONUTF8=1
```

For a measured run you need an API key in a gitignored `.env` at the repository root:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_WORKSPACE_ID=wrkspc_...    # only for identity-linked keys
```

## Reproduce

```bash
cd kuairand-starter-kit

python -m harness.cache                  # build the encoded split cache (~22 s)
python tests/test_harness.py             # 42 invariants, no training, seconds
python -m harness.guards                 # static leak scan of the pipeline

python -m agent.controller --dry-run --iterations 3      # plumbing only, no LLM, ~10 min
python -m agent.controller --iterations 50 --budget 14   # a real run

python verify.py --chain --order runlogs/<run_id>/journal.jsonl
python report.py runlogs/<run_id>/journal.jsonl -o report.html
```

To rebuild the exact pipeline that produced the submitted result — the controller
restores a pristine pipeline when a run ends, so the winning code lives in the journal
rather than the working tree:

```bash
python replay.py runlogs/run_20260831T011354Z/journal.jsonl -o patches/
git apply --directory=kuairand-starter-kit patches/iter004.patch
git apply --directory=kuairand-starter-kit patches/iter012.patch
```

## Tools, APIs, libraries and data

| | |
|---|---|
| **Dataset** | KuaiRand-Pure (only). No external training data, per the rules. |
| **API** | Anthropic Messages API, via the official `anthropic` Python SDK. |
| **Models** | `claude-opus-5` (proposer), `claude-opus-4-8` (coder, reflector) in the scored run. Earlier runs used `claude-sonnet-5`. |
| **API features** | Prompt caching with a 1-hour TTL, adaptive thinking, per-role effort. |
| **Libraries** | `numpy` for the whole ML pipeline. No PyTorch, no GPU — the reference model is a Factorization Machine and CPU is sufficient. |
| **Frozen** | The organizer's `kit/` — `data.py`, `evaluate.py`, `submit.py` — used unmodified as the scoring authority. |
| **Everything else** | Python standard library: `hashlib`, `json`, `difflib`, `subprocess`, `csv`, `argparse`. |

The rules permit any open-source framework. We stayed on numpy deliberately: at ~28
minutes for 100 baseline iterations on one CPU core, compute was never the binding
constraint — the convergence rule was.

---

## Limitations, honestly

- **The gain is small in absolute terms.** +0.0057 against a validation oracle ceiling of
  0.8484. The baseline already captures ~31% of the attainable range above random; this
  adds roughly another 2%.
- **Validation-to-test transfer is unverified by design.** Test is scored once, at the
  end. Expect the test delta to be smaller than the validation delta — the candidate was
  selected as a validation maximum, and that inflates the selected value.
- **Convergence, not the iteration cap, is the binding constraint.** Three consecutive
  non-improving iterations end a run, so runs explored 3–13 hypotheses, never 50. This is
  the organizers' rule applied literally, not a defect — but it means the search is much
  shallower than the cap suggests.
- **Most iterations lost across the project were lost to harness defects, not to bad
  science.** 19 are written up in `BUGS.md`. Several were silently producing false
  negatives — one guard bug rejected *every* training-loop change for nine iterations.
- **The retry paths are new and unexercised.** Handing a guard finding or a traceback back
  to the coder is implemented but has not yet run in a measured run. No journal here yet
  contains a recovered iteration.
- **A human chose the model between runs.** The scored run's configuration was picked by a
  human after earlier runs; the hypotheses within it were not. `INTERVENTIONS.md` sets out
  exactly what was and was not touched.

### With more time

Cross-run memory is the biggest gap. `harness/knowledge.py` records refuted directions,
but nothing carries them between runs, so nine separate runs opened with the same
hypothesis and re-refuted it. Persisting the registry would turn 14 independent runs into
one cumulative search. After that: exercising the retry paths in a real run, and giving
the proposer a way to request a specific seed count.

## Team

| | |
|---|---|
| **Nick** (`nicholassjm2302@gmail.com`) | Harness engineering, live-run debugging and the defect campaign in `BUGS.md`, context package and evidence audit, measured-run operation. |
| **Zehao Zhou** (`zhouzehao1783174@gmail.com`) | System design and orchestration plan, selection and knowledge layers, guard and journal integrity work, requirements audit. |

Development was assisted by Claude Code; the agent itself calls the Anthropic API
directly, as described above.
