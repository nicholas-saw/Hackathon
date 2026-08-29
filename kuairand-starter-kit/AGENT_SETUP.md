# Agentic setup — what this is and how it works

You're building an agent that iterates on the KuaiRand-Pure recommendation task
(see `README.md` for the task itself) to beat the FM baseline score. This doc
explains the guardrails put in place so the agent can experiment freely without
being able to cheat its own scoreboard. It's for you, the human — the agent-facing
rules live in `AGENT_RULES.md`.

## The problem this solves

An agent that's told "raise this score" and given write access to the whole repo
will, sooner or later, find the shortest path to a higher number — which might mean
editing the scoring code, peeking at test labels, or loosening the submission
format, rather than actually improving the model. None of that is useful to you.

The fix is architectural, not just a written promise: **only three files are meant
to be edited**, and everything the agent needs to do real ML work lives inside them —
and that boundary is enforced at the filesystem level, not just in a document an
agent could ignore.

## The file split

```
kit/            FROZEN, filesystem read-only — pristine, unmodified Starter Kit
  data.py         raw CSV loading + official train/valid/test date split
  evaluate.py     the metric (GAUC / nDCG@5) — this IS the scoreboard
  submit.py       submission file writer/checker
  baseline.py     the original reference FM (kept for provenance, not iterated on)
  baseline_scores.json
                    ↓ pipeline/ imports the frozen pieces it needs from kit/ ↓
pipeline/       EDITABLE — the only 3 files the agent may change
  features.py     feature engineering (what goes into the model)
  model.py        model architecture + loss function
  train.py        training loop, batching, early stopping, CLI
```

`kit/data.py` reads the KuaiRand CSVs and hands back rows tagged into
`train`/`valid`/`test` by date — nothing else. It's the original vendor code,
untouched. `pipeline/features.py` documents that row's layout itself (an `IDX` dict)
since `kit/data.py` can't be edited to expose one — this is a one-time redundancy,
not a maintenance burden, since `kit/data.py`'s row format is frozen by definition.

`pipeline/features.py` turns those raw rows into the numeric arrays the model trains
on. `pipeline/model.py` holds the FM model and its loss. `pipeline/train.py` runs the
training loop and exposes the CLI:

```bash
python pipeline/train.py --model fm
```

## Enforcement is physical, not just written

Every file in `kit/` has the Windows read-only attribute set. This isn't a metaphor —
try it:

```bash
python -c "open('kit/data.py', 'a').write('x')"
# PermissionError: [Errno 13] Permission denied: 'kit/data.py'
```

A coding agent that tries to `open(path, 'w')` on anything in `kit/` gets a real
error, not a polite request it can route around. Reads are unaffected — everything
in `pipeline/` still imports from `kit/` normally at runtime (via a small `sys.path`
shim in `pipeline/train.py`, since `kit/` and `pipeline/` are sibling directories).
If a locked file genuinely needs to change, the fix is a deliberate human action
(clearing the attribute), not something that happens as a side effect of an agent's
edit — see `AGENT_RULES.md` §0.

## The rule set (`AGENT_RULES.md`)

This is what you'd hand to the agent as its system prompt / operating instructions
for this repo. Highlights:

- **Only the 3 files in `pipeline/` are editable.** Every other file gets a one-line
  reason in a table (e.g. "editing `kit/evaluate.py` invalidates every score it
  produces"). If the agent thinks a locked file has a real bug, the rule is: stop and
  report it, don't patch it yourself — and now, don't-patch-it-yourself is backed by
  an actual permission error, not just discipline.
- **Function contracts are spelled out** (`encode(splits) -> (enc, dim)`, `FM.step()`,
  `FM.predict()`, ...) so the agent knows what it must preserve when it swaps in a new
  loss function or feature set — the three files still have to work together.
- **Anti-gaming rules**, the ones that matter most for an autonomous loop:
  - Never build features or pick a model checkpoint using `test`-split data — `train`
    (and `valid` for early stopping) only.
  - Never hardcode behavior for specific `user_id`/`video_id` values seen in eval splits.
  - Never touch the date boundaries that define the splits.
  - Never call the scoring function on `test` except once, at the very end.
- **A "don't bother" list** — static feature stuffing and bigger embeddings were already
  tried and didn't help (documented in the competition's own README), so the agent
  won't burn iterations repeating them.
- **A ranked list of what's actually unexplored** (pairwise/listwise loss, sequence
  modeling, multi-task learning, ...) so it has somewhere useful to start — with a
  note that some of these (sequence/multi-task features) need `pipeline/features.py`
  to read the raw CSVs itself, since `kit/data.py` only loads the 5-field baseline row.

## The one thing that needed fixing in the code, not just the rules

Writing "don't look at the test score during development" as a rule isn't enough if
the code hands you the test score every time you run it — an agent (or a human, for
that matter) will glance at it out of habit. So `pipeline/train.py` defaults to
computing **only** the `valid` score:

```bash
python pipeline/train.py --model fm            # valid only — use this while iterating
python pipeline/train.py --model fm --final    # valid + test — use once, for the real number
```

This is enforced in `run_pop`/`run_random`/`run_fm` themselves (a `report_test=False`
default), not just in the CLI printing — so there's no way to accidentally see `test`
without explicitly asking for it.

## A second leakage class: post-impression columns

The `test`-split leakage rules above aren't the only way to cheat. If you (or the
agent) later wire in columns like `is_click`, `is_like`, `play_time_ms`, etc. —
outcomes that happen *at the same time* as `long_view`, for the same impression —
using one of them as an input feature to predict `long_view` on that same row leaks
the answer, even if you never touch `valid`/`test` at all. They're only legitimate
as (a) an auxiliary multi-task *target* for that same row, or (b) an input built
from a *different* row — the user's past behavior (sequence modeling).

`pipeline/features.py`'s `LEAKY_COLUMNS` lists the 11 forbidden columns, and its
`same_row()` helper actively raises if one of them is used as a same-row input — this
isn't just a written rule, misusing it is a runtime error. Right now `kit/data.py`
only loads `label` (`long_view`) among these 11, so the guard is mostly future-proofing
for when `features.py` sources the other 10 itself.

## Running it yourself

```bash
cd kuairand-starter-kit
python pipeline/train.py --model random --final   # sanity check: test primary should be ≈0.475
python pipeline/train.py --model pop              # trivial baseline, valid only
python pipeline/train.py --model fm               # the model you're trying to beat, valid only
python pipeline/train.py --model fm --final       # same, plus the final test number
```

To produce and check an actual submission file (see `README.md` for the full format) —
`kit/submit.py` is fully self-contained within `kit/`, no cross-directory imports needed:

```bash
python kit/submit.py --make  --split test  submission.csv
python kit/submit.py --check --split test  submission.csv
```

## What to read next

- `AGENT_RULES.md` — the actual rule set, meant to be given to the agent.
- `README.md` — the task definition, baseline scores, and submission format (in Chinese).
- `pipeline/features.py` / `model.py` / `train.py` — read the docstring at the top of
  each; they restate their own contract and what's off-limits.
