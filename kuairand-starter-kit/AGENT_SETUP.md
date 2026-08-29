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
to be edited**, and everything the agent needs to do real ML work lives inside them.

## The file split

```
data.py        LOCKED   raw CSV loading + official train/valid/test date split
evaluate.py    LOCKED   the metric (GAUC / nDCG@5) — this IS the scoreboard
submit.py      LOCKED   submission file format writer/checker
                          ↓ both import from ↓
features.py    EDITABLE  feature engineering (what goes into the model)
model.py       EDITABLE  model architecture + loss function
train.py       EDITABLE  training loop, batching, early stopping, CLI
```

`data.py` reads the four KuaiRand CSVs and hands back rows tagged into
`train`/`valid`/`test` by date, plus an `IDX` dict so code can say `x[IDX['tab']]`
instead of a magic number. That's it — no feature engineering happens there anymore,
so the agent never has a reason to open it.

`features.py` turns those raw rows into the numeric arrays the model trains on
(categorical IDs, buckets, whatever the agent wants to try). `model.py` holds the
FM model and its loss. `train.py` runs the training loop and exposes the CLI:

```bash
python train.py --model fm
```

## The rule set (`AGENT_RULES.md`)

This is what you'd hand to the agent as its system prompt / operating instructions
for this repo. Highlights:

- **Only `features.py`/`model.py`/`train.py` are editable.** Every other file gets a
  one-line reason in a table (e.g. "editing `evaluate.py` invalidates every score it
  produces"). If the agent thinks a locked file has a real bug, the rule is: stop and
  report it, don't patch it yourself.
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
  modeling, multi-task learning, ...) so it has somewhere useful to start.

## The one thing that needed fixing in the code, not just the rules

Writing "don't look at the test score during development" as a rule isn't enough if
the code hands you the test score every time you run it — an agent (or a human, for
that matter) will glance at it out of habit. So `train.py` now defaults to computing
**only** the `valid` score:

```bash
python train.py --model fm            # valid only — use this while iterating
python train.py --model fm --final    # valid + test — use once, for the real number
```

This is enforced in `run_pop`/`run_random`/`run_fm` themselves (a `report_test=False`
default), not just in the CLI printing — so there's no way to accidentally see `test`
without explicitly asking for it.

## A second leakage class: post-impression columns

The `test`-split leakage rules above aren't the only way to cheat. `data.py` also
exposes columns like `is_click`, `is_like`, `play_time_ms`, etc. — outcomes that happen
*at the same time* as `long_view`, for the same impression. Using one of these as an
input feature to predict `long_view` on that same row leaks the answer, even if you
never touch `valid`/`test` at all. They're only legitimate as (a) an auxiliary
multi-task *target* for that same row, or (b) an input built from a *different* row —
the user's past behavior (sequence modeling).

`data.LEAKY_COLUMNS` lists the 11 forbidden columns, and `features.py`'s `same_row()`
helper actively raises if one of them is used as a same-row input — this isn't just a
written rule, misusing it is a runtime error.

## Running it yourself

```bash
cd kuairand-starter-kit
python train.py --model random --final   # sanity check: test primary should be ≈0.475
python train.py --model pop              # trivial baseline, valid only
python train.py --model fm               # the model you're trying to beat, valid only
python train.py --model fm --final       # same, plus the final test number
```

To produce and check an actual submission file (see `README.md` for the full format):

```bash
python submit.py --make  --split test  submission.csv
python submit.py --check --split test  submission.csv
```

## What to read next

- `AGENT_RULES.md` — the actual rule set, meant to be given to the agent.
- `README.md` — the task definition, baseline scores, and submission format (in Chinese).
- `features.py` / `model.py` / `train.py` — read the docstring at the top of each; they
  restate their own contract and what's off-limits.
