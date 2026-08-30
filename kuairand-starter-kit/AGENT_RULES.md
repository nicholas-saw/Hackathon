# Agent rules — KuaiRand-Pure

You are iterating on this repo to raise the **primary** metric (`(GAUC + nDCG@5) / 2`,
see `kit/evaluate.py`) above the FM baseline (test primary **0.5946**). These rules bound
what you're allowed to touch and how you're allowed to get the score up. They exist
because an agent that can edit its own scoring code, its own data split, or its own
submission format can "improve" the number without improving the model — that result
is worthless to whoever reads the leaderboard. Follow the letter and the spirit.

---

## 0a. Reconciliation — these clauses supersede anything below that conflicts

An audit found this document contradicting itself in four places. Where the text below
disagrees with this section, **this section wins**.

**R1 — raw CSV access.** Section 1 says "never open the raw CSVs directly"; section 5
item 2 says reading them from `features.py` is "within your editable surface". Both
cannot hold, because `log_standard_4_22_to_5_08_pure.csv` spans validation **and test**
and carries `long_view` — so the permissive reading hands you every test label.

Resolution: **never call `open()` on a dataset CSV.** Raw columns are available through

```python
from harness.adapter import raw_columns, auxiliary_targets, entity_table
cols = raw_columns(('hourmin', 'time_ms'))        # {'train': {...}, 'valid': {...}}
aux  = auxiliary_targets(('is_click',))           # multi-task targets, float32
vids = entity_table('video_stat')                 # static side table, keyed by id
```

which drops test-period rows during parsing and aligns positionally with
`kit.data.load()`. This keeps organizer directions 2, 3 and 4 reachable. The static guard
rejects a direct CSV read before your code runs.

**R2 — the join key is not a key.** Section 5 says to join raw columns on
`(user_id, video_id)`. `kit/submit.py` documents that pair as non-unique: 3.06% of
evaluation rows repeat it, up to 12 times. Joining on it fans out and mis-attributes
another impression's outcome onto the current row — a leak wearing a feature's clothes.
`harness.adapter` aligns by position instead. Do not hand-roll a join.

**R3 — the contract is wider than section 2 states.** `train.py` also imports `IDX` and
`FIELDS` from `features.py`, and the harness calls `fit_predict`. Preserve all of it:

```python
features.py:  encode(splits) -> (enc, dim);  enc[split] = (X, y, users)
              IDX     — the field order of kit.data.load()'s row tuple
              FIELDS  — NOT documentation. len(FIELDS) sizes vocabs and X's second axis.
                        Add a column to raw() and you MUST append its name here.
model.py:     FM(dim, k, lr, l2, seed).step(X, y) -> loss; .predict(X) -> ndarray; .V/.W/.b
train.py:     fit_predict(enc, dim, model=, seed=, **cfg)
                -> {'train': ndarray, 'valid': ndarray, 'test': ndarray}
```

`fit_predict` is how the harness builds a submission. `kit/submit.py --make` does **not**
use your pipeline — it rebuilds the untouched official baseline from `kit/`, so a
submission made that way contains none of your work.

**R4 — never run `kit/baseline.py`.** It has no `report_test` flag and prints test
metrics unconditionally. `kit/submit.py --score` is equally unsafe: its `--split`
defaults to `test`. Neither is needed; the harness scores validation for you.

`kit/` is now genuinely read-only at the filesystem level — a write raises
`PermissionError`. If you hit one, that is your signal to stop and report, not to work
around it.

---

## 0. Layout

```
kit/            FROZEN  — pristine, unmodified Starter Kit (vendor code)
  data.py         raw CSV loading + official train/valid/test date split
  evaluate.py     the metric (GAUC / nDCG@5) — this IS the scoreboard
  submit.py       submission file writer/checker
  baseline.py     the original reference FM (kept only for provenance/diffing)
  baseline_scores.json
pipeline/       EDITABLE — the only 3 files you may change
  features.py     feature engineering
  model.py        model architecture + loss
  train.py        training loop, batching, early stopping, CLI
ablation_features.py   locked, top-level — a past experiment's record, not part of the pipeline
```

`kit/*.py` and `kit/*.json` are **filesystem read-only** (Windows read-only attribute) —
this is not just a written rule, a plain `open(path, 'w')` against anything in `kit/`
raises `PermissionError` at the OS level. If you think you need to change one of these
files, that permission error is your signal to stop and report, not to work around it
(e.g. by clearing the attribute yourself, copying the file elsewhere, or `exec`-ing
around it). `pipeline/*.py` has no such restriction — that's the point.

## 1. You may only edit three files

- **`pipeline/features.py`** — feature engineering. Add fields, buckets, transforms,
  sequence features, multi-task label construction. Read only from `kit.data.load()`'s
  return value — never open the raw CSVs directly, never import anything from
  `kit/evaluate.py`. `kit/data.py` doesn't expose a schema registry (it's pristine
  vendor code), so `features.py` documents the row layout itself (`IDX`) — if you ever
  need to change how rows are laid out, that means changing `kit/data.py`, which you
  can't do; report it instead of duplicating/forking the loader.
- **`pipeline/model.py`** — model architecture and loss. Swap loss functions
  (pairwise/listwise), add model classes, change the FM's capacity or regularization.
- **`pipeline/train.py`** — training loop, batching, early stopping, CLI. Change how
  batches are built (e.g. grouped by user for listwise loss), how convergence is
  judged, how model selection works.

**Every other file is read-only infrastructure.** Do not edit, monkeypatch, `exec`,
or otherwise alter the behavior of any file not on this list — that includes editing
it "temporarily to debug" and reverting, since a leaderboard run doesn't know it was
temporary. If you believe one of them has an actual bug, stop and report it instead of
patching it yourself.

| File | Why it's locked |
|---|---|
| `kit/evaluate.py` | The official metric implementation. Editing it — even a "fix" — invalidates every score it produces. |
| `kit/data.py` | Owns the official train/valid/test date split (`SPLITS`) and the raw row schema. Changing split boundaries or row semantics can leak test-period data into training without it being obvious from the score alone. |
| `kit/submit.py` | Submission format writer/validator. Don't touch the row/`row_id` contract. |
| `kit/baseline.py` | The original reference FM implementation, kept for provenance — not the one you're iterating on (that's `pipeline/model.py` + `pipeline/train.py`). |
| `kit/baseline_scores.json` | Reference numbers used to self-check the harness (see README's `--model random --final` self-check). Not a config file. |
| `README.md` | Task spec. The task definition table is explicitly frozen ("口径已写死，不要改"). |
| `ablation_features.py` | A past experiment's record, not part of the active pipeline. |

## 2. Contracts you must preserve

`pipeline/train.py` wires the other two together — keep these signatures so that
swapping either module doesn't break the CLI:

```python
# pipeline/features.py
encode(splits) -> (enc, dim)
enc[name] = (X, y, users)   # X: int32 (N, F); y: float32 (N,); users: list[str]

# pipeline/model.py
FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)
    .step(X, y) -> loss: float
    .predict(X, bs=200_000) -> np.ndarray
    .V, .W, .b                 # read/write, used for best-checkpoint snapshotting
```

If a change (e.g. pairwise/listwise loss) needs a different batch shape than flat
`(X, y)` rows, adapt the batching in `train.py` and the corresponding `step()` in
`model.py` together — both are yours to edit, so keep them in sync rather than
smuggling the new logic into `kit/data.py` or `kit/evaluate.py`.

## 3. Hard rules against gaming the metric

- **Never read `test`-split labels for anything except the one final read-only report.**
  All feature construction (bucket edges, vocabs, aggregates) and all model selection
  must be computed from `train` (and `valid` for early stopping only).
- **Never special-case a `user_id` or `video_id`** based on its presence in `valid`/`test`
  (e.g. a lookup table keyed by exact IDs seen in the eval split). Interaction features
  must generalize the same way for unseen IDs (fall through to the UNK slot).
- **Never touch `SPLITS`, the date boundaries, or which CSV rows map to which split.**
- **Never call `evaluate()` on `test` during development iteration** — only on `valid`.
  Report `test` once, at the end, as the final number. `pipeline/train.py`'s
  `run_pop`/`run_random`/`run_fm` all default to `report_test=False` (valid only) for
  exactly this reason — `test` is only computed when `--final` is passed on the CLI.
  Don't add a shortcut that makes `test` visible by default; if you need it back for a
  real final report, use `--final`.
- Don't wrap, shadow, or import-hack `kit/evaluate.py`'s `evaluate`/`auc`/`ndcg_at_k` to
  change their behavior from inside `pipeline/`.
- **Never use a post-impression outcome column as a same-row input feature.** This is a
  different leakage class from the `test`-split rules above — it leaks even if you only
  ever touch `train` data, because these columns are concurrent outcomes of the *same*
  impression as `long_view`, not information available before the outcome. Forbidden as
  same-row inputs: `long_view` (`label`), `is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`, `is_hate`, `play_time_ms`, `profile_stay_time`, `comment_stay_time`,
  `is_profile_enter` — see `pipeline/features.py`'s `LEAKY_COLUMNS`. `kit/data.py`
  currently only loads `label` among these; the other 10 aren't wired into its row tuple
  at all (see §5, item 2, for what this means if you want them). They may only be used
  as (a) an auxiliary *target* for the same row (multi-task learning predicts them,
  doesn't consume them), or (b) an input feature aggregated from a *different* row (the
  user's past interactions — this is exactly what sequence modeling, headroom idea 2,
  needs). `features.py` enforces this: build same-row inputs through `same_row(x, name)`,
  which raises on any name in `LEAKY_COLUMNS` — don't bypass it with direct indexing.

## 4. Already ruled out — don't re-spend iterations here

From the README's "从哪里开始改" section (already measured, no gain):

- **Stuffing in more static feature domains** (CWM's 13 domains vs. the current 5) —
  primary 0.5940 vs 0.5950, within noise, slightly worse.
- **Raising FM embedding dim** (k = 8/16/32) — 0.5895/0.5902/0.5887, basically flat.
- **Pure user-side first-order features contribute exactly zero** — ranking is within-user,
  so anything constant within a user's group doesn't change the order. User-side signal
  only helps through **cross terms with item-side features**.

Bottleneck is not feature count or model capacity; `user_id × video_id` already
captures most of the learnable signal in this dataset size.

## 5. Where the headroom probably is

In the README's judged order of promise:

1. **Loss/objective mismatch** — training is pointwise logloss, but the metric is a
   ranking metric. Try pairwise (BPR) or listwise (softmax over one user's impressions).
   Considered most likely to help.
2. **Sequence modeling** — no behavioral history is used at all; DIN/SIM-style user
   interest modeling is unexplored. **Correction from an earlier version of this doc:**
   `kit/data.py` is pristine vendor code and does *not* expose `is_click`/`is_like`/
   `play_time_ms`/`hourmin`/etc. — it only loads the 5-field baseline row. Since
   `kit/data.py` can't be touched, getting access to those columns for sequence or
   multi-task features means `pipeline/features.py` reading the raw CSVs itself
   (`log_standard_*.csv`, joined on `user_id`/`video_id`), independent of `kit.data.load()`
   — that's within your editable surface, just extra work you should budget for.
3. **Multi-task** — auxiliary heads on `is_click`/`is_like`/`is_follow`/`is_comment`/
   `is_forward`/`play_time_ms` alongside the `long_view` main task. Same caveat as #2 —
   these columns need to be sourced independently in `features.py`.
4. **Censored watch-time regression** (CWM's angle) — advanced, treat as a stretch goal.
5. **Different model family** (DeepFM/DCN/xDeepFM) — lower priority than 1–3 since
   capacity isn't the bottleneck.
6. **Time features / train-test drift** — `hourmin`, `date` (`date` is already in the
   pristine row; `hourmin` would need the same independent CSV read as #2/#3).
7. **Unbiased validation** — `log_random_4_22_to_5_08_pure.csv` (not loaded by
   `kit/data.py`) as a sanity check against overfitting to biased exposure — this is a
   `kit/data.py`-shaped change (a new loader/split source), so flag it rather than hack
   it into `features.py`.

## 6. Process

- Convergence judgment (from README): std of FM primary across seeds is **0.0008** →
  use **ε = 0.002 (≈2.5σ), N = 3** — 3 consecutive iterations with validation primary
  improving by ≤0.002 counts as converged for that line of experimentation.
- One hypothesis per iteration. Run `python pipeline/train.py --model fm` (no `--final`,
  from the `kuairand-starter-kit/` directory), read the **valid** primary, decide
  keep/revert before moving on.
- Before trying a change, know what you'd revert to — commit or snapshot working
  states of `pipeline/features.py`/`model.py`/`train.py` so a regression is a fast
  rollback, not a rewrite.
- Self-check the harness itself is intact by re-running
  `python pipeline/train.py --model random --final` occasionally — test primary should
  stay ≈0.475 (±0.001) as in the README (`--final` is required here since this is a
  harness sanity check, not a feature/model iteration). If it drifts, something in the
  locked files changed, or `kit/`'s read-only protection was bypassed — stop and report,
  don't patch around it.
- Report scores in the same `GAUC | nDCG@5 | primary` format `train.py` already prints.
