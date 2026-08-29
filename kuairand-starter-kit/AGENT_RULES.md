# Agent rules — KuaiRand-Pure

You are iterating on this repo to raise the **primary** metric (`(GAUC + nDCG@5) / 2`,
see `evaluate.py`) above the FM baseline (test primary **0.5946**). These rules bound
what you're allowed to touch and how you're allowed to get the score up. They exist
because an agent that can edit its own scoring code, its own data split, or its own
submission format can "improve" the number without improving the model — that result
is worthless to whoever reads the leaderboard. Follow the letter and the spirit.

## 1. You may only edit three files

- **`features.py`** — feature engineering. Add fields, buckets, transforms, sequence
  features, multi-task label construction. Read only from `data.load()`'s return value
  and `data.py`'s schema constants (`IDX`, `LABEL`, `SPLITS`) — never open the raw CSVs
  directly and never import anything from `evaluate.py`.
- **`model.py`** — model architecture and loss. Swap loss functions (pairwise/listwise),
  add model classes, change the FM's capacity or regularization.
- **`train.py`** — training loop, batching, early stopping, CLI. Change how batches are
  built (e.g. grouped by user for listwise loss), how convergence is judged, how model
  selection works.

**Every other file is read-only infrastructure.** Do not edit, monkeypatch, `exec`,
or otherwise alter the behavior of any file not on this list — that includes editing
it "temporarily to debug" and reverting, since a leaderboard run doesn't know it was
temporary. If you believe one of them has an actual bug, stop and report it instead of
patching it yourself.

| File | Why it's locked |
|---|---|
| `evaluate.py` | The official metric implementation. Editing it — even a "fix" — invalidates every score it produces. |
| `data.py` | Owns the official train/valid/test date split (`SPLITS`) and the raw row schema (`IDX`). Changing split boundaries or row semantics can leak test-period data into training without it being obvious from the score alone. |
| `submit.py` | Submission format writer/validator. Don't touch the row/`row_id` contract. |
| `baseline_scores.json` | Reference numbers used to self-check the harness (see README's `--model random` self-check). Not a config file. |
| `README.md` | Task spec. The task definition table is explicitly frozen ("口径已写死，不要改"). |
| `ablation_features.py` | A past experiment's record, not part of the active pipeline. |

## 2. Contracts you must preserve

`train.py` wires the other two together — keep these signatures so that swapping
either module doesn't break the CLI:

```python
# features.py
encode(splits) -> (enc, dim)
enc[name] = (X, y, users)   # X: int32 (N, F); y: float32 (N,); users: list[str]

# model.py
FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)
    .step(X, y) -> loss: float
    .predict(X, bs=200_000) -> np.ndarray
    .V, .W, .b                 # read/write, used for best-checkpoint snapshotting
```

If a change (e.g. pairwise/listwise loss) needs a different batch shape than flat
`(X, y)` rows, adapt the batching in `train.py` and the corresponding `step()` in
`model.py` together — both are yours to edit, so keep them in sync rather than
smuggling the new logic into `data.py` or `evaluate.py`.

## 3. Hard rules against gaming the metric

- **Never read `test`-split labels for anything except the one final read-only report.**
  All feature construction (bucket edges, vocabs, aggregates) and all model selection
  must be computed from `train` (and `valid` for early stopping only).
- **Never special-case a `user_id` or `video_id`** based on its presence in `valid`/`test`
  (e.g. a lookup table keyed by exact IDs seen in the eval split). Interaction features
  must generalize the same way for unseen IDs (fall through to the UNK slot).
- **Never touch `SPLITS`, the date boundaries, or which CSV rows map to which split.**
- **Never call `evaluate()` on `test` during development iteration** — only on `valid`.
  Report `test` once, at the end, as the final number. `train.py`'s `run_pop`/`run_random`/
  `run_fm` all default to `report_test=False` (valid only) for exactly this reason —
  `test` is only computed when `--final` is passed on the CLI. Don't add a shortcut that
  makes `test` visible by default; if you need it back for a real final report, use `--final`.
- Don't wrap, shadow, or import-hack `evaluate.py`'s `evaluate`/`auc`/`ndcg_at_k` to
  change their behavior from inside `features.py`/`model.py`/`train.py`.
- **Never use a post-impression outcome column as a same-row input feature.** This is a
  different leakage class from the `test`-split rules above — it leaks even if you only
  ever touch `train` data, because these columns are concurrent outcomes of the *same*
  impression as `long_view`, not information available before the outcome. Forbidden as
  same-row inputs: `long_view` (`label`), `is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`, `is_hate`, `play_time_ms`, `profile_stay_time`, `comment_stay_time`,
  `is_profile_enter` — see `data.LEAKY_COLUMNS`. They may only be used as (a) an auxiliary
  *target* for the same row (multi-task learning predicts them, doesn't consume them), or
  (b) an input feature aggregated from a *different* row (the user's past interactions —
  this is exactly what sequence modeling, headroom idea 2, needs). `features.py` enforces
  this: build same-row inputs through `same_row(x, name)`, which raises on any name in
  `LEAKY_COLUMNS` — don't bypass it with direct `x[IDX[...]]` indexing for the same row.

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
   interest modeling is unexplored. `data.py` now exposes `is_click/is_like/is_follow/
   is_comment/is_forward/play_time_ms/hourmin` per row (see `data.IDX`) in addition to
   the 5 base fields, so `features.py` can build richer per-user sequences without
   touching `data.py`.
3. **Multi-task** — auxiliary heads on `is_click`/`is_like`/`is_follow`/`is_comment`/
   `is_forward`/`play_time_ms` alongside the `long_view` main task.
4. **Censored watch-time regression** (CWM's angle) — advanced, treat as a stretch goal.
5. **Different model family** (DeepFM/DCN/xDeepFM) — lower priority than 1–3 since
   capacity isn't the bottleneck.
6. **Time features / train-test drift** — `hourmin`, `date`.
7. **Unbiased validation** — `log_random_4_22_to_5_08_pure.csv` (not currently loaded by
   `data.py`) as a sanity check against overfitting to biased exposure — would need a
   `data.py` change, so flag it rather than add it yourself.

## 6. Process

- Convergence judgment (from README): std of FM primary across seeds is **0.0008** →
  use **ε = 0.002 (≈2.5σ), N = 3** — 3 consecutive iterations with validation primary
  improving by ≤0.002 counts as converged for that line of experimentation.
- One hypothesis per iteration. Run `python train.py --model fm` (no `--final`), read the
  **valid** primary, decide keep/revert before moving on.
- Before trying a change, know what you'd revert to — commit or snapshot working
  states of `features.py`/`model.py`/`train.py` so a regression is a fast rollback,
  not a rewrite.
- Self-check the harness itself is intact by re-running `python train.py --model random --final`
  occasionally — test primary should stay ≈0.475 (±0.001) as in the README (`--final` is
  required here since this is a harness sanity check, not a feature/model iteration). If
  it drifts, something in the locked files changed or `evaluate.py`/`data.py` state is
  corrupted — stop and report, don't patch around it.
- Report scores in the same `GAUC | nDCG@5 | primary` format `train.py` already prints.
