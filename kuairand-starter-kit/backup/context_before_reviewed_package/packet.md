# Problem specification — KuaiRand-Pure

Source-derived. Official Track 2 problem statement (updated 27 Aug 2026) and the shipped
Starter Kit. Where the two disagree, the Starter Kit wins: `kit/evaluate.py` is the
scoreboard.

## The task

Within-user ranking over logged impressions. For each row of the evaluation split the
model emits one score; the evaluator ranks impressions **inside each user**. This is not
full-catalogue retrieval, and scores are never compared across users.

Relevance label: **`long_view`** (a native 0/1 column).

> The official document contains one stale line naming `is_click` with NDCG@10 and
> Recall@50. It is superseded. The benchmark table, the judging section, the appendix and
> the shipped `evaluate.py` all pin the task to `long_view` with GAUC / nDCG@5. Appendix
> A.4 explains why Recall was dropped: at roughly five impressions per user, Recall@50 is
> 0.999+ for every model including random scoring.

## Metrics

- **GAUC** — per-user AUC, weighted by that user's positive count. Only users with
  `0 < positives < impressions` contribute.
- **nDCG@5** — per user. Users with no positive score 0 and are included in the mean.
  Gain is `2^rel - 1`, which is the identity under binary labels.
- **primary = mean(GAUC, nDCG@5)** — this is what ranks submissions.

Scoring formula: `delta(m) = agent(m) - baseline(m)` for each metric, averaged. That is
exactly the primary delta against the official baseline on the hidden test set.

## Splits

Fixed, by date, and never to be changed:

    train       20220408 - 20220421     1,141,112 rows
    validation  20220422 - 20220428       124,909 rows
    test        20220429 - 20220508       170,588 rows

Development uses train and validation only. The hidden test set is scored once, on the
final submission.

## Official baseline

A numpy Factorization Machine, k=16, lr=0.001, five categorical fields. Hidden test:
GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946, mean over 5 seeds, std 0.0008. This is the
number to beat — not a baseline you build yourself.

## Reading the metric range

The metrics do not span [0, 1]. On validation, 30.3% of users are all-negative (nDCG is 0
for any model) and 11.9% are all-positive, so perfect ranking reaches only primary 0.8484
(oracle nDCG@5 0.6968). Random scoring sits at 0.4834. Judge progress against the oracle,
not against 1.0.

## Convergence and limits

    epsilon = 0.002 over N = 3 consecutive non-improving iterations
    50 iterations per run (hard cap)
    6 hours agent wall-clock (backstop)

A run is converged when validation primary has not improved by more than epsilon over the
last three iterations, or when it hits either cap — whichever comes first. **The scored
submission is the validation-best checkpoint at that point**, so continuing past
convergence cannot raise the score.

## Submission format

CSV with header `row_id,user_id,video_id,score`. `row_id` is a 0-based strictly increasing
index into the split as produced by `data.load()`. `user_id` / `video_id` are redundant
alignment checks. `score` is any real number — only relative order matters; NaN and Inf
are rejected. `row_id` is required because `(user_id, video_id)` is **not unique**: 3.06%
of evaluation rows repeat a pair, up to 12 times.

## What is scored

| Criterion | Weight | What it rewards |
|---|---|---|
| Technical Execution | 35% | primary delta on hidden test, plus robustness under failure |
| Innovation & Problem Insight | 20% | what the agent identified as worth trying, and why |
| Impact & Relevance | 20% | autonomy, measured in manual interventions |
| Feasibility & Practicality | 15% | LLM tokens and agent wall-clock — scored only if the submission beats the baseline |
| Presentation | 10% | final event only |

Robustness is judged on how a failure is handled, not on whether one occurs.

## The division of labour

Humans may provide: the task definition, rules, data, codebase, verified prior evidence,
research references and budget.

The agent must provide: interpretation, research questions, hypotheses, prioritisation,
experiment design, code changes, evaluation, reflection, and the next decision.


# Agent rules

# Agent rules — KuaiRand-Pure

You are iterating on this repo to raise the **primary** metric (`(GAUC + nDCG@5) / 2`,
see `kit/evaluate.py`) above the FM baseline (test primary **0.5946**). These rules bound
what you're allowed to touch and how you're allowed to get the score up. They exist
because an agent that can edit its own scoring code, its own data split, or its own
submission format can "improve" the number without improving the model — that result
is worthless to whoever reads the leaderboard. Follow the letter and the spirit.

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


# Constraints — measured facts only

Everything here was measured, by the organizers or by this team, on train or validation.
None of it tells you what to try next; that is your call.

## Structural, provable

C1. Ranking is WITHIN user. Any score term that is constant across one user's impressions
cannot change that user's ordering. A pure user-side first-order feature therefore
contributes exactly zero. User information can only act through a cross with the item
side. (Mathematical consequence of the metric, confirmed by the organizers: `item_pop x
user_bias` scores bit-identically to plain `item_pop`.)

C2. Any per-user monotone transform of the scores at inference is a no-op for both GAUC
and nDCG@5. So is any global calibration.

C3. 42.2% of validation users are metric-invariant: their labels are all-0 or all-1, so
nDCG@5 is pinned and GAUC excludes them. 17.5% have a single impression. No model change
reaches these users.

## Measured negative results — do not re-derive these

C4. Static feature stuffing does not help. The organizers extended the FM from 5 fields
to the 13 CWM fields: primary 0.5940 versus 0.5950 for 5 fields. Within noise, slightly
worse. Reproducible via `ablation_features.py` (do not run it — it scores test).

C5. Embedding capacity is not the bottleneck. k = 8 / 16 / 32 gives 0.5895 / 0.5902 /
0.5887.

C6. Removing fields measured slightly POSITIVE on validation: dropping `author_id` gives
+0.00157 and dropping `video_id` +0.00136, each 5/5 positive across paired seeds. The
organizers' stated reason for C4 and C5 is that the `user_id x video_id` cross already
absorbs most of the learnable signal.

C7. There are 6,510 authors for 7,583 videos and 87% of authors have exactly one video,
so `author_id` is close to a duplicate of `video_id`.

C8. Only 1.63% of validation rows have their (user, video) pair present in train, and
3.38% their (user, author) pair. 98.1% of validation users have some train history.

## Rules that are enforced in code, not by discipline

C9. Post-impression signals (is_click, is_like, is_follow, is_comment, is_forward,
is_hate, play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter, long_view)
are never same-row inputs. They are legal as auxiliary targets and as history aggregated
from strictly earlier rows. `harness/guards.py` rejects violations statically, before the
code runs.

C10. The three editable files are `pipeline/features.py`, `pipeline/model.py`,
`pipeline/train.py`. `kit/` is read-only at the filesystem level.

C11. Raw log columns are reachable only through `harness.adapter`, which date-filters to
train+valid before returning anything and aligns positionally with `kit.data.load()`.
Joining on (user_id, video_id) is wrong: that pair is not unique — 3.06% of evaluation
rows repeat it, up to 12 times.


# Method index

Short cards. What a method is, what it assumes, what it costs. Deliberately unranked and
without recommendations: choosing among these is the research decision you are here to
make, and a pre-ranked list would make you an executor.

**Pairwise ranking (BPR)** — optimise P(positive ranked above negative) within a user
instead of a pointwise probability. Assumes usable positive/negative pairs per user;
degenerate users contribute nothing. Cost: a sampler plus a change to the gradient; same
order of wall-clock as the baseline. Rendle et al., UAI 2009.

**Listwise softmax / ListNet** — a softmax over each user's impression list, cross-entropy
against the normalised label vector. Assumes lists are meaningful units. Note the
validation median list length is 4. Cao et al., ICML 2007.

**LambdaRank / LambdaMART** — weight pairwise gradients by the nDCG change from swapping
the pair. Directly targets a truncated metric. Needs grouped data and a working
delta-nDCG. Burges, 2010.

**Multi-task / ESMM-style** — auxiliary heads on other feedback signals sharing a
representation with the scored task. Assumes the auxiliary signal correlates with the
target and that shared capacity is not the binding constraint. Ma et al., SIGIR 2018.

**Censored watch-time regression (CWM)** — a completed play truncates the true watch time,
so a one-sided loss rather than squared error. Requires play_time_ms and duration_ms.
Zhao et al., KDD 2024.

**Target attention (DIN)** — score a candidate by attending over the user's history.
Assumes the candidate or its attributes recur in that history.

**Field-aware and deep factorisation (FFM, DeepFM, DCN)** — richer interaction structure
over the same sparse fields.

**Inverse propensity weighting** — reweight by exposure probability to debias a logged
policy. Requires propensities; note `is_rand` is 0 on every standard-log row, and the
random-exposure log has no rows before 20220422.

**Seed ensembling / rank averaging** — combine several models' within-user ranks. Reduces
variance rather than bias.


# Data profile (validation)

```json
{
 "_note": "Validation-split observations. Test-split structure is deliberately absent: it describes the hidden set and is not a development input.",
 "split_sizes": {
  "train": 1141112,
  "valid": 124909,
  "test": 170588
 },
 "tier_definition": "Validation users binned by their TRAIN impression count. Cold = 0 train impressions; the remaining 21,955 warm users are quartiled at edges 17 / 36 / 65. Two other tier schemes exist in this repo under the same names T1-T4 — this is the one these numbers come from.",
 "impressions_per_user_valid": {
  "n_users": 22377,
  "min": 1,
  "median": 4.0,
  "mean": 5.582026187603343,
  "max": 74,
  "p90": 12.0,
  "p99": 26.0
 },
 "metric_invariance_valid": {
  "note": "A user whose validation labels are all 0 or all 1 has an nDCG@5 pinned at 0 or 1 and is excluded from GAUC entirely. No model can move them.",
  "invariant_users_pct": 0.4222,
  "invariant_rows_pct": 0.2109,
  "all_negative_users": 6785,
  "all_positive_users": 2663,
  "single_impression_users": 3917,
  "by_tier": {
   "cold-start (0 train impr.)": {
    "n_users": 422,
    "n_rows": 1990,
    "invariant_users_pct": 0.5711
   },
   "T1 [1-17 train impr.]": {
    "n_users": 5713,
    "n_rows": 18419,
    "invariant_users_pct": 0.5881
   },
   "T2 [17-36 train impr.]": {
    "n_users": 5480,
    "n_rows": 24752,
    "invariant_users_pct": 0.4591
   },
   "T3 [36-65 train impr.]": {
    "n_users": 5373,
    "n_rows": 31501,
    "invariant_users_pct": 0.356
   },
   "T4 [65-809 train impr.]": {
    "n_users": 5389,
    "n_rows": 48247,
    "invariant_users_pct": 0.2631
   }
  }
 },
 "baseline_metrics_by_tier_valid": [
  {
   "tier": "cold-start (0 train impr.)",
   "n_users": 422,
   "n_rows": 1990,
   "GAUC": 0.6877021789550781,
   "nDCG@5": 0.530527651309967,
   "primary": 0.6091148853302002
  },
  {
   "tier": "T1 [1-17 train impr.]",
   "n_users": 5713,
   "n_rows": 18419,
   "GAUC": 0.6514714956283569,
   "nDCG@5": 0.535167932510376,
   "primary": 0.5933197140693665
  },
  {
   "tier": "T2 [17-36 train impr.]",
   "n_users": 5480,
   "n_rows": 24752,
   "GAUC": 0.6723355650901794,
   "nDCG@5": 0.5450488328933716,
   "primary": 0.6086921691894531
  },
  {
   "tier": "T3 [36-65 train impr.]",
   "n_users": 5373,
   "n_rows": 31501,
   "GAUC": 0.6633398532867432,
   "nDCG@5": 0.5521388053894043,
   "primary": 0.6077393293380737
  },
  {
   "tier": "T4 [65-809 train impr.]",
   "n_users": 5389,
   "n_rows": 48247,
   "GAUC": 0.6725407242774963,
   "nDCG@5": 0.5112046599388123,
   "primary": 0.5918726921081543
  },
  {
   "tier": "ALL",
   "n_users": 22377,
   "n_rows": 124909,
   "GAUC": 0.6671333909034729,
   "nDCG@5": 0.5358057022094727,
   "primary": 0.6014695167541504
  }
 ],
 "current_representation": {
  "fields": [
   "user_id",
   "video_id",
   "author_id",
   "tab",
   "dur_bucket"
  ],
  "encoded_dim": 40260,
  "row_tuple_from_kit_data_load": [
   "date",
   "user_id",
   "video_id",
   "author_id",
   "tab",
   "duration_ms",
   "label"
  ],
  "note": "kit/data.py exposes only these seven fields. The other 12 log columns and all user/video side tables are reachable only through harness.adapter, which date-filters to train+valid."
 },
 "available_but_unloaded": {
  "log_columns": [
   "hourmin",
   "time_ms",
   "is_click",
   "is_like",
   "is_follow",
   "is_comment",
   "is_forward",
   "is_hate",
   "play_time_ms",
   "profile_stay_time",
   "comment_stay_time",
   "is_profile_enter",
   "is_rand"
  ],
  "side_tables": {
   "user_features_pure.csv": 30,
   "video_features_basic_pure.csv": 12,
   "video_features_statistic_pure.csv": 52
  },
  "access": "harness.adapter.raw_columns() / entity_table()"
 }
}
```

# Baseline and noise

```json
{
 "_note": "Validation reference rungs. The published hidden-test numbers exist in kit/baseline_scores.json and are the competition target, but they are not a development signal and are not repeated here.",
 "target_to_beat_hidden_test_primary": 0.5946,
 "validation": {
  "random": {
   "GAUC": 0.4993,
   "nDCG@5": 0.4675,
   "primary": 0.4834
  },
  "item_popularity": {
   "GAUC": 0.6387,
   "nDCG@5": 0.5227,
   "primary": 0.5807
  },
  "fm_official": {
   "GAUC": 0.6674,
   "nDCG@5": 0.5357,
   "primary": 0.6016
  },
  "oracle_ceiling": {
   "GAUC": 1.0,
   "nDCG@5": 0.6968,
   "primary": 0.8484
  }
 },
 "noise": {
  "published_seed_std_test": 0.0008,
  "measured_seed_std_valid_population": 0.00032,
  "measured_paired_delta_sigma_valid": 0.0005,
  "note": "The widely-quoted 0.0008 is the TEST seed std. On validation, five identity seeds give a population std of 0.00032 and paired deltas run about 0.0005. So eps = 0.002 is roughly 4 sigma: a null iteration essentially never clears it."
 },
 "convergence": {
  "epsilon": 0.002,
  "N": 3
 },
 "run_limits": {
  "max_iterations": 50,
  "wall_clock_hours": 6
 },
 "baseline_config": {
  "model": "FM",
  "k": 16,
  "lr": 0.001,
  "batch": 8192,
  "max_epochs": 40,
  "patience": 4,
  "fields": [
   "user_id",
   "video_id",
   "author_id",
   "tab",
   "dur_bucket"
  ]
 },
 "reproduced_here": {
  "valid_primary": 0.60147,
  "best_epoch": 7,
  "early_stop_epoch": 11
 }
}
```