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

## 0b. Prior-history features: use `harness.history`

**A fifth defect, found by running the agent.**

The rules permit an outcome column to become an input feature when it is aggregated from
*strictly earlier* rows (§3). Nothing made that possible. `same_row()` refuses `label` by
name, the static guard rejects any other `IDX['label']` read inside `features.py`, and no
helper existed — so the one legal route to headroom idea 2 was documented as legal and
was unreachable in practice. §5 item 2 compounded it by pointing at raw-CSV reads, which
the guard also rejects.

> Scope note, for the record: this gap was found alongside a *separate* defect — see §0c —
> and the two were initially conflated. The guard rejections observed in the measured runs
> were caused by §0c, not by this one. No run has yet been blocked by the missing history
> route, because no proposal has reached for it yet. The gap is real regardless: without
> a helper, the first history hypothesis would have hit the same wall.

`harness/history.py` is the route:

```python
from harness.history import prior_stats, bucketize

rate, count = prior_stats(splits, signal='label', key='user_id')
# rate[s], count[s] are float arrays aligned 1:1 with splits[s], for train/valid/test
edges = np.quantile(rate['train'], [0.2, 0.4, 0.6, 0.8])   # TRAIN only
bucket = {s: bucketize(rate[s], edges) for s in splits}
```

- `signal`: `'label'`, or any post-impression column (`is_click`, `is_like`,
  `play_time_ms`, …) — sourced through the adapter, so test rows never enter memory.
- `key`: a row field, or a tuple — `('user_id', 'author_id')` for user-author affinity.
- `prior_weight`: Bayesian smoothing toward the train global mean. `count` tells the
  model how much the estimate rests on.

Three guarantees, proven by tests in `tests/test_harness.py` rather than by discipline:
strictly-earlier only; **ties are not predecessors** (5.60% of validation rows share a
user/timestamp); and **test outcomes never enter any aggregate** — flipping every test
label leaves every output byte-identical.

This is a mechanism, not a recommendation. Whether a prior aggregate helps, at which key
and on which signal, is yours to determine.

## 0c. The feature-builder label rule is scoped to `features.py`

**A sixth defect, found by running the agent — this is the one that was actually
blocking every iteration.**

`pipeline/train.py` legitimately reads `IDX['label']`: it builds training targets and
hands ground truth to `evaluate()`. Three such lines ship in the file today. The static
guard knows this and scopes its feature-builder rule to `features.py`.

`scan_diff` did not honour that scope. It flattened the added lines of *every* file in a
diff into one blob and scanned it as if it were `features.py`. Because `coder.py` emits
whole files, every line of `train.py` reappears as an added line — so **any** change to
`train.py` was rejected, including a re-emission of the pristine shipped file.

The effect: the entire training loop was sealed off — loss function, batching, early
stopping, model selection. Three measured runs proposed a within-user listwise softmax
(headroom idea 1, the one the organizers rate most likely to help) and all nine attempts
were rejected on `train.py`'s own shipped lines. Two runs designated the baseline at a
0.0000 delta before the cause was found.

Fixed in `harness/guards.py`: `_added_by_file()` attributes added lines to their source
file, so the feature-builder rule judges only `features.py`. Verified in three
directions and regression-tested (`test_guard_scopes_label_rule_to_features_file`):
pristine `train.py` passes, a direct label read in `features.py` still fails, and a diff
touching `kit/` still fails.

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
  **For (b), use `harness.history`** — see §0b. Rolling your own aggregate inside
  `features.py` means touching `IDX['label']`, which the static guard rejects, and it is
  the wrong place to get the ordering right anyway.

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
   `play_time_ms`/`hourmin`/etc. — it only loads the 5-field baseline row. **Superseded
   by §0b:** an earlier version of this doc told you to read the raw CSVs from
   `features.py`. Do not — the static guard rejects `open(...log_standard...)`, because
   `log_standard_4_22_to_5_08_pure.csv` spans validation *and* test and carries
   `long_view`. Use `harness.adapter` for raw columns and `harness.history` for prior
   aggregates; both date-filter to train+valid before anything reaches you.
3. **Multi-task** — auxiliary heads on `is_click`/`is_like`/`is_follow`/`is_comment`/
   `is_forward`/`play_time_ms` alongside the `long_view` main task. Source them with
   `harness.adapter.auxiliary_targets()`, which returns them as targets and refuses test
   rows.
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


# KuaiRand-Pure — Established Evidence / Constraints

> Purpose: verified prior knowledge the autonomous research agent may rely on.
>
> This file states **what is true**, not what to try. Every entry says what its evidence
> establishes *and* what it does not establish, because the second half is what keeps a
> measurement from silently becoming a directive.
>
> **Scope of all local evidence: train + validation only.** No entry here rests on
> evaluation/test labels or on evaluation-period outcomes. Published organizer test
> numbers appear only where they are explicitly labelled as official reference material.
>
> Classification vocabulary is defined in `RULES.md` §8. Detailed derivations are in
> `research/PRE_AUDIT.md`; compact numbers are in `research/data_profile.md`.
>
> Constraint IDs C1–C6 retain the meanings they had before the audit update; C7 onward
> were added from the reviewed audit evidence. IDs are stable: a retired entry keeps its
> number and a marker saying where its evidence now lives, so that references from
> `PRE_AUDIT.md`, `REVIEW_REPORT.md`, and the other context files never silently
> re-point.
>
> Do not add to this file without human review. Do not put hypotheses or strategy here.

---

## 1. Official and Mathematical Facts

### C1 — Official target, task form, and metric semantics

**Classification:** HARD FACT

**Evidence:**
- Target `long_view` (binary); task is within-user ranking over logged impressions;
  `primary = (GAUC + nDCG@5) / 2`.
- `source/starter-kit/evaluate.py` is the scoring authority and defines: GAUC over
  users with `0 < positives < impressions`, weighted by each such user's positive
  count; nDCG@5 averaged over **all** users with equal weight; gain `2^rel − 1`; an
  empty GAUC denominator returns the 0.5 fallback.
- Official splits: train 2022-04-08..21, validation 2022-04-22..28, evaluation
  2022-04-29..05-08.

**Interpretation:**
Two metrics with two different user weightings are averaged into one number. A change
can move one and not the other, and the two populations they score are not the same set
of users.

**Does NOT establish:**
Nothing about which model, loss, or feature set scores well. It also does not license
any locally reimplemented or reweighted metric standing in for `evaluate.py`.

---

### C2 — Terms constant within a user cannot change that user's ranking

**Classification:** HARD FACT

**Evidence:**
- Mathematical consequence of ranking strictly within a user.
- Organizer-confirmed by measurement: the official README reports that
  `item_pop × user bias` and plain `item_pop` produce identical scores to the digit.

**Interpretation:**
A purely additive first-order user-side term contributes exactly zero to the metric.
User-side information can only reach the score through terms that vary within the
user — interactions with item/context features, shared representations, or
user-conditioned behaviour.

**Does NOT establish:**
That user features are useless, or that user-side modelling is a dead end. It
constrains the *form* in which user information can act, not its value.

---

### C3 — Same-row post-impression feedback as an input

**Classification:** INVALID / FORBIDDEN

**Evidence:**
- Post-impression columns: `is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`, `is_hate`, `play_time_ms`, `profile_stay_time`, `comment_stay_time`,
  `is_profile_enter`.
- These are concurrent outcomes of the same impression as `long_view`. Measured
  same-row association is large for the dense ones — validation Pearson r with
  `long_view` is 0.7515 for `is_click` and 0.6319 (raw) for `play_time_ms` — which is
  exactly why using them as same-row inputs produces a meaningless score.

**Interpretation:**
Forbidden as an input to the same row's prediction, in any form, including hidden
inside a derived "engagement score". Permitted as (a) auxiliary targets, (b) features
aggregated from strictly earlier rows, (c) diagnostics.

**Does NOT establish:**
Anything about whether auxiliary targets or historical aggregates of these signals
*help*. That is untested and open.

---

### C4 — Multi-feedback auxiliary learning is permitted

**Classification:** HARD FACT

**Evidence:**
- The official material identifies KuaiRand's multi-feedback structure as a legitimate
  setting for auxiliary-task learning. Only `long_view` is scored.

**Interpretation:**
Predicting feedback signals as auxiliary targets is allowed under the rules.

**Does NOT establish:**
That any auxiliary task helps. Negative transfer is a real possibility here and is
unmeasured on this dataset; auxiliary densities range from ~46% (click) to ~0.1%
(follow, forward, hate), which is a wide spread for shared-representation methods.

---

### C23 — Per-user monotone transforms and global calibration are no-ops

> ID out of sequence: constraint IDs are append-only, so a late addition to an early
> section keeps the next free number rather than renumbering the file.

**Classification:** HARD FACT

**Evidence:**
- Both metrics depend only on the within-user *ordering* of scores: GAUC through
  Mann-Whitney U over ranks, nDCG@5 through the label sequence after sorting by score.
- Any strictly increasing function applied to one user's scores preserves that ordering,
  as does any global calibration (Platt scaling, temperature, an isotonic fit over the
  whole split).

**Interpretation:**
Probability calibration cannot move either metric. A well-calibrated score and a wildly
mis-calibrated one that ranks identically receive the same primary.

**Does NOT establish:**
That calibration is pointless during *training* — a calibrated objective can still change
what the model learns, and that is a separate question. Nor does it extend to transforms
that differ across a user's own rows: those change ordering and are not covered here.

---

## 2. Negative Evidence on the Official FM — Organizer-Confirmed and Locally Reproduced

> Both entries in this section are scoped to the **exact official five-field FM**. Neither
> generalises to a different loss, encoding, or model family.

### C5 — Static-feature expansion of the official FM

**Classification:** STRONG NEGATIVE EVIDENCE (for the exact 13-field formulation)

**Evidence:**
- Local, validation, 3 matched seeds. Base (5 fields) 0.601440 ± 0.000275.
- Full static bundle (**13** fields): 0.599930 ± 0.000523; paired delta
  **−0.001510 ± 0.000792**.
- Item-only expansion (**8** fields): 0.601108 ± 0.000461; paired delta
  −0.000332 ± 0.000205.
- Organizer reference (published, test split): 13 fields 0.5940 vs 5 fields 0.5950.
- The configurations contain 8 and 13 fields — the "9/14" labels in older notes are
  wrong; see `DATA_GUIDE.md` §8.

**Interpretation:**
The exact 13-field static bundle is reproducibly slightly worse than the five-field
baseline, and independently so in the organizer's own test-split run. Coarse static
buckets add little on top of identity fields in this FM.

**Does NOT establish:**
That static or derived features are useless in other models, encodings, or objectives.
The 8-field item-only expansion is **INCONCLUSIVE**, not negative — its delta is small
relative to run variability, so that configuration remains an open question rather than
a closed one.

---

### C6 — Simple FM width scaling produced no meaningful gain

**Classification:** STRONG NEGATIVE EVIDENCE (for simple width scaling in this FM)

**Evidence:**
- Local, validation, 3 seeds per setting, mean primary ± population std:
  k=8 0.60111 ± 0.00080; k=16 0.60144 ± 0.00027; k=32 0.60146 ± 0.00069;
  k=64 0.60099 ± 0.00044.
- Spread across all four widths is smaller than the 0.002 practical epsilon and
  comparable to seed noise.
- Organizer reference (published, test split): k = 8/16/32 gave 0.5895 / 0.5902 /
  0.5887.
- Scope: the official five-field FM, embedding width varied alone, nothing else changed.

**Interpretation:**
Embedding width alone is not a lever in this exact model at this data size.

**Does NOT establish:**
That capacity in general is irrelevant, or that other model families, regularisation
schemes, schedules, or objectives cannot benefit from more parameters. It says nothing
about width in a *different* architecture.

---

## 3. Evidence Integrity

### C7 — Evaluation-period information is not development evidence

**Classification:** INVALID / FORBIDDEN

**Evidence:**
- Official rule: development uses train + validation only.
- Two source audits violated this before review — one locally scored standard-test
  labels, another summarised evaluation-period random-log outcomes and used
  evaluation-period identities in a comparison set. Every such result was removed from
  the evidence base rather than annotated.

**Interpretation:**
No locally computed test metric, evaluation-period outcome, evaluation-period feature
summary, or evaluation-period identity comparison may support a development decision.
Permitted during development: counting evaluation-window rows by `date`, and reading
the organizer's published test numbers as reference material.

**Does NOT establish:**
That evaluation rows may never be touched — the frozen final model is applied to
evaluation features to produce a submission. The prohibition is on evaluation
information flowing *back* into development.

---

### C8 — The official baseline reproduces locally, and the noise scale is known

**Classification:** HARD FACT

**Evidence:**
- Reproduced seed 0, official configuration, validation: GAUC 0.667133,
  nDCG@5 0.535806, primary 0.601470 — against published 0.6674 / 0.5357 / 0.6016.
  Seed 0 selected epoch 7 and stopped after epoch 11.
- Five-seed validation-only rerun: mean primary 0.60157, population std 0.00032.
- Official configuration: fields `user_id`, `video_id`, `author_id`, `tab`,
  `dur_bucket`; k=16; lr=0.001; L2=1e-6; batch 8,192; max 40 epochs; patience 4;
  pointwise binary cross-entropy; Adam on W/V with a plain bias update; early stopping
  on validation primary.
- Published organizer seed std: 0.0008. The convergence epsilon 0.002 is roughly 2.5x
  that figure.

**Interpretation:**
The local environment reproduces the official baseline within seed and rounding
variation, so validation deltas measured in this environment are comparable to the
official ones. **Use 0.0008 as the generic noise reference, not the narrower local
0.00032** — a delta under ~0.0008 is indistinguishable from a seed, and a delta under
0.002 is below the competition's own practical threshold.

**Does NOT establish:**
Any local test result, and nothing about alternative models. The reproduction validates
the harness, not a research direction. The training objective being pointwise while the
metric is rank-based is a fact about the code, not evidence for any particular
alternative objective — none was tested.

---

## 4. Metric Structure

### C9 — Validation label composition, GAUC eligibility, and the oracle ceiling

**Classification:** HARD FACT

**Evidence:**
- Validation users 22,377 / rows 124,909.
- All-negative 6,785 users (30.321%), 21,807 rows (17.458%).
- All-positive 2,663 users (11.901%), 4,540 rows (3.635%).
- Mixed-label 12,929 users (57.778%), 98,562 rows (78.907%).
- Single-impression users: 3,917 (17.505%).
- Reproduced validation oracle (true labels as scores): GAUC 1.0000, nDCG@5 0.6968,
  primary 0.848393.
- Scope: official validation split, official evaluator.

**Interpretation:**
42.222% of validation users have uniform labels: they contribute a fixed nDCG (0 or 1)
regardless of ranking and are excluded from GAUC entirely. The mixed-label 57.778% of
users hold 78.907% of validation rows. Progress should be read against the 0.848393
ceiling, not against 1.0.

**Does NOT establish:**
That uniform-label users should be filtered out of training, that mixed users should be
upweighted, or that any reweighting improves the metric. Training composition and
metric composition are different questions, and the relationship between them is
untested.

---

### C10 — Official GAUC weight denominator, and where baseline headroom sits

**Classification:** HARD FACT (for this reproduced baseline and these fixed bucket definitions)

**Evidence:**
- The official GAUC denominator is **34,592 positive rows belonging to mixed-label
  users only** — not all positive validation rows. All shares below use it.
- Activity tiers are quartiles of train interaction count among warm validation users:
  Cold 0, T1 1–17, T2 18–36, T3 37–65, T4 66+.

| Tier | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 1.69% |
| T1 | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 14.67% |
| T2 | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 21.35% |
| T3 | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 27.50% |
| T4 | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 34.79% |

| List length | Users | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|
| 1 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.00% |
| 2–3 | 6,218 | 0.6472 | 0.5413 | 0.6086 | 10.27% |
| 4–5 | 4,119 | 0.6645 | 0.6185 | 0.7492 | 16.36% |
| 6–10 | 5,225 | 0.6756 | 0.5913 | 0.8536 | 36.39% |
| 11–20 | 2,346 | 0.6677 | 0.5037 | 0.9182 | 27.08% |
| 21+ | 552 | 0.6596 | 0.3934 | 0.9420 | 9.90% |

`*` empty GAUC denominator; the evaluator's 0.5 fallback, contributing zero weight.

- Train activity and validation list length are related but distinct: Spearman rho 0.4620.
- The joint cell T3/T4 x list-length 6+ holds 5,680 users (25.38%), 64,133 rows
  (51.34%), 50.79% of official GAUC weight, and 51.72% of the seed-0
  baseline-to-oracle primary gap.
- All 30 activity x list cells reconcile to 22,377 users, 124,909 rows, 100% of GAUC
  weight, and the full baseline-to-oracle gaps.
- Scope: reproduced seed-0 baseline predictions under the official configuration; these
  bucket edges only. nDCG contributions are equal-user weighted and are never
  multiplied by GAUC shares.

**Interpretation:**
Metric weight and current-baseline headroom are unevenly distributed across users, and
activity and list length are correlated without being interchangeable.

**Does NOT establish:**
Causality, attainable gain, or that any weighting, loss, filter, or model closes the
gap where it currently sits. A large share of headroom is not evidence that the share
is reachable, and headroom measured under *this* baseline may move under another.

---

## 5. Dataset Structure

### C11 — Development scale, coverage, and missingness

**Classification:** HARD FACT

**Evidence:**
- Train 1,141,112 rows / 26,210 users / 7,538 videos / 6,482 authors / 15 tabs / 13 dates.
- Validation 124,909 rows / 22,377 users / 5,951 videos / 5,315 authors / 15 tabs / 7 dates.
- Side tables: 27,285 users, 7,583 videos (basic), 7,583 videos (statistic); each covers
  100% of train and validation entities.
- Logs and the video-statistic table have no missing cells. Localised missingness: user
  `onehot_feat4` 3.2032%; `onehot_feat12`..`17` 2.6168% each; basic `video_duration`
  3.1518%; `music_type` 2.6770%; `tag` 1.2660%. Basic `visible_status` is constant.
- Train impressions per user: median 31, p90 97, p99 207, max 809 (all train users).
- Validation list length: median 4, p90 12, p99 26, max 74.

**Interpretation:**
Every side table joins completely; missingness is confined to a handful of columns and
is small; validation lists are short, with a median of four impressions.

**Does NOT establish:**
Usefulness, encoding, or causal validity of any field. Complete coverage is a join
property, not an information property.

---

### C12 — Warm entities, novel pairs, and author/video structural redundancy

**Classification:** HARD FACT

**Evidence:**

*Entity vs relationship overlap (validation against train):*
- Users seen in train: 21,955 / 22,377 = **98.114%**
- Videos seen: 5,944 / 5,951 = **99.882%**
- Authors seen: 5,310 / 5,315 = **99.906%**
- Unique user–video pairs seen: 1,974 / 121,337 = **1.627%**
- Unique user–author pairs seen: 4,081 / 120,885 = **3.376%**
- Raw user–tag-string pairs seen (missing tag = one explicit category):
  61,405 / 90,121 = **68.14%**. A parsed multi-token construction instead yields
  71.913% pair overlap and 78.413% validation-row coverage — a *different definition*,
  not a competing measurement of the same thing.

*Within-train repeat structure:*
- 4.130% of unique user–video pairs repeat, covering 8.194% of rows.
- 5.913% of unique user–author pairs repeat, covering 11.750% of rows.

*Author/video redundancy:*
- Every video maps to exactly one author (functional dependency).
- Full basic file: 5,661 / 6,510 authors (**86.96%**) have exactly one video; median 1,
  mean 1.165, max 26.
- Restricted to train/validation-observed videos: 5,647 / 6,487 (**87.051%**), max 24.

**Interpretation:**
Validation is almost entirely warm at the level of individual entities but almost
entirely novel at the level of the exact user–item and user–author relationship.
`author_id` carries substantial structural redundancy with `video_id` for most of the
catalogue. Coarser relationships (tags) have far broader support, with the number
depending on which tag representation is used.

**Does NOT establish:**
That `author_id` should be removed, that author-level interactions cannot add value,
that sparse exact-pair history is useless, or that tag features help. It also does not
merge the two tag definitions into one number — they must never be compared directly.

---

### C13 — Strictly prior history: broad at coarse granularity, sparse at fine

**Classification:** HARD FACT

**Evidence:**
- Every train timestamp precedes every validation timestamp, so all of train is
  legitimate history for validation.
- Validation users with ≥1 / ≥5 / ≥10 prior train interactions: 98.114% / 92.854% /
  **85.168%**. Median / mean / p90 prior interactions per validation user: 35 / 47.42 /
  103. (This median is over *validation users' train history*; the median over all
  train users is 31 — see C11. Different populations, both correct.)
- Validation users with ≥1 / ≥5 / ≥10 prior clicks: 96.157% / 82.531% / 66.309%.
  With prior likes: 23.229% / 4.683% / 2.239%. Follows, comments, forwards, and hates
  are sparser still.
- Validation rows with a prior interaction on the **same video**: **1.624%**.
  Same **author**: **3.381%**. A prior **parsed tag token**: **78.413%**.
- Availability diagnostic: 81.57% of validation rows have a strictly earlier same-user
  validation timestamp. Tied timestamps are **not** predecessors, and 5.60% of rows sit
  in non-unique user/timestamp groups.

**Interpretation:**
User-level and click-level history is broadly available; exact item or author repeats
are rare; coarse content granularity has wide coverage. Support therefore depends
heavily on the granularity chosen, and the two ends differ by more than an order of
magnitude.

**Does NOT establish:**
That aggregates, sequence models, or tag attention improve validation — none was
tested. The 81.57% figure is an availability count, **not** a validated online-history
protocol; whether within-validation outcomes can be made available before scoring
remains open. The official documentation also notes KuaiRand-Pure has incomplete
sequential logs.

---

### C14 — Temporal volume and period-level distribution shift

**Classification:** HARD FACT (component measurements)

**Evidence:**
- The nominal train date 2022-04-08 has **zero rows**; train rows span 13 dates.
- Daily volume falls overall across the train window with small reversals: peak 278,835
  rows on 04-11, final train day 20,021 rows on 04-21 — a **13.9x** ratio.

| Period | Rows | Rows/day | `long_view` rate | Mean duration |
|---|---:|---:|---:|---:|
| Early train 04-09..14 | 891,418 | 148,570 | 0.33228 | 98,553 ms |
| Late train 04-15..21 | 249,694 | 35,671 | 0.35211 | 95,477 ms |
| Validation 04-22..28 | 124,909 | 17,844 | 0.31328 | 102,820 ms |

- Validation sits closer to **early** train in target rate (gap 0.01900 vs 0.03882) and
  mean duration (4,267 vs 7,343 ms), but closer to **late** train in tab distribution,
  volume, and some entity-set measures.

**Interpretation:**
Temporal change across the window is real, large in volume terms, and
multidimensional — different dimensions point in different directions.

**Does NOT establish:**
Whether validation "resembles" late train overall — that single verdict is
**INCONCLUSIVE**, and the component measurements must not be collapsed into it. It
establishes nothing about whether recency weighting, date features, or dropping early
rows helps.

---

## 6. Controlled Field Ablation on the Official FM

> Scoped to the **exact official FM formulation**, train/validation, official evaluator.
> It does not generalise to a different loss, encoding, or model family.

### C15 — Removing `tab` from the official FM

**Classification:** STRONG NEGATIVE EVIDENCE (against the removal)

**Evidence:**
- 3 matched seeds. Base 0.601440 ± 0.000275; without `tab` 0.585538 ± 0.000429.
- Paired delta **−0.015903 ± 0.000467** — roughly 8x the practical epsilon and 20x the
  published seed std.

**Interpretation:**
`tab` carries distinct, large value in this FM. Dropping it is a substantial regression.

**Does NOT establish:**
That `tab` must appear in every future model in this exact form, or that context
information is exhausted by a single 15-value categorical field.

---

### C16 — Retired (dual `video_id` + `author_id` ablation)

**Status:** withdrawn from `constraints.md` by human decision; the ID is retired rather
than reused.

The controlled five-seed ablation of the two identity fields is a **measurement**, not a
constraint. It is recorded in `research/data_profile.md` §12 with its paired deltas,
seed counts, and the reviewer's WEAK NEGATIVE EVIDENCE classification for the exact FM
formulation.

It was removed from this file because it was the only positive-direction result in the
package, and a positive result sitting among established constraints risks reading as a
direction to pursue. The related **structural** redundancy between `author_id` and
`video_id` remains available as evidence in C12, where it belongs.

---

## 7. Data-Source Scope Boundaries

### C17 — Retired (video-statistic provenance and causal safety)

**Status:** withdrawn from `constraints.md` by human decision; the ID is retired rather
than reused.

This was the package's one INCONCLUSIVE entry, retained against the general rule that
inconclusive findings stay out of this file. It has been removed to keep that rule
absolute: an open question does not belong among established constraints, however
consequential it is.

The substance is preserved in full, and in the two places an implementer actually reads:

- `DATA_GUIDE.md` §6 — the semantic caveat, stated before any use of these fields:
  values are averages not totals, the window's endpoints are undocumented, the source
  population is undocumented, 54 field pairs are near-redundant, and causal
  admissibility for an April 22–28 impression is therefore **not established** —
  neither a settled ban nor a clearance.
- `research/data_profile.md` §9 — the measurements: reconstruction ratios, redundancy
  count, marginal association, and the explicit INCONCLUSIVE verdict on aggregation
  population, calendar window, and causal safety.

Whether these fields can be given acceptable provenance, and whether they add anything
beyond identity features, is open work for the agent.

---

### C18 — Random-exposure log: permitted scope and validation-slice structure

**Classification:** HARD FACT (retained structure). The evaluation-period portion is
INVALID / FORBIDDEN as development evidence under C7.

**Evidence:**
- File total 1,186,059 rows spanning 2022-04-22..05-08.
- Validation dates 04-22..28: **288,338** rows. Evaluation dates 04-29..05-08:
  **897,721** rows, counted by `date` only.
- Validation slice: 19,091 users, 7,546 videos, `long_view` rate **0.08056** — against
  standard validation's 0.31328.
- The validation slice shares **17 of its 288,328** unique user–video pairs (0.006%)
  with standard validation.
- Older figures derived from full-file entity/pair inspection or from locally inspected
  evaluation-period outcomes were removed, not corrected.

**Interpretation:**
The eligible validation-period random stream is distributionally very different from
standard validation traffic — roughly a quarter of the positive rate — and is almost
entirely pair-disjoint from it.

**Does NOT establish:**
A propensity estimator, an unbiased replacement metric, a training use, or predictive
validity for standard traffic. Whether this slice is a useful secondary diagnostic is
**INCONCLUSIVE** and untested.

---

## 8. Engineering Constraints

### C19 — A bare Windows subprocess timeout did not bound the tested process tree

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- On the tested Windows inherited-pipe process tree,
  `subprocess.run(timeout=3, capture_output=True)` returned only after the grandchild's
  full **30.13 s** lifetime.
- A separate recursive-termination probe did successfully remove both parent and child.

**Interpretation:**
A timeout argument alone is not a guarantee on this platform when a child spawns its own
child and pipes are inherited. Process-tree control has to be explicit.

**Does NOT establish:**
That any particular replacement mechanism works — the recursive probe was a different
test under different conditions, and the harness that will run the measured iterations
has not been validated against this failure mode.

---

### C20 — Baseline runtime and caching, as observed

**Classification:** ENGINEERING CONSTRAINT (run- and implementation-specific)

**Evidence:**
- One reviewer rerun: ~57.5 s cold baseline (2.99 s load, 4.81 s encode, 49.7 s train);
  0.018 s cache read; bit-identical arrays on reload.
- A separate fingerprinted implementation: 78.52 s cold; 1.384 s for full-content
  fingerprint plus cache read; it correctly rejected a changed source fingerprint.
- Environment: Windows 11, Python 3.13.7, CPU only.

**Interpretation:**
A baseline-class iteration costs on the order of a minute on CPU, and deterministic
caching of the load/encode stages is achievable. The two timings come from different
implementations under different instrumentation — they are not a contradiction and not a
stable benchmark.

**Does NOT establish:**
Runtime for any more complex model, a validated six-hour autonomous run, resume or
checkpoint behaviour, or a production cache policy. Budgeting 50 iterations against
6 hours on these numbers assumes iterations stay baseline-class, which is not given.

---

### C21 — Submission and scoring validity contract

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- The official submission reader rejects `NaN` and `Inf` scores.
- `row_id` is the required key: `(user_id, video_id)` is not unique in the evaluation
  split (the organizer reports 3.06% duplicated pairs, up to 12 repeats).
- A syntax error in a child process returns a detectable nonzero exit status.

**Interpretation:**
Score-validity and row-alignment failures are detectable before submission and must be
checked rather than assumed.

**Does NOT establish:**
That the current repository implements any of these checks — see C22.

---

### C22 — The implementation layer is unimplemented scaffolding

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- All **15** files under `harness/`, `pipeline/`, and `agent/` contain **zero**
  executable non-comment lines. `pipeline/models/`, `pipeline/objectives/`, and
  `agent/prompts/` are empty directories.
- An earlier probe counted only seven such files; review corrected this to 15.

**Interpretation:**
Nothing in the autonomous system is built yet. Every guarantee the run depends on —
scoring, execution, timeout handling, caching, logging, submission validation — is
currently a comment.

**Does NOT establish:**
Any estimate of the effort required, or that the intended design is sound. It states
repository state at audit time; re-verify before relying on it.

---

> Provenance citations (PRE_AUDIT / REVIEW_REPORT per entry) and the file-maintenance
> section are omitted from this packet copy. They are intact in `context/constraints.md`.


# KuaiRand-Pure — Research Reference Index

> Purpose: give the proposer/research agent a compact technical toolbox.
> This file describes methods; it does **not** rank them or recommend a winner.
>
> Entries are grouped by the problem they address, and the ordering within a group is
> arbitrary. Where the **organizer** states a preference, it is quoted as the
> organizer's judgement and labelled as such — that is a description of an official
> source, not this project's research strategy.
>
> A method staying in this file is not an endorsement, and a method whose one tested
> implementation underperformed is not removed: a negative result belongs to the exact
> formulation tested (see `constraints.md`), not to the method family.

## 0. Task Shape — Read This Before Reaching for a Method

The scored task is **re-ranking a fixed, logged candidate list within each user**. It is
not full-catalogue retrieval. Consequences for method selection:

- There is no candidate-generation stage to optimise, and no negative sampling problem
  in the usual sense — the impressions are given.
- Retrieval-oriented architectures (two-tower + ANN index, in-batch softmax over the
  full catalogue) do not map onto this task without being repurposed. Their loss
  formulations may still transfer; their serving structure does not.
- Anything constant within a user's list cannot change that user's score (`constraints.md`
  C2). This is a hard filter on what a method can possibly do here.
- Validation lists are short (median 4 impressions), which bounds how much a list-level
  method has to work with per group.

---

## 1. Factorization Machines (FM)

**Problem addressed:** sparse categorical interactions with limited data.

**Mechanism:** each categorical value gets a latent vector; the model scores first-order
terms plus all pairwise dot products between field embeddings, computed in linear time
via the sum-of-squares identity.

**Assumptions:** interactions are well approximated by inner products of low-rank
embeddings; features are categorical or bucketed.

**Implementation notes:** this is the official baseline family
(`source/starter-kit/baseline.py`) — numpy only, roughly a minute per CPU run. Its exact
mechanism and configuration are recorded in `constraints.md` C8.

**Variants worth knowing:** field-aware FM (FFM) gives each field pair its own embedding,
multiplying parameters by the field count; higher-order FM extends beyond pairwise.

---

## 2. BPR / Pairwise Ranking

**Problem addressed:** the mismatch between a pointwise probability objective and a
rank-based metric.

**Mechanism:** train on (positive, negative) pairs drawn from the same user, maximising
the probability that the positive scores higher. Gradients depend only on score
differences within a user.

**Assumptions:** meaningful pairs exist within a group — a user with uniform labels
generates none.

**Implementation notes:** pair construction must respect user grouping. Note that 42.222%
of validation users have uniform labels (`constraints.md` C9) and would supply no pairs;
what that implies for training composition is untested. Pair count grows with the product
of positives and negatives per user, so sampling policy matters for runtime.

---

## 3. Listwise Ranking

**Problem addressed:** optimising a whole ranked list rather than independent rows.

**Mechanism:** a softmax (or other list-level loss) over the scores of one user's
impressions; the loss sees the entire group at once.

**Assumptions:** the group used in training matches the group used in scoring — here,
one user's logged impressions.

**Implementation notes:** batching must preserve user grouping, which usually means
restructuring the batch builder rather than only the loss. Short lists (median 4) mean
many groups contribute few terms.

---

## 4. LambdaRank / LambdaMART

**Problem addressed:** directly targeting a rank metric such as nDCG.

**Mechanism:** reweight pairwise gradients by the metric change that swapping the pair
would cause, so the optimiser spends effort where the metric is sensitive.

**Assumptions:** the target metric decomposes into pairwise swaps — nDCG does.

**Implementation notes:** commonly implemented over GBDTs (LightGBM `lambdarank`), which
needs group boundaries supplied explicitly and dense/numeric features rather than raw
high-cardinality IDs. LightGBM is explicitly in scope (see the resource policy in §17);
the Starter Kit's numpy-only footprint is not a restriction, though adding any dependency
still costs setup time against the wall-clock budget.

---

## 5. Deep Interaction Models — DeepFM / DCN / xDeepFM

**Problem addressed:** interaction orders and nonlinearities an FM cannot express.

**Mechanism:** DeepFM pairs an FM component with an MLP over the same embeddings; DCN
stacks explicit feature-crossing layers; xDeepFM adds a compressed interaction network
for explicit higher-order interactions.

**Assumptions:** enough data to fit the extra parameters without overfitting.

**Implementation notes:** all three need a tensor framework. Relevant context: simple
width scaling of the official FM produced no gain (`constraints.md` C6) — that result is
about *width in that FM*, and says nothing directly about interaction *structure*, which
is what these models change.

---

## 6. DIN / SIM — Behaviour-History Attention

**Problem addressed:** representing a user's history conditioned on the candidate item.

**Mechanism:** attend over the user's historical behaviour sequence with the candidate as
the query, producing a candidate-specific user representation.

**Assumptions:** history is available, ordered, and shares a vocabulary with the
candidate.

**Implementation notes:** feasibility depends on measured coverage, which differs by
granularity by more than an order of magnitude (`constraints.md` C13): exact same-video
history covers 1.624% of validation rows, same-author 3.381%, parsed tag tokens 78.413%.
The official documentation also notes KuaiRand-**Pure** has incomplete sequential logs
and points to the 27K/1K variants when rigorous sequences are needed. Strict
history-before-row ordering is mandatory (`RULES.md` §4).

---

## 7. Sequential Recommendation

**Problem addressed:** temporal user state and order effects.

**Mechanism:** SASRec applies causal self-attention over the interaction sequence;
BERT4Rec uses masked-item prediction over a bidirectional encoder (bidirectional training
requires care to stay causal at scoring time).

**Assumptions:** sequence order is meaningful and reliably recorded.

**Implementation notes:** tied timestamps are not ordered — 5.60% of validation rows sit
in non-unique user/timestamp groups (`constraints.md` C13), so a "previous item" is not
always well defined. Runtime and sequence-length truncation dominate cost.

---

## 8. Multi-Task Learning

**Problem addressed:** exploiting several feedback signals when only one is scored.

**Mechanism and structures:**
- **Shared-bottom** — one shared representation, per-task heads. Cheapest; most exposed
  to interference.
- **MMoE** — a set of expert subnetworks with per-task gating over them.
- **PLE** — separates shared experts from task-specific experts explicitly, layer by layer.

**Assumptions:** tasks share exploitable structure, and auxiliary labels are dense enough
to train their heads.

**Implementation notes:** permitted by the official rules (`constraints.md` C4). Auxiliary
density spans two orders of magnitude here — click ~46%, like ~1.8%, follow ~0.1%
(`research/data_profile.md` §7) — so head-level loss weighting and the sparse-task
contribution are design decisions with real consequences. **Negative transfer / seesaw is
the standard failure mode** and must be measured, not assumed away. The feedback columns
are auxiliary *targets* only; same-row use as inputs is forbidden (`constraints.md` C3).

---

## 9. ESMM-Style Funnel Modelling

**Problem addressed:** sample-selection bias when one label is only observed downstream
of another.

**Mechanism:** model the full-space product of a funnel (e.g. impression → click →
conversion) so that the downstream task is trained over the whole impression space.

**Assumptions:** a genuine funnel/ordering exists between the labels.

**Implementation notes:** this task is not classic CVR. The official appendix (A.2)
settles the funnel question directly: in KuaiRand the scored label `long_view` is logged
on **every impression**, not only on clicked ones, so classic sample-selection bias does
**not** apply here and the ESMM correction it was designed for has no target. Data
sparsity still applies, and the official text names the multi-feedback structure as a
legitimate reason to use other signals as auxiliary tasks. So the transferable part is
the sharing across feedback tasks, not the funnel correction.

---

## 10. Watch-Time Modelling / CWM

**Problem addressed:** duration bias — watch time is mechanically truncated by video
length, so naive regression conflates interest with duration.

**Mechanism:** treat completed plays as censored observations and use one-sided /
censored-regression losses rather than squared error (the CWM paper's contribution:
*Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time*,
https://github.com/hyz20/CWM).

**Assumptions:** the censoring mechanism is known and duration is observed.

**Implementation notes:** CWM is research code pinned to `torch==1.6.0`, and it optimises
counterfactual watch time against its own reconstructed `long_view2` label — its metric
is **not** this competition's metric. Use as a conceptual reference, not a drop-in.
`play_time_ms` may only be an auxiliary target or strictly historical feature
(`constraints.md` C3).

---

## 11. Recency and Temporal Weighting

**Problem addressed:** distribution shift between training data and the scored period.

**Mechanism:** weight training rows by age, decay old data, restrict the window, or add
explicit time features.

**Assumptions:** the shift is monotone in time and the recent past resembles the scored
period more than the distant past.

**Implementation notes:** that assumption is **not** settled here. Measured drift is
multidimensional and points in different directions: validation is closer to *early*
train in target rate and duration, and closer to *late* train in tab distribution and
volume (`constraints.md` C14). Train volume also falls 13.9x across the window, so a
recency window and a sample-size reduction are confounded.

---

## 12. Historical Aggregates and Target Encoding

**Problem addressed:** turning a user's or item's past behaviour into dense features.

**Mechanism:** counts, rates, and ratios computed over prior interactions — prior user
click rate, prior item engagement rate, prior user–author affinity, repeat-exposure
counts. Target encoding replaces a category with a smoothed statistic of the label.

**Assumptions:** the statistic is computed only from information available before the
scored row.

**Implementation notes:** the ordering rule is non-negotiable — build from strictly
earlier history, score, then update (`RULES.md` §4), with tied timestamps excluded from
"earlier". Low-count categories need smoothing toward a prior; the official item-popularity
baseline uses a prior weight of 20 as one concrete example. Out-of-fold or
time-sliced construction is the standard defence against target leakage.

---

## 13. Counterfactual / Off-Policy Evaluation

**Problem addressed:** logged feedback reflects what the deployed policy chose to show.

**Mechanism:** propensity estimation, inverse propensity scoring, doubly-robust
estimators, off-policy evaluation against a randomised-exposure log.

**Assumptions:** propensities are estimable and bounded away from zero; the randomised
log is genuinely randomised.

**Implementation notes:** the random log's exposure mechanism is a uniform replacement
from the 7,583-item pool with a fixed **undisclosed** probability, which limits exact
propensity reconstruction. Only the 2022-04-22..28 slice is usable in development
(`constraints.md` C18, `RULES.md` §1a); it is distributionally distinct from standard
traffic (0.08056 vs 0.31328 positive rate) and almost pair-disjoint from it.

---

## 14. Cold-Start and Unseen-Identifier Handling

**Problem addressed:** identifiers at scoring time that were unseen or barely seen at
training time.

**Mechanism:** UNK slots (the official encoder's approach), hashing tricks, count-based
backoff to coarser granularity, or embedding regularisation toward a prior.

**Assumptions:** a coarser level exists that generalises when the fine level does not.

**Implementation notes:** relevant given the structure in `constraints.md` C12 — entities
are almost all warm (98–99.9%), but the exact user–video relationship is novel for
98.373% of validation pairs. Backoff must be a genuine feature-space fallback, never a
lookup keyed on membership in the validation or evaluation split (`RULES.md` §2).

---

## 15. Ensembling and Rank Aggregation

**Problem addressed:** variance reduction and combining complementary models.

**Mechanism:** average scores, average ranks, or fit a combiner. For within-user ranking,
rank averaging avoids the scale-calibration problem that score averaging inherits.

**Assumptions:** component models make partly independent errors.

**Implementation notes:** combination weights are model selection and must be fitted on
validation, never on evaluation labels (`RULES.md` §1). Each component multiplies
iteration cost against the 6-hour / 50-iteration budget.

---

## 16. Experimental Methodology Under Seed Noise

**Problem addressed:** distinguishing a real effect from run-to-run variation.

**Mechanism:** matched seeds across configurations, paired deltas rather than raw means,
population std reported alongside, and a decision threshold fixed in advance.

**Assumptions:** seeds are the dominant noise source and configurations are otherwise
identical.

**Implementation notes:** the reference scales for this benchmark are a published seed
std of **0.0008** and a convergence epsilon of **0.002** (`PROBLEM.md` §7,
`constraints.md` C8). A single run cannot establish anything; the audit's own controlled
results used 3–5 matched seeds and still landed several deltas below epsilon. Budget the
seed cost into the iteration plan.

---

## 17. Official Source Material

**Resource policy.** Any open-source library or framework is in scope — the official
text names PyTorch, RecBole, TorchRec and LightGBM — as are any papers, public
solutions, and pretrained weights. The single hard rule is no external training data
(`RULES.md` §1b). Methods below that need a tensor framework or a GBDT library are
therefore admissible; the Starter Kit's numpy-only footprint is a property of the
reference implementation, not a constraint on the agent.

**Competition and starter-kit sources (authoritative):**

- `source/starter-kit/README.md` — task definition, published baseline ladder, and the
  organizer's own two sections: *already measured, no gain* (static feature stuffing;
  embedding width; zero contribution of pure user-side first-order terms) and *where the
  headroom probably is*. The latter is presented in **the organizer's judged order of
  promise**: (1) loss/objective mismatch, (2) user history sequences, (3) multi-task,
  (4) watch-time modelling, (5) alternative model family, (6) time features and drift,
  (7) unbiased validation via the random log. That ordering is the organizer's, and it is
  quoted here because it is official source material — it is not a plan handed to the
  agent, and the agent is expected to form and justify its own priorities.
- `source/starter-kit/evaluate.py` — the metric definition; the scoring authority.
- `source/starter-kit/baseline_scores.json` — published scores, seed variance, and the
  convergence parameters.
- `source/starter-kit/ablation_features.py` — reproduction of the static-feature result
  (note the 8/13 field correction in `DATA_GUIDE.md` §8).
- Official KuaiRand documentation (https://github.com/chongminggao/KuaiRand) — dataset
  description, the video-statistic aggregation description, the random-intervention
  mechanism, and the note that KuaiRand-Pure has incomplete sequential logs.

**Autonomous-agent references from the official challenge material:**

1. MLE-Bench — benchmark for autonomous ML engineering.
2. AIDE — LLM-driven code exploration for ML tasks.
3. AI Scientist-v2 — agentic scientific exploration loop.
4. CWM — duration-bias / watch-time modelling.

**Starter-level recommender references from the official appendix:**

- Google Machine Learning Crash Course — Recommendation Systems overview.
- Wang Shusen — Recommender Systems lecture series.

---

## 18. Adding to This File

When adding a reference, record:

- the problem it addresses
- its mechanism
- its assumptions
- implementation and runtime considerations

Do not write "the agent should use this next", "recommended", or "most promising" in this
project's own voice. Do not delete a method because one implementation of it
underperformed — record that result in `constraints.md` with its exact scope and leave
the method described here.


# KuaiRand-Pure — Consolidated Verified Data Profile

> Compact final facts only. Local outcome statistics use train/validation only. For
> interpretation and unresolved semantics, see `PRE_AUDIT.md`; for the reviewed
> evidence the autonomous agent may rely on, see `../context/constraints.md`.
>
> Integrity scope: no value below is derived from evaluation/test labels or from
> evaluation-period outcomes. Evaluation-window figures are date-only row counts.
> Every GAUC weight share uses the official denominator — positives belonging to
> mixed-label users only.

## 1. Official task and split

| Item | Definition / scope | Value |
|---|---|---|
| Task | Official Starter Kit | Within-user ranking over logged impressions |
| Target | Official Starter Kit | `long_view` |
| Metrics | Official `evaluate.py` | GAUC, nDCG@5 |
| Primary | Official `evaluate.py` | `(GAUC + nDCG@5) / 2` |
| Train | Official dates / reproduced rows | 2022-04-08..21 / 1,141,112 rows |
| Train dates represented | Raw train file | 2022-04-09..21; 13 dates; 0 rows on 04-08 |
| Validation | Official dates / reproduced rows | 2022-04-22..28 / 124,909 rows |
| Evaluation | Official dates / date-only row count | 2022-04-29..05-08 / 170,588 rows |

Standard-log files do not align with splits: `log_standard_4_22_to_5_08_pure.csv`
holds 295,497 rows spanning both the validation (124,909) and evaluation (170,588)
windows. Split by `date`.

GAUC includes only mixed-label users and weights each by its positive count. nDCG@5 averages all users equally; all-negative users receive 0 and all-positive users are ranking-invariant.

Official published validation reference ladder (`baseline_scores.json`):
random 0.4834, item popularity 0.5807, FM baseline 0.6016, oracle 0.8484.
Convergence: epsilon 0.002, N 3, against a published seed std of 0.0008.

## 2. Official FM validation reproduction

| Metric | Published validation | Reproduced seed 0 |
|---|---:|---:|
| GAUC | 0.6674 | 0.667133 |
| nDCG@5 | 0.5357 | 0.535806 |
| Primary | 0.6016 | 0.601470 |

Fields: `user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`; k=16; lr=0.001; batch=8,192; max 40 epochs; patience 4. Seed 0 best epoch 7, stop epoch 11. Five-seed validation primary: mean 0.60157, population std 0.00032. Published generic seed std: approximately 0.0008.

## 3. Cardinality, coverage, and missingness

| Metric | Scope | Train | Validation |
|---|---|---:|---:|
| Rows | Standard logs | 1,141,112 | 124,909 |
| Users | Standard logs | 26,210 | 22,377 |
| Videos | Standard logs | 7,538 | 5,951 |
| Authors | Standard logs + basic-video join | 6,482 | 5,315 |
| Tabs | Standard logs | 15 | 15 |

| Side table | Rows | Development-entity coverage |
|---|---:|---:|
| User features | 27,285 | 100% users |
| Video basic | 7,583 | 100% videos |
| Video statistics | 7,583 | 100% videos |

Logs and video-statistic fields have 0% missingness. Selected missingness: user `onehot_feat4` 3.2032%; user `onehot_feat12..17` 2.6168% each; basic `video_duration` 3.1518%; `music_type` 2.6770%; `tag` 1.2660%. Basic `visible_status` has cardinality 1.

## 4. Activity and validation composition

| Metric | Population | Value |
|---|---|---:|
| Train impressions/user median / p90 / p99 / max | All 26,210 train users | 31 / 97 / 207 / 809 |
| Validation impressions/user median / p90 / p99 / max | All 22,377 validation users | 4 / 12 / 26 / 74 |
| Prior train interactions median / mean / p90 | Validation users | 35 / 47.42 / 103 |

| Validation user type | Users | % users | Rows | % rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.321% | 21,807 | 17.458% |
| All positive | 2,663 | 11.901% | 4,540 | 3.635% |
| Mixed / movable | 12,929 | 57.778% | 98,562 | 78.907% |
| Single impression | 3,917 | 17.505% | 3,917 | 3.136% |

Validation oracle: GAUC 1.0000, nDCG@5 0.6968, primary 0.848393.

## 5. Train→validation overlap and repeat structure

| Metric | Definition | Value |
|---|---|---:|
| Validation users seen | Unique IDs | 21,955 / 22,377 (98.114%) |
| Validation videos seen | Unique IDs | 5,944 / 5,951 (99.882%) |
| Validation authors seen | Unique IDs | 5,310 / 5,315 (99.906%) |
| Validation user–video pairs seen | Unique pairs | 1,974 / 121,337 (1.627%) |
| Validation user–author pairs seen | Unique pairs | 4,081 / 120,885 (3.376%) |
| Validation raw user–tag-string pairs seen | Missing tag is one category | 61,405 / 90,121 (68.14%) |
| Validation parsed-token pairs seen | Multi-token parsing | 68,316 (71.913%) |
| Validation rows with a prior parsed tag token | Multi-token parsing | 78.413% |

| Pair | Scope | Repeated unique pairs | Rows in repeated pairs |
|---|---|---:|---:|
| User–video | Train | 4.130% | 8.194% |
| User–author | Train | 5.913% | 11.750% |
| Raw user–tag string | Train; explicit missing category | 51.77% | 84.98% |
| Raw user–tag string | Validation; explicit missing category | 24.45% | 45.49% |

Full basic-video file: 5,661/6,510 authors (86.96%) have exactly one video; median 1, mean 1.165, max 26. Video→author mapping is functional. For train/validation-observed videos only: 5,647/6,487 (87.051%), max 24.

## 6. Corrected metric buckets

Activity tiers are based on train counts among warm validation users: Cold 0, T1 1–17, T2 18–36, T3 37–65, T4 66+. GAUC weight denominator is 34,592 positive rows from mixed-label users only.

| Tier | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 1.69% |
| T1 | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 14.67% |
| T2 | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 21.35% |
| T3 | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 27.50% |
| T4 | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 34.79% |

| List length | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.00% |
| 2–3 | 6,218 | 15,323 | 0.6472 | 0.5413 | 0.6086 | 10.27% |
| 4–5 | 4,119 | 18,326 | 0.6645 | 0.6185 | 0.7492 | 16.36% |
| 6–10 | 5,225 | 39,587 | 0.6756 | 0.5913 | 0.8536 | 36.39% |
| 11–20 | 2,346 | 32,609 | 0.6677 | 0.5037 | 0.9182 | 27.08% |
| 21+ | 552 | 15,147 | 0.6596 | 0.3934 | 0.9420 | 9.90% |

`*` Empty GAUC denominator; official evaluator returns 0.5.

Train activity vs validation list length: Spearman rho 0.4620. T3/T4 × list 6+ contains 5,680 users (25.38%), 64,133 rows (51.34%), 50.79% of official GAUC weight, and 51.72% of the seed-0 baseline-to-oracle primary gap.

## 7. Feedback diagnostics

Current-row values below are diagnostics only and are forbidden as current-row inputs.

| Signal | Train mean | Validation mean | Validation Pearson r with `long_view` |
|---|---:|---:|---:|
| `is_click` | 0.46345 | 0.44383 | 0.7515 |
| `is_like` | 0.01868 | 0.01797 | 0.0949 |
| `is_follow` | 0.00101 | 0.00130495 (163 rows) | 0.0253 |
| `is_comment` | 0.00257 | 0.00233 | 0.0587 |
| `is_forward` | 0.00100 | 0.00078 | 0.0245 |
| `is_hate` | 0.00042 | 0.00062 | −0.0038 |
| `is_profile_enter` | 0.02539 | 0.01945 | 0.1271 |
| `play_time_ms` | 23,260.5 | 21,486.8 | 0.6319 raw |

Validation click/play-time inter-correlation: 0.5167.
Zero-value rates, stated by scope: `profile_stay_time` 99.994% (validation) /
99.989% (train); `comment_stay_time` 95.542% (validation) / 94.564% (train).

## 8. Strict train-derived history

| Metric | Scope | Value |
|---|---|---:|
| Users with ≥1 / ≥5 / ≥10 prior interactions | Validation users | 98.114% / 92.854% / 85.168% |
| Users with ≥1 / ≥5 / ≥10 prior clicks | Validation users | 96.157% / 82.531% / 66.309% |
| Users with ≥1 / ≥5 / ≥10 prior likes | Validation users | 23.229% / 4.683% / 2.239% |
| Rows with prior same video | Validation rows | 1.624% |
| Rows with prior same author | Validation rows | 3.381% |
| Rows with prior parsed tag token | Validation rows | 78.413% |
| Rows with strictly earlier same-user validation timestamp | Availability diagnostic | 81.57% |
| Rows in non-unique user/timestamp groups | Validation | 5.60% |

The 81.57% statistic is not a validated online feature protocol.

## 9. Video features and statistics

| Metric | Scope | Value |
|---|---|---:|
| Logged duration exact match | Rows where basic duration is nonmissing | 100% |
| Basic duration missing | Full basic table | 3.1518% |
| Numeric statistic pairs with |Spearman|≥0.95 | Full statistic table | 54 |
| `like_cnt` mean / median | Full statistic table | 230.75 / 57.54 |
| `long_time_play_cnt` mean / median | Full statistic table | 3,687 / 978 |
| `comment_cnt` mean / median | Full statistic table | 12.93 / 2.46 |
| Reconstructed/observed impression ratio | `(show_cnt × counts)` / train+valid standard impressions | median 11,465×; p10 5,248×; p90 38,199× |

`show_cnt × counts` is near-integer for every video, consistent with the official
description of these fields as per-day, per-scenario averages over one month with
`counts` as the number of component statistics. The documentation does not state the
window's endpoints or the source population.

Aggregation population, exact calendar window, and causal safety: **INCONCLUSIVE**.
No causal-admissibility claim is made in either direction. See `../context/DATA_GUIDE.md`
§6 for the semantic caveat an implementer needs before using these fields.

| Marginal association | Scope | Value |
|---|---|---:|
| Smoothed long-time-play/show ratio, Pearson r with `long_view` | Validation | 0.302 |
| Same ratio, bottom / top quintile `long_view` rate | Validation | 0.105 / 0.505 |
| Like-ratio quintile trend | Validation | rises through Q4, dips at Q5 — **not** monotonic |

| Fixed ratio standalone score | Validation primary | Delta vs train item popularity (0.580722) |
|---|---:|---:|
| Long-time-play/show | 0.580378 | −0.000344 |
| Valid-play/show | 0.570874 | −0.009848 |
| Complete-play/show | 0.550128 | −0.030594 |
| Play/show | 0.540600 | −0.040122 |

## 10. Temporal profile

| Period | Rows | Rows/day | `long_view` rate | Mean duration |
|---|---:|---:|---:|---:|
| Early train, 04-09..14 | 891,418 | 148,570 | 0.33228 | 98,553 ms |
| Late train, 04-15..21 | 249,694 | 35,671 | 0.35211 | 95,477 ms |
| Validation, 04-22..28 | 124,909 | 17,844 | 0.31328 | 102,820 ms |

Peak train volume: 278,835 rows on 04-11. Final train day: 20,021 rows on 04-21. Ratio: 13.9×. A single early-vs-late similarity verdict is **INCONCLUSIVE**.

## 11. Random-exposure validation slice

| Metric | Permitted scope | Value |
|---|---|---:|
| Total random rows | Date-only count | 1,186,059 |
| Validation-date rows | 04-22..28 outcomes/features | 288,338 |
| Evaluation-date rows | 04-29..05-08 date-only count | 897,721 |
| Validation-slice users / videos | 04-22..28 | 19,091 / 7,546 |
| Validation-slice `long_view` rate | 04-22..28 | 0.08056 |
| Standard-validation `long_view` rate | 04-22..28 | 0.31328 |
| Shared random/standard validation pairs | Unique pairs, 04-22..28 | 17 / 288,328 (0.006%) |

No evaluation-period random-log outcome or feature is included.

## 12. Controlled FM evidence

| Configuration | Seeds | Primary mean ± population std | Paired delta | Final classification |
|---|---:|---:|---:|---|
| Base five fields | 3 | 0.601440 ± 0.000275 | Reference | HARD FACT |
| Remove `tab` | 3 | 0.585538 ± 0.000429 | −0.015903 ± 0.000467 | STRONG NEGATIVE EVIDENCE against removal |
| Remove `dur_bucket` | 3 | 0.600849 ± 0.000225 | −0.000591 ± 0.000156 | INCONCLUSIVE |
| Remove `author_id` | 5 | 0.602889 ± 0.000451 | +0.001316 ± 0.000426 | WEAK NEGATIVE EVIDENCE against dual-ID FM |
| Remove `video_id` | 5 | 0.602654 ± 0.000307 | +0.001082 ± 0.000585 | WEAK NEGATIVE EVIDENCE against dual-ID FM |
| 8-field item-static | 3 | 0.601108 ± 0.000461 | −0.000332 ± 0.000205 | INCONCLUSIVE |
| 13-field full static | 3 | 0.599930 ± 0.000523 | −0.001510 ± 0.000792 | STRONG NEGATIVE EVIDENCE, exact formulation |

FM width means (3 seeds each): k=8 0.60111 ± 0.00080; k=16 0.60144 ± 0.00027;
k=32 0.60146 ± 0.00069; k=64 0.60099 ± 0.00044. Simple width scaling is STRONG NEGATIVE
EVIDENCE for meaningful gain in this FM.

Learning rate (3 seeds each, all other settings official): lr=0.0003 0.60179 ± 0.00011;
lr=0.0005 0.601776 ± 0.000280; lr=0.001 0.60144 ± 0.00027; lr=0.002 0.601364 ± 0.000826;
lr=0.003 0.60009 ± 0.00084; lr=0.01 0.59709 ± 0.00053. WEAK NEGATIVE EVIDENCE for the
tested high rates, principally the clear lr=0.01 degradation; differences among nearby
rates are INCONCLUSIVE.

## 13. Engineering observations

| Observation | Scope | Value |
|---|---|---:|
| Cold baseline | Reviewer rerun | ~57.5 s |
| Cache reload | Same rerun; bit-identical arrays | 0.018 s |
| Cold baseline | Separate fingerprinted implementation | 78.52 s |
| Fingerprint + cache read | Separate implementation | 1.384 s |
| Bare timeout overrun | Windows inherited-pipe child/grandchild; timeout=3 s | 30.13 s elapsed |
| Comment-only implementation files | `harness/`, `pipeline/`, `agent/` | 15 |

Timings are run- and implementation-specific.


# Repo representation and column access

Only the fields that are unique to this repo. The tier, invariance, split-size and
baseline-by-tier figures that `data_profile.json` also carries are omitted here because
the reviewed profile above already states them, cell for cell, and duplicating ~9,000
characters of agreeing numbers costs tokens and invites the model to treat two copies as
two sources. The full generated file remains at `context/data_profile.json`.

```json
{
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
  "reference_seed_std": 0.0008,
  "measured_seed_std_valid_population": 0.00032,
  "note": "Use the published 0.0008 as the noise reference: it is the organizer-reported TEST seed std. The locally measured validation population std of 0.00032 is narrower, and the pre-audit review ruled that treating the narrower figure as the reference is over-confident. Against 0.0008, eps = 0.002 is about 2.5 sigma, which is the derivation the organizers give. Read a delta under roughly 0.0008 as indistinguishable from a seed, and one under 0.002 as below the practical threshold set by the competition."
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