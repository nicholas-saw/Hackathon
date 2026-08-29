# KuaiRand-Pure — Data Guide

> Purpose: compact map of the available data so agents do not waste tokens rediscovering basic file structure.
> This is descriptive, not a list of recommended features.

## 1. Source Files

Expected raw dataset structure:

```text
KuaiRand-Pure/
└── data/
    ├── log_standard_4_08_to_4_21_pure.csv
    ├── log_standard_4_22_to_5_08_pure.csv
    ├── log_random_4_22_to_5_08_pure.csv
    ├── user_features_pure.csv
    ├── video_features_basic_pure.csv
    └── video_features_statistic_pure.csv
```

Treat the raw dataset as read-only.

---

## 2. Standard Interaction Logs

The standard logs contain interaction/impression rows.

Important columns include:

### Identity / context

- `user_id`
- `video_id`
- `date`
- `hourmin`
- `time_ms`
- `duration_ms`
- `tab`
- `is_rand`

### Main target

- `long_view`

### Other post-impression feedback

- `is_click`
- `is_like`
- `is_follow`
- `is_comment`
- `is_forward`
- `is_hate`
- `play_time_ms`
- `profile_stay_time`
- `comment_stay_time`
- `is_profile_enter`

### Important semantic rule

The current row's post-impression feedback must not be used as current-row input for `long_view`.

These signals may still be useful as:

- auxiliary targets
- historical signals from earlier rows
- diagnostics

---

## 3. Official Time Splits

From the standard logs:

```text
train:
2022-04-08 .. 2022-04-21

validation:
2022-04-22 .. 2022-04-28

evaluation/test:
2022-04-29 .. 2022-05-08
```

Do not change the date boundaries.

---

## 4. User Feature File

`user_features_pure.csv`

Contains user-level properties and bucketed/static user attributes.

Examples include:

- user activity degree
- follower/fan/friend counts or ranges
- registration-age/range features
- one-hot feature columns

### Structural note

Many user features are constant within a user.

Because evaluation is within-user ranking, a purely additive user-only first-order term cannot change that user's ordering.

This does **not** prove user information is useless. It may matter through:

- interactions with item/context features
- cold-user generalization
- multi-task/shared representations

The pre-audit should measure rather than assume.

---

## 5. Basic Video Features

`video_features_basic_pure.csv`

Examples include:

- `video_id`
- `author_id`
- `video_type`
- upload-related fields
- visibility/status fields
- video duration
- width/height
- `music_id`
- `music_type`
- `tag`

The pre-audit should measure:

- cardinality
- missingness
- overlap
- redundancy
- coverage

before using these fields.

---

## 6. Video Statistic Features

`video_features_statistic_pure.csv`

Contains aggregated video statistics.

Examples may include:

- `show_cnt`
- `play_cnt`
- `complete_play_cnt`
- `valid_play_cnt`
- `long_time_play_cnt`
- `short_time_play_cnt`
- `like_cnt`
- `comment_cnt`
- `follow_cnt`
- `share_cnt`
- other aggregated statistics

### Important caution

Before using any statistic:

1. determine what it represents
2. inspect missingness
3. inspect scale
4. establish whether its aggregation window is documented or inferable
5. check whether its use is causally defensible under the competition rules

Do not assume every statistic is safe merely because it is shipped.

---

## 7. Random Exposure Log

`log_random_4_22_to_5_08_pure.csv`

This represents randomized exposure data.

The official material notes that KuaiRand's randomized-exposure data supports counterfactual / off-policy analysis.

The pre-audit should determine:

- exact date coverage
- overlap with standard logs
- overlap with validation/evaluation periods
- safe diagnostic uses
- whether any use would introduce temporal leakage

Do not use evaluation-period labels to guide model selection.

---

## 8. Official Starter Kit

Expected reference files:

```text
starter-kit/
├── README.md
├── data.py
├── evaluate.py
├── baseline.py
├── baseline_scores.json
├── submit.py
└── ablation_features.py
```

These should be treated as competition/reference truth.

### `evaluate.py`

Defines:

- GAUC
- nDCG@5
- primary score

Do not modify.

### `data.py`

Defines official date splitting and baseline row construction.

Do not modify in place.

### `baseline.py`

Official reference FM.

### `baseline_scores.json`

Published reference scores and variance.

### `submit.py`

Submission format and row-alignment checks.

### `ablation_features.py`

Organizer-provided record/reproduction of certain static-feature experiments.

---

## 9. Baseline Feature Fields

The official FM baseline uses:

```text
user_id
video_id
author_id
tab
dur_bucket
```

`dur_bucket` is derived from `duration_ms`.

---

## 10. Pre-Audit Questions This Guide Does Not Answer

The following must be measured:

- Which fields are redundant?
- Which fields generalize from train to validation?
- How much user/video/author overlap exists?
- Which history signals have adequate support?
- Which auxiliary targets are dense enough?
- Which video statistics contain distinct information?
- Which user groups contain the largest movable metric headroom?
- How strong is temporal drift?
- What is the actual cost of each pipeline stage?

Those findings belong in `PRE_AUDIT.md` and later, after review, in `constraints.md`.
