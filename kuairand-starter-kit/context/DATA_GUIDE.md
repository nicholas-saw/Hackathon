# KuaiRand-Pure — Data Guide

> Purpose: compact map of the available data so agents do not waste tokens rediscovering
> basic file structure.
> This file describes **what exists and what it means**. It is not a list of recommended
> features, and it contains no experiment results — those live in
> `context/constraints.md` (established evidence) and `research/data_profile.md`
> (verified measurements).
>
> Schemas, row counts, and cardinalities below were verified against the files in
> `source/KuaiRand-Pure/data/`.

## 1. Source Files

```text
source/KuaiRand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv     19 cols, 1,141,112 rows   (train window)
├── log_standard_4_22_to_5_08_pure.csv     19 cols,   295,497 rows   (valid + eval)
├── log_random_4_22_to_5_08_pure.csv       19 cols, 1,186,059 rows   (random exposure)
├── user_features_pure.csv                 31 cols,    27,285 rows
├── video_features_basic_pure.csv          12 cols,     7,583 rows
└── video_features_statistic_pure.csv      52 cols,     7,583 rows
```

The raw dataset is read-only (`RULES.md` §3).

File boundaries are **not** split boundaries: the second standard log holds both the
validation window (124,909 rows) and the evaluation window (170,588 rows). Always split
by the `date` column, never by which file a row came from.

---

## 2. Standard Interaction Logs

Both standard logs and the random log share one 19-column schema, in file order:

```text
user_id, video_id, date, hourmin, time_ms,
is_click, is_like, is_follow, is_comment, is_forward, is_hate,
long_view, play_time_ms, duration_ms,
profile_stay_time, comment_stay_time, is_profile_enter,
is_rand, tab
```

### 2.1 Available before the impression outcome

| Column | Role | Notes |
|---|---|---|
| `user_id` | identity | 26,210 distinct in train, 22,377 in validation |
| `video_id` | identity | 7,538 distinct in train, 5,951 in validation |
| `date` | context | `YYYYMMDD` integer; the official split key |
| `hourmin` | context | time-of-day; not loaded by the official `data.py` |
| `time_ms` | context | event timestamp; ties occur within a user |
| `duration_ms` | item property | video length; the source of the baseline `dur_bucket` |
| `tab` | context | 15 distinct values (0–14) in both train and validation; highly concentrated — tab 1 alone is about 73% of train rows |
| `is_rand` | provenance flag | **constant 0** in both standard logs and **constant 1** in the random log; it distinguishes files and carries no within-file information |

### 2.2 Main target

- `long_view` — binary 0/1. The only scored label.

### 2.3 Post-impression outcome columns

These are concurrent outcomes of the *same* impression as `long_view`:

```text
is_click, is_like, is_follow, is_comment, is_forward, is_hate,
play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter
```

**Semantic rule.** The current row's post-impression value must never be an input for
that same row's `long_view` prediction — including hidden inside a derived
"engagement score". They may be used as auxiliary *targets*, as *historical* features
built strictly from earlier rows, or as diagnostics. See `RULES.md` §4 and
`constraints.md` C3.

Density varies enormously across these columns: click covers roughly half of all rows,
while follow, forward, and hate are each well under 1%, and `profile_stay_time` is
almost entirely zero. Verified prevalence figures are in `research/data_profile.md` §7.

### 2.4 What the official loader actually exposes

`source/starter-kit/data.py` reads only `date`, `user_id`, `video_id`, `tab`,
`duration_ms`, and `long_view` — plus `author_id` joined from the basic video file.
Every other column (`hourmin`, `time_ms`, the feedback columns, `is_rand`) exists in the
CSV but is **not** in the loader's row tuple. Reaching them means reading the raw CSVs
from the editable pipeline; `data.py` itself is read-only.

---

## 3. Official Time Splits

```text
train:            2022-04-08 .. 2022-04-21      (rows only on 04-09..04-21, 13 dates)
validation:       2022-04-22 .. 2022-04-28      (7 dates)
evaluation/test:  2022-04-29 .. 2022-05-08      (10 dates)
```

Do not change the date boundaries. Every train timestamp precedes every validation
timestamp, so the whole train window is legitimate history for validation rows.

---

## 4. User Feature File

`user_features_pure.csv` — 27,285 rows, one per user, 31 columns:

```text
user_id,
user_active_degree, is_lowactive_period, is_live_streamer, is_video_author,
follow_user_num, follow_user_num_range,
fans_user_num,   fans_user_num_range,
friend_user_num, friend_user_num_range,
register_days,   register_days_range,
onehot_feat0 .. onehot_feat17
```

**All of these are static per user** — one row per user, no time dimension. Several
appear in both a raw-count and a bucketed `_range` form, so those pairs are redundant
with each other by construction.

Coverage: 100% of train and validation users.
Missingness is localised: `onehot_feat4` 3.2032%; `onehot_feat12`..`onehot_feat17`
2.6168% each; the remaining columns are complete.

### Structural note

Because scoring ranks impressions **within** a user, any term that is constant across
that user's impressions cannot change their order. A purely additive first-order
user-side feature therefore has exactly zero effect on the metric (organizer-confirmed;
see `constraints.md` C2). User-side information can still act through interactions with
item/context features, through shared representations, or through cold-user behaviour —
this is a statement about additive terms, not about user information in general.

---

## 5. Basic Video Features

`video_features_basic_pure.csv` — 7,583 rows, one per video, 12 columns:

```text
video_id, author_id, video_type, upload_dt, upload_type, visible_status,
video_duration, server_width, server_height, music_id, music_type, tag
```

All fields are **static per video**. Coverage: 100% of train and validation videos.

| Field | Cardinality / note |
|---|---|
| `video_id` | 7,583 (primary key) |
| `author_id` | 6,510 distinct; the video→author map is **functional** (each video has exactly one author) and heavily one-to-one — see `constraints.md` C12 for the verified quantitative structure |
| `video_type` | 3 |
| `upload_dt` | 3 |
| `upload_type` | 14 |
| `visible_status` | **cardinality 1** — constant across the file, so it carries no information |
| `video_duration` | 3.1518% missing; where present it matches the log's `duration_ms` exactly |
| `server_width` / `server_height` | 154 / 120 |
| `music_id` | 7,167 — nearly as high-cardinality as `video_id` itself |
| `music_type` | 5 distinct; 2.6770% missing |
| `tag` | 110 distinct raw strings; 1.2660% missing |

### Tag semantics — two different representations

`tag` is a **raw string** field that can encode several tokens. Two representations
appear in the audit record and they are **not interchangeable**:

- **Raw tag string**, with missing treated as one explicit category — 110 distinct values.
- **Parsed tag tokens**, splitting each string into individual tokens.

Any tag statistic must state which representation produced it; the two yield materially
different overlap numbers (`research/data_profile.md` §5).

---

## 6. Video Statistic Features

`video_features_statistic_pure.csv` — 7,583 rows, one per video, 52 columns, **0%
missing**. Columns cover impressions (`show_cnt`, `show_user_num`), plays (`play_cnt`,
`complete_play_cnt`, `valid_play_cnt`, `long_time_play_cnt`, `short_time_play_cnt`,
`play_duration`, `play_progress`), and social actions (`like_*`, `comment_*`,
`follow_*`, `share_*`, `download_*`, `collect_*`, `report_*`, `reduce_similar_*`, plus
their cancel variants), with a leading `counts` column.

### Aggregation caveat — read before using any of these

1. **Values are averages, not raw totals.** The official KuaiRand documentation
   describes them as per-day, per-scenario averages over one month, with `counts` being
   the number of component statistics. `show_cnt` multiplied by `counts` is near-integer
   for every video, which is consistent with that description.
2. **The window's endpoints are undocumented.** The documentation does not state which
   month, nor where the window sits relative to a scored impression. This is unresolved,
   not merely unverified. It is an open question rather than an established constraint,
   so it is deliberately absent from `constraints.md`; the measurements behind it are in
   `research/data_profile.md` §9.
3. **The population is undocumented.** Reconstructed impression volume exceeds the
   observed train+validation impressions of the same videos by a median factor in the
   thousands, so these counts describe a much larger population than this dataset's logs.
4. **Heavy internal redundancy.** 54 numeric field pairs have absolute Spearman
   correlation of at least 0.95 (`like_cnt` vs `like_user_num` = 0.999865, and so on) —
   the 51 statistic columns carry far fewer than 51 independent signals.

Consequence: whether these fields are causally admissible for scoring an April 22–28
impression is **not established**. Establishing or bounding it is open work — neither a
settled ban nor a clearance.

---

## 7. Random Exposure Log

`log_random_4_22_to_5_08_pure.csv` — 1,186,059 rows, same 19-column schema, `is_rand`
constant 1.

Under the random-exposure intervention, a normally recommended list item is replaced by
a uniformly sampled item from the 7,583-item pool with a fixed, undisclosed probability.
This is the dataset's support for counterfactual / off-policy analysis.

### Exact date coverage (verified from the `date` column alone)

| Portion | Dates | Rows | Development status |
|---|---|---:|---|
| Validation window | 2022-04-22 .. 04-28 | 288,338 | outcomes and features **permitted** |
| Evaluation window | 2022-04-29 .. 05-08 | 897,721 | **date-only counts; all other columns forbidden** |

Filter by `date` **before** materialising any other column (`RULES.md` §1a). The
validation slice is distributionally distinct from standard validation traffic and
almost pair-disjoint from it; the verified figures are in `research/data_profile.md` §11.

---

## 8. Official Starter Kit

```text
source/starter-kit/
├── README.md              task spec; the definition table is frozen
├── data.py                official date splitting + baseline row construction
├── evaluate.py            GAUC, nDCG@5, primary — the scoring authority
├── baseline.py            official reference FM, item popularity, random
├── baseline_scores.json   published scores, seed variance, convergence parameters
├── submit.py              submission writer/validator, row alignment checks
└── ablation_features.py   organizer's record of the static-feature experiment
```

All are competition truth and read-only. `evaluate.py` in particular must never be
modified, reimplemented, wrapped, or shadowed.

### `evaluate.py` semantics worth knowing before designing anything

- Groups rows by `user_id` and sorts each user's rows by descending score.
- **GAUC:** only users with `0 < positives < impressions` contribute; each contributes
  weighted by its positive count. AUC is Mann-Whitney U with tie correction.
- **nDCG@5:** every user contributes with equal weight; gain is `2^rel − 1`, which is
  the identity for binary labels. A user with no positives contributes 0.
- **primary:** `(GAUC + nDCG@5) / 2`.
- With an empty GAUC denominator the function returns 0.5 — a fallback, not a score.

### `ablation_features.py` field counts

The script builds three configurations. Its printed labels are stale; the actual field
lists are:

- `base` — 5 fields (the current kit)
- `item` — **8** fields (base + `music_id`, `video_type`, `upload_type`)
- `cwm13` — **13** fields (item + `follow_user_num_range`, `register_days_range`,
  `fans_user_num_range`, `friend_user_num_range`, `user_active_degree`)

Use 8/13 when referring to these configurations; the "9/14" labels in older notes are
incorrect.

---

## 9. Baseline Feature Fields

The official FM baseline uses five fields:

```text
user_id, video_id, author_id, tab, dur_bucket
```

`dur_bucket` is `duration_ms` discretised into 10 quantile bins whose edges are fitted on
train only. Unseen categorical values fall into a per-field UNK slot.

---

## 10. Questions This Guide Does Not Answer

This file says what the data *is*. It does not say which fields help.

Verified measurements — overlap, redundancy, history coverage, feedback density, metric
headroom by segment, temporal drift, runtime — are in `research/data_profile.md`.
Findings that survived review are in `context/constraints.md`. Everything else is open,
and deciding what to measure next is the agent's job.
