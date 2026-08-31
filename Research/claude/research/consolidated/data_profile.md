# KuaiRand-Pure — Consolidated Data Profile

> Compact, agent-readable snapshot of verified numerical facts only. No reasoning, no
> recommendations, no unresolved claims presented as fact — see `research/consolidated/PRE_AUDIT.md`
> for interpretation, evidence classification, and scope discussion of every number below.
> Consolidated from three independently-produced audits (Claude, Gemini, GPT); see
> `research/consolidated/MERGE_WORKLOG.md` for the full crosswalk.
>
> Where sources disagreed only by scope (not a true conflict), both scoped values are given.
> Where a genuine unresolved discrepancy exists, it is marked `MINOR DISCREPANCY (unresolved)`.
> Nothing derived from evaluation/test labels appears anywhere in this file.

## 1. Task and Official Splits

```yaml
dataset: KuaiRand-Pure
task: within-user ranking over logged impressions
target: long_view
metrics: [GAUC, nDCG@5]
primary: (GAUC + nDCG@5) / 2
train_dates_declared: 2022-04-08..2022-04-21
train_dates_with_rows: 2022-04-09..2022-04-21   # zero rows dated 2022-04-08, see §9
validation_dates: 2022-04-22..2022-04-28
evaluation_dates: 2022-04-29..2022-05-08
test_labels_accessed_by_any_merged_source: false
official_source_modified: false
```

| Split | Dates | Rows |
|---|---|---:|
| Train | 2022-04-08..2022-04-21 | 1,141,112 |
| Validation | 2022-04-22..2022-04-28 | 124,909 |
| Evaluation/Test | 2022-04-29..2022-05-08 | 170,588 (date-only count in every source; no label ever read) |

## 2. Official Baseline

| Metric | Published (organizer) | GPT reproduction (seed 0) | Claude reproduction (seed 0 / 5-seed mean) |
|---|---:|---:|---:|
| GAUC | 0.6674 | 0.667133 | 0.6671 / 0.66740 |
| nDCG@5 | 0.5357 | 0.535806 | 0.5358 / 0.53574 |
| Primary | 0.6016 | 0.601470 | 0.6015 / 0.60157 |

```yaml
model: Factorization Machine
fields: [user_id, video_id, author_id, tab, dur_bucket]
k: 16
lr: 0.001
l2: 0.000001
objective: pointwise binary cross-entropy
optimizer: Adam for W/V; plain gradient step for bias
batch_size: 8192
max_epochs: 40
patience: 4
published_seed_std: 0.0008
claude_5seed_std_primary: 0.00032    # local noise floor; treat 0.0008 as the more conservative reference
```

## 3. Entity Cardinalities

| Measurement | Train | Validation | Agreement |
|---|---:|---:|---|
| Rows | 1,141,112 | 124,909 | exact, 3/3 |
| Unique users | 26,210 | 22,377 | exact, 3/3 |
| Unique videos | 7,538 | 5,951 | exact, 3/3 |
| Unique authors | 6,482 | 5,315 | exact, 3/3 |
| Unique `tab` values | 15 | 15 | exact, 3/3 |
| Unique tag strings (full `video_basic` file) | 110 | — | Gemini/GPT agree; Claude reports 111 — MINOR DISCREPANCY (unresolved) |
| Unique parsed tag tokens (comma-split) | 46 | — | GPT only |
| Unique `music_id` | 7,202 | — | GPT; Claude reports same order (7,202) |

Side-table totals: `user_features_pure.csv` 27,285 rows (100% coverage of train+valid users);
`video_features_basic_pure.csv` / `video_features_statistic_pure.csv` 7,583 rows each (100%
coverage of train+valid videos).

## 4. Missingness

| Field | Missing % |
|---|---:|
| `video_features_basic_pure.csv`: `video_duration` | 3.152% |
| `video_features_basic_pure.csv`: `music_type` | 2.677% |
| `video_features_basic_pure.csv`: `tag` | 1.266% |
| `user_features_pure.csv`: `onehot_feat4` | 3.203% |
| `user_features_pure.csv`: `onehot_feat12`-`onehot_feat17` | 2.617% each |
| Standard interaction logs (all fields) | 0% |
| `video_features_statistic_pure.csv` (all fields) | 0% |
| `video_features_basic_pure.csv`: `visible_status` | 0% missing; cardinality = 1 (constant field) |

Source agreement: exact match, 3/3.

## 5. Train → Validation Overlap

| Measurement | Value | Scope | Agreement |
|---|---:|---|---|
| Validation users seen in train | 98.11% (21,955/22,377) | — | exact, 3/3 |
| Validation videos seen in train | 99.88% (5,944/5,951) | — | exact, 3/3 |
| Validation authors seen in train | 99.91% (5,310/5,315) | — | exact, 3/3 |
| Validation user-video PAIRS seen in train | 1.63% (1,974/121,337) | unique pairs | exact, 3/3 |
| Validation user-author PAIRS seen in train | 3.38% | unique pairs | exact, 3/3 |
| Validation user-**tag** PAIRS seen in train | 68.14% | tag = single raw string | Claude, Gemini |
| Validation user-**tag** PAIRS seen in train | 71.913% (68,316 pairs) | tag = parsed comma-split tokens | GPT |
| Cold validation users | 422 (1.89%) | — | Claude, GPT |
| Cold validation videos | 7 (0.12%) | — | Claude, GPT |
| Cold validation authors | 5 (0.09%) | — | Claude |

The two user-tag figures are **different scopes, not a conflict** — see PRE_AUDIT §3.2.

## 6. Author / Video Structure

| Measurement | Value | Scope |
|---|---:|---|
| Authors with exactly 1 video | 86.96%-86.959% (5,661/6,510) | full `video_basic` file (all 3 sources agree) |
| Authors with exactly 1 video | 87.05%-87.07% (5,647/6,487, or train-only) | observed-in-logs only (Claude, GPT) |
| Videos/author — median | 1.0 | both scopes |
| Videos/author — mean | 1.16-1.165 | full table |
| Videos/author — p90 | 2 | full table |
| Videos/author — p90 | 3 | observed-only |
| Videos/author — max | 26 | full table |
| Videos/author — max | 24 | observed-only |
| Video → author functional mapping | 100% | observed videos (GPT) |

## 7. Repeat-Pair Frequency (within TRAIN)

| Pair | % pairs repeated >1x | % rows in repeated pairs | Scope |
|---|---:|---:|---|
| user-video | 4.13%-4.130% | 8.19%-8.194% | exact match, Claude/GPT |
| user-author | 5.91%-5.913% | 11.75%-11.750% | exact match, Claude/GPT |
| user-tag | 51.77% | 84.98% | raw string (Claude, review-corrected) |
| user-tag | 55.250% | 87.819% | parsed tokens (GPT) |

Within VALIDATION (Claude only): user-video pairs repeated >1x 2.90% (5.67% of rows); user-author
3.25% (6.37% of rows); user-tag (raw string) 24.45% (45.49% of rows).

## 8. Row-Level Repeat Coverage (validation rows where the user's TRAIN history contains the same entity)

| Coverage | Claude | GPT | Gemini | Scope |
|---|---:|---:|---:|---|
| Same video seen before | 1.62% | 1.624% | 1.58% | Claude/GPT match; Gemini MINOR DISCREPANCY (unresolved, ~0.04pp) |
| Same author seen before | 3.38% | 3.381% | 3.27% | Claude/GPT match; Gemini MINOR DISCREPANCY (unresolved, ~0.11pp) |
| Same tag seen before (raw string) | 73.19% | — | — | Claude only |
| Same tag seen before (parsed tokens) | — | 78.413% | — | GPT only |

By activity tier (all use each source's own tier scheme — see §11):

| Tier scheme (source) | Cold | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| Claude same-video % | 0% | 0.73% | 1.09% | 1.37% | 2.47% |
| Claude same-author % | 0% | 1.57% | 2.44% | 2.90% | 5.01% |
| Claude same-tag % | 0% | 45.80% | 67.97% | 77.59% | 86.47% |

## 9. Temporal Profile

`log_standard_4_08_to_4_21_pure.csv` has **zero rows for 2022-04-08** (13 distinct dates present,
04-09..04-21). Total train row count (1,141,112) unaffected. Exact match, 3/3.

| Date | Rows | long_view rate |
|---|---:|---:|
| 2022-04-08 | 0 | — |
| 2022-04-09 | 52,736 | 0.3362 |
| 2022-04-10 | 227,808 | 0.3409 |
| 2022-04-11 (peak) | 278,835 | 0.3330 |
| 2022-04-12 | 166,076 | 0.3322 |
| 2022-04-13 | 94,711 | 0.3171 |
| 2022-04-14 | 71,252 | 0.3195 |
| 2022-04-15 | 58,892 | 0.3634 |
| 2022-04-16 | 60,904 | 0.3770 |
| 2022-04-17 | 44,023 | 0.3644 |
| 2022-04-18 | 24,560 | 0.3308 |
| 2022-04-19 | 20,443 | 0.3229 |
| 2022-04-20 | 20,851 | 0.3114 |
| 2022-04-21 (train end) | 20,021 | 0.3146 |
| Validation (7-day avg) | 17,844/day | ~0.29-0.34 |

Peak/trough ratio within train alone: 13.9x. Source agreement: exact match, 3/3 (row counts
independently reproduced by all three; long_view-rate column from Gemini's daily table, not
independently re-verified against Claude/GPT's own per-day figures beyond the period aggregates
below, which do match).

Period aggregates (early train / late train / validation) — Claude and GPT computed different
boundary splits; both given, not merged:

| Split scheme | Period | Rows/day | long_view rate |
|---|---|---:|---:|
| Claude (early=04-09..14, late=04-15..21) | Early train | 148,570 (=891,418/6d) | 0.3323 |
| Claude | Late train | 35,671 (=249,694/7d) | 0.3521 |
| Claude | Validation | 17,844 | 0.3133 |
| GPT (early=04-09..14, late=04-15..21) | Early train | 148,570 | 0.33228 |
| GPT | Late train | 35,671 | 0.35211 |
| GPT | Validation | 17,844 | 0.31328 |

(Claude's and GPT's period boundaries and resulting numbers match to 4-5 decimals — both are the
same underlying split, independently computed.)

## 10. Metric Structure — Uniform-Label / Invariant Validation Users

| Type | Users | % Users | Rows | % Rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.32%-30.321% | 21,807 | 17.46%-17.458% |
| All positive | 2,663 | 11.90%-11.901% | 4,540 | 3.63%-3.635% |
| Mixed / movable | 12,929 | 57.78%-57.778% | 98,562 | 78.91%-78.907% |
| Single impression (⊂ above) | 3,917 | 17.50%-17.505% | 3,917 | — |

Exact match, 3/3. Local oracle (labels-as-scores): GAUC 1.0000, nDCG@5 0.6968-0.6969, primary
0.8484 — matches organizer's published `baseline_scores.json` valid-oracle to 4 decimals.

GAUC official denominator (mixed-label-user positive rows only): **34,592** (Claude).

## 11. List-Length and Activity-Tier Buckets

### List length (validation) — GAUC weight share exact match, Claude/GPT

| Length | Users | Rows | Oracle nDCG@5 | Baseline nDCG@5 (Claude/GPT) | Baseline nDCG@5 (Gemini, capped run) | GAUC weight share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.4054 | 0.4054 | 0.4054 | 0.00% |
| 2-3 | 6,218 | 15,323 | 0.6086 | 0.5413 | 0.5413 | 10.27% |
| 4-5 | 4,119 | 18,326 | 0.7492 | 0.6185 | 0.6140 | 16.36% |
| 6-10 | 5,225 | 39,587 | 0.8536 | 0.5913 | 0.5880 | **36.39%** |
| 11-20 | 2,346 | 32,609 | 0.9182 | 0.5037 | 0.4875 | 27.08% |
| 21+ | 552 | 15,147 | 0.9420 | 0.3934 | 0.4008 | 9.90% |

Overall list-length distribution: min 1, median 4, mean 5.58, p90 12, p99 26, max 74 (exact
match, 3/3).

### Activity tier — THREE non-comparable tier schemes (do not merge these rows across schemes)

**Claude** (edges 17/36/65, quartiles among users with ≥1 train row):

| Tier | Users | Rows | GAUC | nDCG@5 | Oracle nDCG@5 | Fixed-users % | GAUC weight share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold (0) | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 57.11% | 1.69% |
| T1 (1-17) | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 58.81% | 14.67% |
| T2 (18-36) | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 45.91% | 21.35% |
| T3 (37-65) | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 35.60% | 27.50% |
| T4 (66+) | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 26.31% | 34.79% |

**GPT** (edges 13/31/59, quartiles over all train users):

| Tier | Users | Rows | GAUC | nDCG@5 | Fixed-users % | GAUC weight share |
|---|---:|---:|---:|---:|---:|---:|
| Cold (0) | 422 | 1,990 | 0.6877 | 0.5305 | 57.11% | 1.69% |
| T1 (1-13) | 4,351 | 13,437 | 0.6550 | 0.5374 | 61.25% | 10.55% |
| T2 (14-31) | 5,582 | 23,310 | 0.6686 | 0.5409 | 48.01% | 19.95% |
| T3 (32-59) | 5,791 | 32,052 | 0.6624 | 0.5521 | 37.70% | 27.80% |
| T4 (60+) | 6,231 | 54,120 | 0.6720 | 0.5154 | 26.95% | 40.01% |

**Gemini** (fixed thresholds <10/10-49/50-149/150+; single-seed 12-epoch-capped FM — lower
confidence baseline run):

| Tier | Users | Rows | GAUC | nDCG@5 | Fixed-users % |
|---|---:|---:|---:|---:|---:|
| Cold (0) | 422 | 1,990 | 0.6741 | 0.5262 | 57.11% |
| T1 (<10) | 2,897 | 8,721 | 0.6475 | 0.5344 | 62.82% |
| T2 (10-49) | 11,138 | 49,716 | 0.6590 | 0.5444 | 45.97% |
| T3 (50-149) | 7,119 | 53,802 | 0.6620 | 0.5282 | 28.99% |
| T4 (150+) | 801 | 10,680 | 0.6856 | 0.4069 | 25.34% |

### Joint activity-tier × list-length (Claude only, Claude's tier scheme)

Spearman ρ = 0.4620 (all users) / 0.4677 (warm users only); Pearson r = 0.4419.

| Intersection | Users | Rows | GAUC weight | Primary gap contribution |
|---|---:|---:|---:|---:|
| T4 × list 6+ | 3,453 (15.43%) | 42,020 (33.64%) | 30.35% | 31.35% |
| T3/T4 × list 6+ | 5,680 (25.38%) | 64,133 (51.34%) | 50.79% | 51.72% |
| T2/T3/T4 × list 6+ | 7,165 (32.02%) | 78,253 (62.65%) | 64.36% | 64.36% |
| T3/T4 × list 11+ | 2,307 (10.31%) | 38,219 (30.60%) | 27.80% | 28.18% |

## 12. Feedback Signal Profile

| Signal | Train | Validation | Same-row r w/ long_view (valid, Claude/Gemini) | Same-row r w/ long_view (train, GPT) |
|---|---:|---:|---:|---:|
| `is_click` | 46.34%-46.345% | 44.38%-44.383% | 0.751 / 0.7515 | 0.7605 |
| `is_like` | 1.868%-1.87% | 1.797%-1.80% | 0.095 | 0.0992 |
| `is_follow` | 0.101%-0.10% | **0.130%** (all 3 agree) | 0.025 | 0.0250 |
| `is_comment` | 0.257%-0.26% | 0.233% | 0.059 | 0.0590 |
| `is_forward` | 0.100%-0.10% | 0.078% | 0.025 | 0.0226 |
| `is_hate` | 0.042%-0.04% | 0.062% | −0.004 | −0.0039 |
| `is_profile_enter` | 2.539%-2.54% | 1.945%-1.95% | 0.127 | 0.1461 |
| `play_time_ms` (mean) | 23,260-23,260.5 | 21,486.8-21,487 | 0.632 / 0.6319 (raw) | 0.6351 raw / 0.5960 log1p |
| `profile_stay_time` (mean) | ≈3.3 | ≈1.9 | −0.0005 | 0.0079 log1p |
| `comment_stay_time` (mean) | ≈553 | ≈460 | 0.169 | 0.2702 log1p |

`play_time_ms` additional: validation median 4,607ms (Claude), train median 4,970ms (GPT); p90
≈62,826-62,800ms; p99 ≈206,270-213,231ms; ~11.7-13.9% exactly zero (split-dependent).

Inter-feedback correlation (Claude, validation): click-play_time_ms r=0.5167; comment-comment_stay_time
r=0.3029. Two structurally distinct clusters observed: watch-related (click/play-time) vs.
active-engagement (like/follow/comment/forward).

Prior-signal availability by count threshold, validation users (GPT only):

| Prior signal | ≥1 | ≥5 | ≥10 |
|---|---:|---:|---:|
| Interactions | 98.114% | 92.854% | 85.168% |
| Clicks | 96.157% | 82.531% | 66.309% |
| Likes | 23.229% | 4.683% | 2.239% |
| Comments | 7.785% | 0.241% | 0.018% |
| Follows | 3.423% | 0.054% | 0.022% |
| Forwards | 3.365% | 0.049% | 0.013% |
| Hates | 1.028% | 0.063% | 0.031% |
| Positive-play-time rows | 97.640% | 91.053% | 82.111% |

## 13. Historical Availability

| Measurement (population: validation users' own TRAIN-side rows) | Value | Agreement |
|---|---:|---|
| ≥1 prior train interaction | 98.11%-98.114% | exact, 3/3 |
| ≥5 | 92.85%-92.854% | exact, 3/3 |
| ≥10 | 85.17%-85.168% | exact, 3/3 |
| Median prior interactions | 35 | exact, 3/3 |
| Mean prior interactions | 47.4-47.42 | Claude, GPT |
| p90 prior interactions | 103 | Claude, GPT |
| p99 / max | 216 / 809 | GPT only |

**A different population** (train-side activity across ALL 26,210 train users, not restricted to
those also in validation) — Gemini only, review-corrected:

```yaml
train_impressions_per_train_user:
  min: 1
  median: 31        # was incorrectly 35 before Gemini's own review — see PRE_AUDIT §8.2
  mean: 43.54
  p90: 97
  p99: 207           # was incorrectly ~250 before review
  max: 809
validation_impressions_per_validation_user:   # third, again-different population — matches §11 list-length distribution
  min: 1
  median: 4
  mean: 5.58
  p90: 12
  p99: 26
  max: 74
```

By activity tier (each source's own tier scheme, see §11):

| Tier scheme (source) | Cold | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|
| Claude ≥10% | 0% | 49.3% | 100% | 100% | 100% |
| GPT ≥10% | 0% | 33.42% | 100% | 100% | 100% |
| Gemini ≥10% | 0% | 0% | 100% | 100% | 100% |

## 14. Video Basic Features

| Field | Cardinality | Notes |
|---|---:|---|
| `video_type` | 3 | ~99% one value ("NORMAL") |
| `upload_type` | 14 | top: LongImport 38.6%, Web 31.9% |
| `visible_status` | 1 | constant, zero information |
| `music_type` | 5-6 nonmissing | one dominant value |
| `tag` | 110 nonmissing strings / 46 parsed tokens | see §3 discrepancy note |
| `music_id` | 7,202 | ≈1 per video |
| `video_duration` | continuous | median ≈81,171ms, p90 ≈237,830ms, 3.15% missing |

`tab` long_view-rate spread (TRAIN, Claude only): 0.42% (tab 3, n=3,574) to 61.25% (tab 10, n=80);
dominant tabs: tab 1 (834,876 rows, 38.61%), tab 0 (150,013 rows, 4.22%).
`dur_bucket` long_view-rate spread (TRAIN, Claude only): 0.273 to 0.376 across 10 buckets.

`duration_ms` (interaction logs) == `video_duration` (basic file) on 100% of nonmissing joined
rows, Spearman 1.0 (GPT only).

## 15. Video Statistics

Raw means/medians (Gemini only, review-corrected):

| Statistic | Mean | Median |
|---|---:|---:|
| `show_cnt` | 10,552 | 4,519 |
| `play_cnt` | 7,747 | 2,560 |
| `long_time_play_cnt` | 3,687 | 978 |
| `like_cnt` | **230.75** (was incorrectly 158 before review) | 57.54 |
| `comment_cnt` | **12.93** (was unmeasured before review) | 2.46 |
| `follow_cnt` | 17.41 | 3.80 |

Pairwise redundancy (GPT only): 54 field pairs have |Spearman| ≥ 0.95, e.g. `like_cnt` vs.
`like_user_num` 0.999865, `follow_cnt` vs. `follow_user_num` 0.999754, `long_time_play_cnt` vs.
`long_time_play_user_num` 0.999678, `valid_play_cnt` vs. `valid_play_user_num` 0.999499,
`play_cnt` vs. `play_user_num` 0.999000.

Aggregation-window evidence (Claude): `show_cnt × counts` near-integer for 100% of videos;
reconstructed-total / observed-train+valid-impressions ratio: median 11,465x, p10 5,248x, p90
38,199x, 0% of videos below 1x. `counts` field range: 45-181 (median 147).

Aggregation-window evidence (GPT, official documentation, verbatim): "average per day and
scenario... over one month"; exact calendar window/cutoff undisclosed.

`causal_validity_for_scored_period: unresolved` (all three sources concur; see PRE_AUDIT §9.3).

Ratio-feature evidence (α=1, β=20 smoothing):

| Ratio | Claude: Pearson r w/ long_view (valid) | GPT: standalone primary as ranker (valid) |
|---|---:|---:|
| long_time_play_cnt/show_cnt | 0.302 | 0.580378 (vs. item-popularity baseline 0.580722) |
| valid_play_cnt or complete_play_cnt/show_cnt | 0.181-0.185 | 0.550128-0.570874 |
| play_cnt/show_cnt | 0.185 | 0.540600 |
| like_cnt/show_cnt | 0.040 | 0.483741 |
| comment_cnt/show_cnt | — | 0.454772 |
| follow_cnt/show_cnt | — | 0.456476 |
| share_cnt/show_cnt | — | 0.448518 |

## 16. Random-Exposure Log (`log_random_4_22_to_5_08_pure.csv`)

```yaml
rows: 1186059                      # exact match, 3/3
date_min: 2022-04-22
date_max: 2022-05-08
rows_in_evaluation_date_range: 897721    # 75.69%; date-only count in every source, no label ever read
rows_in_validation_date_range: 288338
rows_in_train_date_range: 0        # the file's date range does not overlap train at all
```

Full-log-minus-test vs. standard train+validation (Gemini review-corrected 759→702; independently
matched by GPT):

| Measurement | Value |
|---|---:|
| Unique (user,video) pairs, full random log | 1,186,006 |
| Overlap with standard train+validation pairs | **702** / 1,186,006 (0.06%/0.0592%) |
| Random users also in standard train+valid | 97.607% |
| Random videos also in standard train+valid | 99.499% |

Validation-period-slice-only vs. standard-**validation**-only (Claude only — a different,
narrower comparison than the 702-pair figure above):

| Measurement | Value |
|---|---:|
| Random log rows in validation-date range | 288,338 |
| Shared (user,video) pairs vs. standard validation | 17 / 288,328 (0.006%) |
| Validation-period users also in train+standard-valid | 98.89% |
| Random-log videos also in standard logs | 99.50% |
| `long_view` rate, random log, validation-period rows | 8.06% (Claude only; GPT deliberately did not load labels from this file) |
| `long_view` rate, standard validation log | 31.3% |

## 17. Engineering Profile

> Each row is a fact about its own source's machine/run — not a dataset property. Do not average
> across sources.

| Stage | Claude (review-rerun) | GPT |
|---|---:|---:|
| CSV load (all files) | 2.99s | 2.88s |
| Encoding | 4.81s | 8.47s |
| FM training + epoch evals | 49.7s (11 epochs) | 66.60s (11 epochs) |
| Final evaluation | 0.079s | 0.52s |
| **Cold total** | **~57.5s** | **78.52s** |
| Peak RSS | ~491MB | ~1.41GB |
| Cache read | 0.018s | 0.043s |
| Cache-read speedup | ~263x vs. re-encoding (4.81s) | ~72.8x vs. raw CSV load (3.139s); ~2.27x "effective" incl. fingerprint cost |
| Cache correctness | bit-identical reload | bit-identical hashes; changed fingerprint correctly rejected |

```yaml
claude_environment: {os: Windows, python: "3.13.7", cpu_cores: 8, ram_gb: 16.76, numpy: "2.3.2", pandas: "2.3.2"}
gpt_environment: {os: "Windows 11", python: "3.13.7", cpu_cores: 8, ram_gb: 16.76, numpy: "2.3.2", pandas: "2.3.2"}
```

Windows subprocess-timeout probes:

```yaml
claude_finding: "subprocess.run(timeout=3) blocked 30.13s when tracked child spawned an unmanaged grandchild inheriting stdio pipes — FAILURE under this condition"
gpt_finding: "subprocess.run(timeout=0.3) fired at 0.313s; psutil recursive process-tree termination left neither parent nor child alive — PASS under this (simpler, non-adversarial) condition"
nan_inf_and_syntax_error_recovery: "cleanly detected in both Claude and GPT tests"
```

Harness/pipeline/agent scaffold status (GPT only, reviewer-corrected from 7 to 15 files):

```yaml
scaffold_only_files_count: 15
scaffold_files:
  - harness/executor.py
  - harness/guards.py
  - harness/cache.py
  - harness/diagnostics.py
  - harness/logger.py
  - harness/score.py
  - harness/submission.py
  - pipeline/data_adapter.py
  - pipeline/features.py
  - pipeline/train.py
  - agent/coder.py
  - agent/controller.py
  - agent/governor.py
  - agent/proposer.py
  - agent/reflector.py
empty_directories: [reports/, submissions/, runlogs/, tests/]
each_scaffold_file_executable_lines: 0
```

## 18. Known Evidence Links

```text
§1  task/splits/baseline reproduction      -> PRE_AUDIT.md §1
§3  entity cardinalities                   -> PRE_AUDIT.md §2.1
§5  train/valid overlap incl. tag scope    -> PRE_AUDIT.md §3.1-3.2
§6  author/video redundancy                -> PRE_AUDIT.md §3.3
§7-8 repeat-pair / row-level coverage      -> PRE_AUDIT.md §3.4-3.5
§9  temporal profile                       -> PRE_AUDIT.md §10.1-10.2
§10-11 metric structure / activity tiers   -> PRE_AUDIT.md §4, §5
§12 feedback signal profile                -> PRE_AUDIT.md §7
§13 historical availability                -> PRE_AUDIT.md §8
§14 video basic features                   -> PRE_AUDIT.md §9.1-9.2
§15 video statistics                       -> PRE_AUDIT.md §9.3-9.6
§16 random-exposure log                    -> PRE_AUDIT.md §11
§17 engineering profile                    -> PRE_AUDIT.md §12
```
