# KuaiRand-Pure — Verified Data Profile

> Compact train+validation snapshot produced by the pre-audit on 2026-08-30. Observations only; no model recommendation.

## 1. Task and Guardrails

```yaml
dataset: KuaiRand-Pure
task: within-user ranking over logged impressions
target: long_view
metrics: [GAUC, nDCG@5]
primary: (GAUC + nDCG@5) / 2
train_dates_declared: 2022-04-08..2022-04-21
train_dates_with_rows: 2022-04-09..2022-04-21
validation_dates: 2022-04-22..2022-04-28
test_labels_accessed_in_pre_audit: false
official_source_modified: false
context_constraints_modified: false
```

Evaluator facts:

- GAUC includes only mixed-label users and weights each by positive count.
- nDCG@5 averages all users; all-negative users score 0 and all-positive users are ranking-invariant at 1.
- Scores are used only for within-user order.

## 2. Baseline Reproduction

| Metric | Published validation | Reproduced seed 0 |
|---|---:|---:|
| GAUC | 0.6674 | 0.667133 |
| nDCG@5 | 0.5357 | 0.535806 |
| Primary | 0.6016 | 0.601470 |

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
best_epoch_seed_0: 7
stop_epoch_seed_0: 11
three_seed_primary_mean: 0.601440
three_seed_primary_population_std: 0.000275
```

## 3. Split Cardinalities

| Measurement | Train | Validation |
|---|---:|---:|
| Rows | 1,141,112 | 124,909 |
| Unique users | 26,210 | 22,377 |
| Unique videos | 7,538 | 5,951 |
| Unique authors | 6,482 | 5,315 |
| Tabs | 15 | 15 |
| Hour buckets | 24 | 24 |
| Dates with rows | 13 | 7 |
| Distinct duration values | 5,726 | 4,736 |

Side-table rows and coverage:

| Source | Rows | Train coverage | Validation coverage |
|---|---:|---:|---:|
| User features | 27,285 | 100% users | 100% users |
| Video basic | 7,583 | 100% videos | 100% videos |
| Video statistics | 7,583 | 100% videos | 100% videos |

Selected categorical cardinalities:

| Field | Cardinality |
|---|---:|
| Basic author_id | 6,510 |
| music_id | 7,202 |
| video_type | 3 |
| upload_type | 14 |
| music_type | 5 nonmissing |
| tag strings | 110 nonmissing |
| parsed tag tokens | 46 |
| user_active_degree | 9 observed |
| follow_user_num_range | 8 |
| fans_user_num_range | 9 |
| friend_user_num_range | 7 |
| register_days_range | 8 |

## 4. Missingness

Train and validation logs and all video-statistic fields have `0%` missingness.

| Source / field | Missing % |
|---|---:|
| user onehot_feat4 | 3.2032 |
| user onehot_feat12..17 | 2.6168 each |
| video video_duration | 3.1518 |
| video music_type | 2.6770 |
| video tag | 1.2660 |
| video visible_status | 0; cardinality 1 |

Full inventory: `research/experiment_results/missingness_inventory.csv`.

## 5. Train→Validation Overlap

| Measurement | Count / percentage |
|---|---:|
| Validation users seen in train | 21,955 / 98.114% |
| Cold validation users | 422 / 1.886% |
| Validation videos seen in train | 5,944 / 99.882% |
| Cold validation videos | 7 / 0.118% |
| Validation authors seen in train | 5,310 / 99.906% |
| Unique validation user–video pairs seen | 1,974 / 1.627% |
| Unique validation user–author pairs seen | 4,081 / 3.376% |
| Unique validation user–tag pairs seen | 68,316 / 71.913% |
| Validation rows with any prior user–tag | 78.413% |

## 6. Repeat and Author Structure

| Pair | Unique train pairs | Median count | Mean | p99 | Max | Repeated pairs | Repeated-interaction share |
|---|---:|---:|---:|---:|---:|---:|---:|
| User–video | 1,092,750 | 1 | 1.044 | 2 | 22 | 4.130% | 8.194% |
| User–author | 1,070,326 | 1 | 1.066 | 2 | 22 | 5.913% | 11.750% |
| User–tag | 345,211 | 2 | 3.674 | 29 | 167 | 55.250% | 87.819% |

```yaml
observed_authors: 6487
videos_per_author_median: 1
videos_per_author_mean: 1.1631
videos_per_author_p99: 3
videos_per_author_max: 24
authors_with_exactly_one_observed_video: 5647
authors_with_exactly_one_observed_video_pct: 87.051
video_to_author_functional_mapping_pct: 100
```

## 7. User Activity and Metric Composition

Train impressions per user:

```yaml
min: 1
median: 31
mean: 43.537
p90: 97
p99: 207
max: 809
```

Validation impressions per user:

```yaml
min: 1
median: 4
mean: 5.582
p90: 12
p99: 26
max: 74
single_impression_users: 3917
single_impression_users_pct: 17.505
```

Validation label composition:

| Type | Users | % users | Rows | % rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.321 | 21,807 | 17.458 |
| All positive | 2,663 | 11.901 | 4,540 | 3.635 |
| Mixed / movable | 12,929 | 57.778 | 98,562 | 78.907 |

Validation per-user positive rate: mean `0.3483`, median `0.2857`. User counts by rate: 0=`6,785`; `(0,.25]`=`4,040`; `(.25,.5]`=`6,057`; `(.5,.75]`=`2,425`; `(.75,1)`=`407`; 1=`2,663`.

## 8. Activity Buckets

Buckets use train-count quartiles over train users: T1 `1–13`, T2 `14–31`, T3 `32–59`, T4 `60+`; Cold `0`.

| Tier | Users | Validation rows | GAUC | nDCG@5 | Primary | Invariant users | GAUC weight | Oracle nDCG | nDCG gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6091 | 57.11% | 1.69% | 0.6422 | 0.1117 |
| T1 | 4,351 | 13,437 | 0.6550 | 0.5374 | 0.5962 | 61.25% | 10.55% | 0.6173 | 0.0799 |
| T2 | 5,582 | 23,310 | 0.6686 | 0.5409 | 0.6047 | 48.01% | 19.95% | 0.6603 | 0.1194 |
| T3 | 5,791 | 32,052 | 0.6624 | 0.5521 | 0.6073 | 37.70% | 27.80% | 0.7177 | 0.1656 |
| T4 | 6,231 | 54,120 | 0.6720 | 0.5154 | 0.5937 | 26.95% | 40.01% | 0.7692 | 0.2538 |

Aggregate nDCG-gap contributions: Cold `0.00211`, T1 `0.01554`, T2 `0.02979`, T3 `0.04286`, T4 `0.07068`.

## 9. Validation List-Length Buckets

| Length | Users | Rows | GAUC | nDCG@5 | Oracle nDCG | Gap | GAUC weight | Aggregate gap contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.0000 | 0.00% | 0.0000 |
| 2–3 | 6,218 | 15,323 | 0.6472 | 0.5413 | 0.6086 | 0.0673 | 10.27% | 0.01869 |
| 4–5 | 4,119 | 18,326 | 0.6645 | 0.6185 | 0.7492 | 0.1307 | 16.36% | 0.02405 |
| 6–10 | 5,225 | 39,587 | 0.6756 | 0.5913 | 0.8536 | 0.2623 | 36.39% | 0.06125 |
| 11–20 | 2,346 | 32,609 | 0.6677 | 0.5037 | 0.9182 | 0.4145 | 27.08% | 0.04345 |
| 21+ | 552 | 15,147 | 0.6596 | 0.3934 | 0.9420 | 0.5486 | 9.90% | 0.01353 |

`*` Empty GAUC denominator; evaluator returns 0.5.

## 10. Feedback Signals

| Signal | Train mean/prevalence | Validation | Train corr with long_view | Validation corr | Notes |
|---|---:|---:|---:|---:|---|
| is_click | 0.46345 | 0.44383 | 0.7605 | 0.7515 | Dense; definition overlaps watch-time threshold |
| is_like | 0.01868 | 0.01797 | 0.0992 | 0.0949 | Sparse |
| is_follow | 0.00101 | 0.00130 | 0.0250 | 0.0253 | Very sparse |
| is_comment | 0.00257 | 0.00233 | 0.0590 | 0.0587 | Very sparse |
| is_forward | 0.00100 | 0.00078 | 0.0226 | 0.0245 | Very sparse |
| is_hate | 0.00042 | 0.00062 | −0.0039 | −0.0038 | Very sparse |
| is_profile_enter | 0.02539 | 0.01945 | 0.1461 | 0.1271 | Sparse |
| play_time_ms | 23,260.5 | 21,486.8 | 0.6351 raw | 0.6319 raw | Train median 4,970; p99 213,231 |
| profile_stay_time | 3.31 | 1.88 | 0.0079 log1p | 0.0057 log1p | >99.98% zero |
| comment_stay_time | 552.9 | 460.3 | 0.2702 log1p | 0.2500 log1p | ~95% zero |

Train conditional long-view rate: click=0 `0.00263`; click=1 `0.72330`.

Validation click prevalence by activity: Cold `48.59%`, T1 `50.83%`, T2 `48.58%`, T3 `46.82%`, T4 `39.38%`.

## 11. Strict Train-Derived History

| Measurement | Value |
|---|---:|
| Validation users with ≥1 prior interaction | 98.114% |
| ≥5 prior interactions | 92.854% |
| ≥10 prior interactions | 85.168% |
| Median prior interactions | 35 |
| Mean prior interactions | 47.42 |
| Validation rows with prior same video | 1.624% |
| Prior same author | 3.381% |
| Prior same tag | 78.413% |

User coverage by prior signal count:

| Prior signal | ≥1 | ≥5 | ≥10 |
|---|---:|---:|---:|
| Interactions | 98.114% | 92.854% | 85.168% |
| Clicks | 96.157% | 82.531% | 66.309% |
| Likes | 23.229% | 4.683% | 2.239% |
| Follows | 3.423% | 0.054% | 0.022% |
| Comments | 7.785% | 0.241% | 0.018% |
| Forwards | 3.365% | 0.049% | 0.013% |
| Hates | 1.028% | 0.063% | 0.031% |
| Positive-play-time rows | 97.640% | 91.053% | 82.111% |

| Tier | Median prior rows | ≥1 | ≥5 | ≥10 | Same video | Same author | Same tag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 0 | 0% | 0% | 0% | 0% | 0% | 0% |
| T1 | 7 | 100% | 72.95% | 33.42% | 0.68% | 1.48% | 46.97% |
| T2 | 22 | 100% | 100% | 100% | 1.03% | 2.27% | 71.65% |
| T3 | 43 | 100% | 100% | 100% | 1.29% | 2.79% | 81.75% |
| T4 | 89 | 100% | 100% | 100% | 2.37% | 4.80% | 90.04% |

Official caveat: KuaiRand-Pure contains incomplete sequential logs.

## 12. Video Features and Statistics

Basic redundancy:

```yaml
visible_status_cardinality: 1
video_duration_missing_pct: 3.1518
logged_duration_rows_with_basic_duration_pct: 97.9319
logged_duration_exact_match_pct_when_basic_present: 100
logged_vs_basic_duration_spearman: 1.0
```

Statistic redundancy:

```yaml
numeric_field_pairs_abs_spearman_ge_0_95: 54
like_cnt_vs_like_user_num: 0.999865
follow_cnt_vs_follow_user_num: 0.999754
long_time_play_cnt_vs_long_time_play_user_num: 0.999678
valid_play_cnt_vs_valid_play_user_num: 0.999499
play_cnt_vs_play_user_num: 0.999000
```

Fixed standalone ratio diagnostics (beta=20 smoothing; train item-pop primary `0.580722`):

| Ratio | Validation primary | Delta vs item-pop | Spearman vs train item rate |
|---|---:|---:|---:|
| long_time_play_cnt / show_cnt | 0.580378 | −0.000344 | 0.7167 |
| valid_play_cnt / show_cnt | 0.570874 | −0.009848 | 0.6543 |
| complete_play_cnt / show_cnt | 0.550128 | −0.030594 | 0.4396 |
| play_cnt / show_cnt | 0.540600 | −0.040122 | 0.4483 |
| like_cnt / show_cnt | 0.483741 | −0.096981 | 0.2633 |
| comment_cnt / show_cnt | 0.454772 | −0.125950 | 0.0024 |
| follow_cnt / show_cnt | 0.456476 | −0.124246 | 0.1586 |
| share_cnt / show_cnt | 0.448518 | −0.132204 | −0.1233 |

```yaml
official_stat_semantics: average per day and scenario over one month
exact_calendar_window_documented: false
causal_validity_for_validation: unresolved
```

## 13. Temporal Profile

| Period | Days with rows | Rows | Rows/day | Long-view rate | Users/day | Videos/day | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early train (Apr 8–14) | 6 | 891,418 | 148,570 | 0.33228 | 18,471 | 5,257 | 98,553 |
| Late train (Apr 15–21) | 7 | 249,694 | 35,671 | 0.35211 | 12,424 | 4,207 | 95,477 |
| Validation | 7 | 124,909 | 17,844 | 0.31328 | 9,140 | 3,429 | 102,820 |

Validation distance:

| Observable | vs early | vs late | Closer period |
|---|---:|---:|---|
| Absolute long-view-rate gap | 0.01900 | 0.03882 | Early |
| Absolute mean-duration gap | 4,267ms | 7,343ms | Early |
| Tab JS divergence (bits) | 0.003916 | 0.002515 | Late |

Validation is not uniformly closer to one train period.

## 14. Random-Exposure Log

```yaml
columns_loaded_in_audit: [user_id, video_id, date]
labels_loaded: false
rows: 1186059
date_min: 2022-04-22
date_max: 2022-05-08
unique_users: 27285
unique_videos: 7583
unique_pairs: 1186006
evaluation_date_rows: 897721
evaluation_date_rows_pct: 75.6894
```

Overlap with standard train+validation:

| Object | Overlap count | % of random uniques |
|---|---:|---:|
| Users | 26,632 | 97.607% |
| Videos | 7,545 | 99.499% |
| User–video pairs | 702 | 0.0592% |

## 15. Controlled FM Evidence

All results use validation only; population std across seeds.

| Config | Seeds | Primary mean ± std | Paired delta mean ± std |
|---|---:|---:|---:|
| Base | 3 | 0.601440 ± 0.000275 | reference |
| Remove author | 5 | 0.602889 ± 0.000451 | +0.001316 ± 0.000426 |
| Remove video | 5 | 0.602654 ± 0.000307 | +0.001082 ± 0.000585 |
| Remove tab | 3 | 0.585538 ± 0.000429 | −0.015903 ± 0.000467 |
| Remove duration bucket | 3 | 0.600849 ± 0.000225 | −0.000591 ± 0.000156 |
| Add item static fields | 3 | 0.601108 ± 0.000461 | −0.000332 ± 0.000205 |
| Full 13-field static | 3 | 0.599930 ± 0.000523 | −0.001510 ± 0.000792 |
| k=8 | 3 | 0.601110 ± 0.000796 | −0.000330 ± 0.000790 |
| k=32 | 3 | 0.601460 ± 0.000688 | +0.000020 ± 0.000504 |
| lr=0.0005 | 3 | 0.601776 ± 0.000280 | +0.000336 ± 0.000353 |
| lr=0.002 | 3 | 0.601364 ± 0.000826 | −0.000076 ± 0.000625 |

## 16. Engineering Profile

| Stage | Time |
|---|---:|
| Validation-only official loader | 2.88s |
| Baseline encoding | 8.47s |
| Baseline train + 11 epoch evaluations | 66.60s |
| Approximate update time excluding epoch evaluations | 60.08s |
| Final validation evaluation | 0.52s |
| Full cold baseline | 78.52s |
| Descriptive profile | 37.36s |
| Raw pandas train+validation load | 3.139s |
| Verified cache read | 0.043s |
| Full content fingerprint | 1.341s |
| Effective fingerprint + cache read | 1.384s |

```yaml
os: Windows 11
python: 3.13.7
numpy: 2.3.2
pandas: 2.3.2
processor: ARMv8, AMD64-compatible Python environment
logical_cpus: 8
physical_memory_bytes: 16756445184
profile_peak_rss_bytes_approx: 1412079616
cache_bytes: 81028199
cache_frame_hashes_identical: true
changed_source_fingerprint_rejected: true
subprocess_probe_passed: true
timeout_probe_passed: true
recursive_process_tree_termination_passed: true
syntax_error_recovery_probe_passed: true
nan_inf_detection_probe_passed: true
```

Current final-system readiness:

```yaml
harness_executor_implemented: false
harness_guards_implemented: false
harness_cache_implemented: false
harness_diagnostics_implemented: false
harness_logger_implemented: false
harness_score_implemented: false
harness_submission_implemented: false
pipeline_data_adapter_implemented: false
pipeline_features_implemented: false
pipeline_train_implemented: false
agent_coder_implemented: false
agent_controller_implemented: false
agent_governor_implemented: false
agent_proposer_implemented: false
agent_reflector_implemented: false
```

Each of these 15 files contains zero executable non-comment lines as of the audit. `reports/`, `submissions/`, `runlogs/`, and `tests/` contain no files. Review correction: the original J01/J02 probe inventoried only the first 7 of these; the remaining 8 (`harness/logger.py`, `score.py`, `submission.py`, and all five `agent/*.py` files) were confirmed empty during review.

## 17. Artifact Index

```text
baseline reproduction       -> experiment_results/baseline_validation.json
main structured profile     -> experiment_results/data_profile_results.json
missingness                 -> experiment_results/missingness_inventory.csv
activity metrics            -> experiment_results/metric_by_activity_bucket.csv
list-length metrics         -> experiment_results/metric_by_list_length.csv
feedback                    -> experiment_results/feedback_*.csv
history                     -> experiment_results/history_*.csv
video inventory/ratios      -> experiment_results/video_*.csv
temporal                    -> experiment_results/daily_standard_profile.csv
controlled FM experiments   -> experiment_results/controlled_fm_*.json
cache verification          -> experiment_results/cache_probe.json
engineering probes          -> experiment_results/engineering_environment_probe.json
full interpretation         -> PRE_AUDIT.md
```
