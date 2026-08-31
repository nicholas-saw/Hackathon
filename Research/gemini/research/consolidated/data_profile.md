# KuaiRand-Pure — Consolidated Data Profile

> Purpose: compact, agent-readable snapshot of dataset and baseline measurements.
> Contains ONLY final verified numerical facts and short structural notes.

## 1. Official Dataset Summary

```text
dataset: KuaiRand-Pure
task: within-user ranking
target: long_view
metrics: GAUC, nDCG@5
primary: mean(GAUC, nDCG@5)
```

| Split           | Dates                    |      Rows |
| --------------- | ------------------------ | --------: |
| Train           | 2022-04-08 .. 2022-04-21 | 1,141,112 |
| Validation      | 2022-04-22 .. 2022-04-28 |   124,909 |
| Evaluation/Test | 2022-04-29 .. 2022-05-08 |   170,588 |

_Note: The raw train log file has zero rows on 2022-04-08._

## 2. Official Baseline

| Metric  | Validation |
| ------- | ---------: |
| GAUC    |     0.6671 |
| nDCG@5  |     0.5358 |
| Primary |     0.6015 |

Baseline model: Factorization Machine (k=16, lr=0.001, fields: user_id, video_id, author_id, tab, dur_bucket).
Published seed standard deviation is 0.0008.

## 3. Entity Cardinalities (Train & Validation)

| Measurement    |  Train | Validation |
| -------------- | -----: | ---------: |
| Unique users   | 26,210 |     22,377 |
| Unique videos  |  7,538 |      5,951 |
| Unique authors |  6,482 |      5,315 |
| Unique tags    |    110 |        104 |

## 4. Train → Validation Overlap

| Measurement                               |  Value |
| ----------------------------------------- | -----: |
| Validation users seen in train            | 98.11% |
| Validation videos seen in train           | 99.88% |
| Validation authors seen in train          | 99.91% |
| Validation user-video pairs seen in train |  1.63% |
| Validation user-tag pairs seen in train   | 68.14% |

## 5. Missingness

| File / Field                         | Missing % |
| ------------------------------------ | --------: |
| Core interaction logs                |     0.00% |
| Video statistics                     |     0.00% |
| video_features_basic: video_duration |     3.15% |
| video_features_basic: music_type     |     2.68% |
| video_features_basic: tag            |     1.27% |
| user_features: onehot_feat4          |     3.20% |
| user_features: onehot_feat12-17      |     2.62% |

## 6. Author / Video Structure

| Measurement                  | Scope                    |  Value |
| ---------------------------- | ------------------------ | -----: |
| Authors with exactly 1 video | video basic feature file | 87.05% |
| Videos per author (median)   | video basic feature file |    1.0 |

## 7. User Activity

### Train impressions per user

min: 1 | median: 31 | mean: 43.54 | p99: 207 | max: 809

### Validation impressions per user

min: 1 | median: 4 | mean: 5.58 | p99: 26 | max: 74

## 8. Metric Structure (Validation)

### Uniform-label users

| Type            |  Users | % Users |
| --------------- | -----: | ------: |
| All negative    |  6,785 |  30.32% |
| All positive    |  2,663 |  11.90% |
| Mixed / movable | 12,929 |  57.78% |

### Baseline GAUC Weight Share by List Length

_Weight share uses official denominator (positives from mixed-label users only: 34,592)_
| List Length | GAUC Weight Share |
|---|---:|
| 1 | 0.00% |
| 2-3 | 10.27% |
| 4-5 | 16.36% |
| 6-10 | 36.39% |
| 11-20 | 27.08% |
| 21+ | 9.90% |

## 9. Feedback Signal Profile (Validation)

| Signal           | Mean / Prevalence | Pearson r with `long_view` (Same-row diagnostic) |
| ---------------- | ----------------: | -----------------------------------------------: |
| is_click         |            44.38% |                                            0.751 |
| play_time_ms     |          21,487ms |                                            0.632 |
| is_profile_enter |             1.95% |                                            0.127 |
| is_like          |             1.80% |                                            0.095 |
| is_comment       |             0.23% |                                            0.059 |
| is_follow        |            0.130% |                                            0.025 |

## 10. Historical Availability

| Measurement                                           |  Value |
| ----------------------------------------------------- | -----: |
| % validation users with >=1 prior train interaction   | 98.11% |
| % validation users with >=5 prior train interactions  | 92.85% |
| % validation users with >=10 prior train interactions | 85.17% |
| Repeat-video history coverage                         |  1.63% |
| Repeat-tag history coverage                           | 68.14% |

## 11. Video Statistics

| Statistic          | Missing % |   Mean |
| ------------------ | --------: | -----: |
| show_cnt           |      0.0% | 10,552 |
| play_cnt           |      0.0% |  7,747 |
| long_time_play_cnt |      0.0% |  3,687 |
| like_cnt           |      0.0% | 230.75 |
| comment_cnt        |      0.0% |  12.93 |

_Causal validity / aggregation window is INCONCLUSIVE._

## 12. Temporal Profile

| Period             | Dates          | Mean Rows/day |
| ------------------ | -------------- | ------------: |
| Early train (peak) | 04-09 .. 04-14 |      ~148,570 |
| Late train         | 04-15 .. 04-21 |       ~35,671 |
| Validation         | 04-22 .. 04-28 |        17,844 |

_Peak volume (04-11) is 278,835 rows, dropping to 20,021 on 04-21 (13.9x swing)._

## 13. Random Exposure Log

| Measurement              |                    Value |
| ------------------------ | -----------------------: |
| Date coverage            | 2022-04-22 .. 2022-05-08 |
| Overlap with train dates |                   0 rows |

## 14. Engineering Profile

| Stage              | Notes                                                                                                          |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| Baseline Execution | ~1 min (57-78s depending on environment)                                                                       |
| Codebase readiness | 15 scaffold files across `harness/`, `pipeline/`, and `agent/` must be implemented before experiments can run. |
