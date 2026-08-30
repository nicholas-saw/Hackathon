# KuaiRand-Pure — Consolidated Verified Data Profile

> Compact final facts only. Local outcome statistics use train/validation only. For interpretation and unresolved semantics, see `PRE_AUDIT.md`.

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
| Evaluation | Official dates / official row count | 2022-04-29..05-08 / 170,588 rows |

GAUC includes only mixed-label users and weights each by its positive count. nDCG@5 averages all users equally; all-negative users receive 0 and all-positive users are ranking-invariant.

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

Validation click/play-time inter-correlation: 0.5167. Profile stay is 99.99% zero; comment stay is 95.54% zero.

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

Aggregation population, exact calendar window, and causal safety: **INCONCLUSIVE**.

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

FM width means: k=8 0.60111; k=16 0.60144; k=32 0.60146; k=64 0.60099. Simple width scaling is STRONG NEGATIVE EVIDENCE for meaningful gain in this FM.

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
