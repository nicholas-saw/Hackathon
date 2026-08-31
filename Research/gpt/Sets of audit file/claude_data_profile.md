# KuaiRand-Pure — Data Profile

> Purpose: compact, agent-readable snapshot of dataset and baseline measurements.
> Populated from the pre-audit (`research/scripts/*`, `research/experiment_results/*.json`).
> Observations only — interpretation lives in `PRE_AUDIT.md`.

## 1. Official Dataset Summary

```text
dataset: KuaiRand-Pure
task: within-user ranking
target: long_view
metrics: GAUC, nDCG@5
primary: mean(GAUC, nDCG@5)
```

### Official splits (reproduced locally)

| Split | Dates | Rows |
|---|---|---:|
| Train | 2022-04-08 .. 2022-04-21 | 1,141,112 |
| Validation | 2022-04-22 .. 2022-04-28 | 124,909 |
| Evaluation/Test | 2022-04-29 .. 2022-05-08 | 170,588 |

Row counts confirmed exactly via train/validation loading plus a date-only count of evaluation rows (Investigation A01).

Do not inspect evaluation/test labels during development.

### Anomaly: date 2022-04-08 has zero rows

`log_standard_4_08_to_4_21_pure.csv` contains only **13** distinct dates (04-09 .. 04-21), not the 14 implied by the official range. 2022-04-08 itself has **0 rows**. See Investigation A06.

---

## 2. Official Baseline — Reproduction

| Metric | Published Valid | Reproduced Valid (seed 0) | Diff |
|---|---:|---:|---:|
| GAUC | 0.6674 | 0.6671 | 0.0003 |
| nDCG@5 | 0.5357 | 0.5358 | 0.0001 |
| Primary | 0.6016 | 0.6015 | 0.0001 |

All diffs are well inside the published seed std of 0.0008. Environment reproduction: **CONFIRMED**.

Baseline model:

```text
Factorization Machine
k = 16, lr = 0.001, batch = 8192, max_epochs = 40, patience = 4, Adam optimizer
fields: user_id, video_id, author_id, tab, dur_bucket
```

5-seed variance (this environment, VALIDATION only — Investigation C02):

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Mean (5 seeds) | 0.66740 | 0.53574 | 0.60157 |
| Std (5 seeds) | 0.00031 | 0.00038 | 0.00032 |

Our local std (0.00032) is the same order of magnitude as the published std (0.0008).

---

## 3. Entity Cardinalities

| Measurement | Train | Validation |
|---|---:|---:|
| Unique users | 26,210 | 22,377 |
| Unique videos | 7,538 | 5,951 |
| Unique authors | 6,482 | 5,315 |
| Unique tab values | 15 | 15 |
| Total users in user_features_pure.csv | 27,285 | — |
| Total videos in video_features_basic_pure.csv | 7,583 | — |
| Unique tags | 111 | — |
| Unique music_id | 7,202 | — |
| Unique user_active_degree values | 9 | — |

---

## 4. Train → Validation Overlap

| Measurement | Value |
|---|---:|
| Validation users seen in train | 98.11% |
| Validation videos seen in train | 99.88% |
| Validation authors seen in train | 99.91% |
| Validation user-video PAIRS seen in train | 1.63% |
| Validation user-author PAIRS seen in train | 3.38% |
| Validation user-tag PAIRS seen in train | 68.14% |
| Cold validation users (count / %) | 422 / 1.89% |
| Cold validation videos (count / %) | 7 / 0.12% |
| Cold validation authors (count / %) | 5 / 0.09% |
| Valid rows belonging to a cold user | 1.59% |
| Valid rows belonging to a cold video | 0.014% |

---

## 5. Missingness (selected; full detail in `phase_a_structure.json`)

| File / Field | Missing % |
|---|---:|
| video_basic.video_duration | 3.15% |
| video_basic.music_type | 2.68% |
| video_basic.tag | 1.27% |
| user_features.onehot_feat4 | 3.20% |
| user_features.onehot_feat12-17 | 2.62% each |
| Standard logs (all fields) | 0.0% |
| video_stat (all fields) | 0.0% |
| valid videos missing from video_features_statistic_pure.csv | 0.0% |

---

## 6. Author / Video Structure

| Measurement | Value |
|---|---:|
| Videos per author — median | 1.0 |
| Videos per author — mean | 1.165 |
| Videos per author — p90 | 2.0 |
| Videos per author — max | 26 |
| Authors with exactly 1 video | 5,661 / 6,510 |
| % authors with exactly 1 video | 86.96% |

---

## 7. Repeat-Pair Frequency

| Measurement | Train | Validation |
|---|---:|---:|
| user-video unique pairs | 1,092,750 | 121,337 |
| user-video rows | 1,141,112 | 124,909 |
| % user-video pairs repeated >1x | 4.13% | 2.90% |
| % rows in repeated user-video pairs | 8.19% | 5.67% |
| user-author unique pairs | 1,070,326 | 120,885 |
| % user-author pairs repeated >1x | 5.91% | 3.25% |
| user-tag unique pairs | 355,242 | 90,121 |
| % user-tag pairs repeated >1x | 51.77% | 24.45% |
| % rows in repeated user-tag pairs | 84.98% | 45.49% |
| Duplicate (user_id, video_id) pairs within valid | 2.90% of pairs, max repeat 7 |

---

## 8. Metric Structure (Investigation B01-B06)

### Uniform-label users — validation

| Type | Users | % Users | Rows | % Rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.32% | 21,807 | 17.46% |
| All positive | 2,663 | 11.90% | 4,540 | 3.63% |
| Mixed / movable | 12,929 | 57.78% | 98,562 | 78.91% |
| Single impression | 3,917 | 17.50% | 3,917 | 3.14% |

### Oracle ceiling — LOCAL validation split (own reproduction)

| Metric | Value |
|---|---:|
| GAUC | 1.0000 |
| nDCG@5 | 0.6968 |
| Primary | 0.8484 |

Matches `baseline_scores.json` published valid-oracle exactly (0.8484) — cross-check passed.

### List-length buckets — validation (baseline seed-0 scores vs. oracle)

| List Length | Users | Rows | Baseline nDCG@5 | Oracle nDCG@5 | Movable Gap | GAUC Weight Share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.4054 | 0.4054 | 0.000 | 0.00% |
| 2–3 | 6,218 | 15,323 | 0.5413 | 0.6086 | 0.067 | 10.27% |
| 4–5 | 4,119 | 18,326 | 0.6185 | 0.7492 | 0.131 | 16.36% |
| 6–10 | 5,225 | 39,587 | 0.5913 | 0.8536 | 0.262 | 36.39% |
| 11–20 | 2,346 | 32,609 | 0.5037 | 0.9182 | 0.414 | 27.08% |
| 21+ | 552 | 15,147 | 0.3934 | 0.9420 | 0.549 | 9.90% |

Overall validation list-length distribution: min 1, median 4, mean 5.58, p90 12, p99 26, max 74.

### Activity-tier buckets — validation (train-derived tiers; edges at train-impression-count 17/36/65)

| Tier | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG@5 | Movable nDCG Gap | Fixed-users % | GAUC Weight Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold (0 train rows) | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 0.112 | 57.11% | 1.69% |
| T1 (1-17) | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 0.087 | 58.81% | 14.67% |
| T2 (18-36) | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 0.127 | 45.91% | 21.35% |
| T3 (37-65) | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 0.177 | 35.60% | 27.50% |
| T4 (66+) | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 0.262 | 26.31% | 34.79% |

GAUC shares use the official denominator: positive rows from mixed-label users only (34,592 positives). The earlier profile incorrectly included positives from all-positive users.

### Joint activity-tier × validation-list-length analysis

Train-side user activity and validation list length are moderately positively associated: Spearman correlation is 0.4620 across all validation users and 0.4677 among warm users only (Pearson correlation 0.4419 across all users).

| Tier | List 1 | 2–3 | 4–5 | 6–10 | 11–20 | 21+ |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 37.20% | 24.64% | 12.09% | 16.82% | 5.92% | 3.32% |
| T1 | 33.43% | 36.85% | 14.88% | 11.50% | 2.78% | 0.56% |
| T2 | 18.32% | 33.61% | 20.97% | 20.51% | 5.49% | 1.09% |
| T3 | 10.81% | 25.54% | 22.20% | 28.64% | 11.11% | 1.69% |
| T4 | 4.92% | 14.75% | 16.26% | 34.03% | 23.46% | 6.59% |

Cells are percentages of users within each activity tier. The share with lists of 6+ rises from 14.84% in T1 to 64.07% in T4.

| Intersection | Users | Rows | GAUC Weight | Total GAUC Gap | Total nDCG Gap | Total Primary Gap |
|---|---:|---:|---:|---:|---:|---:|
| T4 × list 6+ | 3,453 (15.43%) | 42,020 (33.64%) | 30.35% | 29.88% | 34.37% | 31.35% |
| T3/T4 × list 6+ | 5,680 (25.38%) | 64,133 (51.34%) | 50.79% | 50.65% | 53.94% | 51.72% |
| T2/T3/T4 × list 6+ | 7,165 (32.02%) | 78,253 (62.65%) | 64.36% | 63.65% | 65.83% | 64.36% |
| T3/T4 × list 11+ | 2,307 (10.31%) | 38,219 (30.60%) | 27.80% | 27.96% | 28.64% | 28.18% |

All 30 disjoint joint cells reconcile to 22,377 users, 124,909 rows, 100% of official GAUC weight, and the complete baseline-to-oracle GAUC/nDCG/primary gaps. These are diagnostic headroom shares under the current seed-0 baseline, not causal effects or guaranteed attainable gains.

---

## 9. Post-Impression Feedback Profile (validation; train in `phase_d_feedback.json`)

| Signal | Valid Mean/Prevalence | Pearson r with `long_view` (same-row, diagnostic only) |
|---|---:|---:|
| is_click | 44.38% | 0.751 |
| is_like | 1.80% | 0.095 |
| is_follow | 0.130% | 0.025 |
| is_comment | 0.233% | 0.059 |
| is_forward | 0.078% | 0.025 |
| is_hate | 0.062% | -0.004 |
| is_profile_enter | 1.95% | 0.127 |
| play_time_ms | mean 21,487ms, median 4,607ms, 11.7% zero | 0.632 |
| profile_stay_time | 99.99% zero | -0.0005 |
| comment_stay_time | 95.54% zero | 0.169 |

`is_click` and `play_time_ms` are very strongly associated with `long_view` at the same row (r = 0.75, 0.63) — consistent with `long_view` being functionally derived from watch duration gated by click. **This is same-row diagnostic evidence only; RULES.md forbids using these as same-row inputs.**

### Feedback by activity tier (validation)

| Tier | long_view rate | is_click mean | play_time_ms mean |
|---|---:|---:|---:|
| Cold | 0.361 | 0.486 | 25,583 |
| T1 | 0.374 | 0.501 | 28,367 |
| T2 | 0.350 | 0.483 | 24,302 |
| T3 | 0.331 | 0.466 | 22,354 |
| T4 | 0.258 | 0.386 | 16,681 |

Notable non-monotonic pattern: highest-activity users (T4) show the *lowest* long_view rate and click rate.

---

## 10. Historical Information Availability

Because train dates strictly precede validation dates, "prior train interactions" for a validation row equals that user's total train impression count.

| Measurement | Value |
|---|---:|
| % validation users with ≥1 prior train interaction | 98.11% |
| % validation users with ≥5 prior train interactions | 92.85% |
| % validation users with ≥10 prior train interactions | 85.17% |
| Median prior interactions | 35 |
| Mean prior interactions | 47.4 |
| p90 prior interactions | 103 |

### By activity tier

| Tier | Median Prior Rows | ≥1 | ≥5 | ≥10 |
|---|---:|---:|---:|---:|
| Cold | 0 | 0% | 0% | 0% |
| T1 | 9 | 100% | 79.4% | 49.3% |
| T2 | 26 | 100% | 100% | 100% |
| T3 | 49 | 100% | 100% | 100% |
| T4 | 95 | 100% | 100% | 100% |

### Row-level repeat coverage (validation rows where user's train history contains the same entity)

| Coverage | Overall | Cold | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|
| Same video seen before | 1.62% | 0% | 0.73% | 1.09% | 1.37% | 2.47% |
| Same author seen before | 3.38% | 0% | 1.57% | 2.44% | 2.90% | 5.01% |
| Same tag seen before | 73.19% | 0% | 45.80% | 67.97% | 77.59% | 86.47% |

Bonus (not explicitly requested): **81.57%** of validation rows have at least one row from the same user at a strictly earlier `(date, hourmin, time_ms)` timestamp. The earlier 82.09% value incorrectly ordered rows tied at the same timestamp; 5.60% of rows belong to non-unique user/timestamp groups. This is an availability diagnostic, not a validated feature protocol.

---

## 11. Video Basic Features

| Field | Cardinality | Notes |
|---|---:|---|
| video_type | 3 | 98.98% NORMAL, 1.00% AD |
| upload_type | 14 | top: LongImport 38.6%, Web 31.9% |
| visible_status | 1 | constant (0.0) — no information |
| music_type | 6 | 87.9% value 9.0 |
| tag | 111 | 1.27% missing |
| music_id | 7,202 | ≈1 per video |
| video_duration | continuous | median 81,171ms, p90 237,830ms, 3.15% missing |
| server_width/height | continuous | median 720×1280 |

`tab` (15 categories) shows large long_view-rate spread by category in TRAIN: from 0.42% (tab 3, n=3,574) to 61.25% (tab 10, n=80) to 48.9% (tab 4, n=75,524); the two dominant tabs are tab 1 (834,876 rows, rate 38.6%) and tab 0 (150,013 rows, rate 4.2%).

`dur_bucket` (10 quantile buckets of duration_ms) shows a milder but real spread: 0.273 (bucket 1) to 0.376 (bucket 6) in TRAIN long_view rate.

---

## 12. Video Statistics (`video_features_statistic_pure.csv`)

| Statistic | Missing % | Notes |
|---|---:|---|
| show_cnt | 0% | mean ≈ per-window daily average, not a raw count (see below) |
| play_cnt, complete_play_cnt, valid_play_cnt, long_time_play_cnt, short_time_play_cnt | 0% | same normalization |
| like_cnt, comment_cnt, follow_cnt, share_cnt, download_cnt, collect_cnt | 0% | same normalization |
| counts | 0% | integer, range 45–181, median 147 |

### Aggregation-window finding (Investigation E/G — IMPORTANT, flagged uncertain)

`show_cnt × counts` is near-integer for **100%** of videos, supporting the hypothesis that `counts` is a denominator for averaged statistics. It does not identify what a `counts` unit means.

Reconstructed total (`show_cnt × counts`) vs. actual observed impressions for that video in **train+valid standard logs**:

| Measurement | Value |
|---|---:|
| Median ratio (reconstructed / observed) | 11,465× |
| p10 ratio | 5,248× |
| p90 ratio | 38,199× |
| % videos where reconstructed < observed | 0.0% |

**Interpretation is uncertain (labeled explicitly):** the statistic file's aggregation window/population is not documented anywhere in the Starter Kit or data guide. The ratio (always ≥1, typically ~4 orders of magnitude larger than our sampled train+valid traffic) is consistent with these statistics being computed over the **full platform's traffic** (all users, not just the ~27K sampled KuaiRand-Pure users) and/or a longer external time window (the `counts` field itself ranges 45–181, i.e. up to ~6x longer than the dataset's 31-day span). Whether the window overlaps the evaluation period cannot be determined from available files. **Causal validity of any feature derived from this file is therefore not established and should be treated as uncertain, not assumed safe.**

### Ratio-feature associations with `long_view` (VALIDATION only, smoothed with α=1, β=20)

| Ratio | Pearson r | Bottom Quintile Rate | Top Quintile Rate |
|---|---:|---:|---:|
| (long_time_play_cnt+1)/(show_cnt+20) | 0.302 | 0.105 | 0.505 |
| (play_cnt+1)/(show_cnt+20) | 0.185 | 0.180 | 0.396 |
| (complete_play_cnt+1)/(show_cnt+20) | 0.181 | 0.190 | 0.436 |
| (like_cnt+1)/(show_cnt+20) | 0.040 | 0.249 | 0.353 |

Three ratios show monotonic quintile trends; the like ratio rises through Q4 and dips slightly in Q5 (0.357 to 0.353). The data source is associated with the target, but incremental value over the existing baseline and causal safety are not established.

---

## 13. Temporal Profile

| Period | Dates | Rows | long_view Rate | Unique Users | Unique Videos |
|---|---|---:|---:|---:|---:|
| Early train | 04-09..04-14 (6 days) | 891,418 | 0.3323 | 25,151 | 7,521 |
| Late train | 04-15..04-21 (7 days) | 249,694 | 0.3521 | 24,262 | 6,571 |
| Validation | 04-22..04-28 | 124,909 | 0.3133 | 22,377 | 5,951 |

### Daily row-count decay within train (HARD FACT)

```text
04-09: 52,736   04-13: 94,711   04-17: 44,023   04-21: 20,021
04-10: 227,808  04-14: 71,252   04-18: 24,560
04-11: 278,835  04-15: 58,892   04-19: 20,443
04-12: 166,076  04-16: 60,904   04-20: 20,851
```

Peak volume (04-11: 278,835 rows) is **~14x** the last train day's volume (04-21: 20,021 rows). Validation's daily volume (13,972–26,645 rows/day) is much closer in scale to the *tail* of train (last ~7 days, 20-60K/day) than to the volume peak in the middle of train.

| Jaccard overlap | early-vs-late train | early-vs-valid | late-vs-valid |
|---|---:|---:|---:|
| Video sets | 0.869 | 0.787 | 0.818 |
| User sets | 0.885 | 0.808 | 0.809 |

| long_view rate gap vs. validation | early train | late train |
|---|---:|---:|
| Absolute gap | 0.0190 | 0.0388 |

Note: despite volume/entity-set structure suggesting validation resembles the *tail* of train more, validation's long_view rate is numerically closer to *early*-train's rate than late-train's. Both facts are reported; no recency-weighting conclusion is drawn here.

---

## 14. Random Exposure Log (`log_random_4_22_to_5_08_pure.csv`)

| Measurement | Value |
|---|---:|
| Total rows | 1,186,059 (date-only count across full file) |
| Date coverage | 2022-04-22 .. 2022-05-08 (17 dates) |
| Rows overlapping validation dates (04-22..04-28) | 288,338 (24.3%) |
| Rows overlapping evaluation dates (04-29..05-08) | 897,721 (75.7%; date-only count, no outcomes/features accessed) |
| `is_rand`=1 for 100% of rows | vs. 0% in both standard logs |
| Validation-period random-log users also present in train+standard-valid | 98.89% |
| Random-log videos also present in standard logs | 99.50% |
| Train+standard-valid users covered by validation-period random log | 70.89% |
| Train+standard-valid videos covered by validation-period random log | 99.51% |
| Shared (user,video) pairs: random(valid-period) ∩ standard-valid | 17 of 288,328 (0.006%) |
| long_view rate, random log, validation-period rows | 8.06% |
| long_view rate, standard valid log (for comparison) | 31.3% |

The random log's engagement rate (~8%) is roughly 4x lower than the standard log (~31%), a large distribution difference consistent with exposure/selection effects but not by itself a causal estimate of bias.

---

## 15. Engineering Profile

| Stage | Time (s) | Notes |
|---|---:|---|
| CSV load (train log) | 1.09 | pandas read_csv |
| CSV load (valid log, strict date-filtered materialization) | 1.72 | evaluation rows not materialized |
| CSV load (video_basic) | 0.015 | |
| CSV load (video_stat) | 0.067 | |
| CSV load (user_features) | 0.096 | |
| **Total load** | **2.99** | |
| Encoding (5-field baseline) | 4.81 | |
| FM training (seed 0, official config, 11 epochs to early-stop) | 49.7 | ≈4.43s/epoch |
| Predict-only (validation) | 0.079 | |
| **Cold run total (load+encode+train)** | **~57.5** | |
| Cache write (pickle encoded arrays) | 0.029 | |
| Cache read | 0.018 | **263x** faster than re-encoding on this run |

Cache correctness check: reload reproduces bit-identical arrays; cache built from train+valid only (no test).

### Environment

```text
OS: Windows-11-10.0.26200-SP0
Python: 3.13.7
CPU: 8 logical cores
RAM: 16.76 GB total; 0.68 GB available at review-rerun measurement time (other
     processes on this workstation consume the rest) — see Engineering Constraints
     in PRE_AUDIT.md.
numpy 2.3.2, pandas 2.3.2
Peak RSS observed for the full load+encode+train pipeline: ~491MB
```

### Windows subprocess-timeout finding (HARD FACT, verified)

`subprocess.run(cmd, timeout=3, capture_output=True)` against a child that itself spawns an unmanaged grandchild did **not** return in 3 seconds; it blocked for the grandchild's full **30.13s** lifetime. The timing is consistent with inherited pipe handles preventing `communicate()` from seeing EOF. The overrun is verified for this condition; the mechanism and any replacement process-tree termination strategy should be tested explicitly by the future harness.

### Other verified recovery behaviors

- Syntax error in a child script: clean nonzero return code (1) + `"SyntaxError"` present in stderr — a reliably detectable failure signature.
- NaN/Inf in a submission file: the official `submit.py::read_submission` already rejects it (unmodified official code, exercised on a synthetic validation-shaped CSV only).

---

## 16. Baseline Mechanism (Investigations C01-C05)

All numbers: VALIDATION only, 3 seeds per configuration unless noted (seed variance itself uses 5 seeds), official FM class/optimizer, k=16/lr=0.001 except where swept.

### C01 — Field ablations (leave-one-out from the 5 official fields)

| Config | Fields | Mean Primary | Std | Δ vs. full |
|---|---|---:|---:|---:|
| full_5field | user_id, video_id, author_id, tab, dur_bucket | 0.60144 | 0.00027 | — |
| drop_user_id | video_id, author_id, tab, dur_bucket | 0.59325 | 0.00006 | **-0.00819** |
| drop_video_id | user_id, author_id, tab, dur_bucket | 0.60280 | 0.00032 | **+0.00136** |
| drop_author_id | user_id, video_id, tab, dur_bucket | 0.60301 | 0.00026 | **+0.00157** |
| drop_tab | user_id, video_id, author_id, dur_bucket | 0.58554 | 0.00043 | **-0.01590** |
| drop_dur_bucket | user_id, video_id, author_id, tab | 0.60085 | 0.00023 | -0.00059 |

`user_id` and `tab` are indispensable (large, unambiguous drops). `dur_bucket`'s removal effect is within ~2x the combined std (inconclusive). **Removing `video_id` or `author_id` individually did not hurt, and both showed a small, direction-consistent primary-score increase** (+0.0014 and +0.0016 respectively, each ~3-4x the combined per-config std) — see Investigation C01 for the cautious interpretation of this surprising result.

Reviewer 5-seed matched rerun: full 0.60157; drop-video 0.60265 (paired Δ +0.00108, positive 5/5); drop-author 0.60289 (paired Δ +0.00132, positive 5/5). See `review_artifacts/c01_ablation_reproduction.json`.

### C02 — Seed variance (5 seeds, official 5-field config)

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Mean | 0.66740 | 0.53574 | 0.60157 |
| Std | 0.00031 | 0.00038 | 0.00032 |

### C03 — Learning-rate sensitivity (3 seeds each, official 5-field config)

| lr | Mean Primary | Std |
|---:|---:|---:|
| 0.0003 | 0.60179 | 0.00011 |
| 0.001 (official) | 0.60144 | 0.00027 |
| 0.003 | 0.60009 | 0.00084 |
| 0.01 | 0.59709 | 0.00053 |

Clear degradation at lr ≥ 0.003 (several std below the official setting); lr=0.0003 is marginally higher than lr=0.001 but within ~1 combined std.

### C05 — Embedding dimension (3 seeds each, official 5-field config, lr=0.001)

| k | Mean Primary | Std |
|---:|---:|---:|
| 8 | 0.60111 | 0.00080 |
| 16 (official) | 0.60144 | 0.00027 |
| 32 | 0.60146 | 0.00069 |
| 64 | 0.60099 | 0.00044 |

All four values are mutually within ~1 combined std — **flat**, independently reproducing the organizer's k=8/16/32 null result (`constraints.md` C6) on VALIDATION and extending it to k=64.

### C04 — Static-feature expansion (3 seeds each, VALIDATION only)

| Config | Fields added | Mean Primary | Std | Δ vs. base |
|---|---|---:|---:|---:|
| base_5field | (none) | 0.60144 | 0.00027 | — |
| item_8field | +music_id, video_type, upload_type | 0.60111 | 0.00046 | -0.00033 (≈0.7 combined std, not significant) |
| cwm_13field | +item_8field fields +follow/register/fans/friend range, user_active_degree | 0.59993 | 0.00052 | -0.00151 (≈2.5 combined std) |

Review correction: these configurations were previously mislabeled `item_9field` and `cwm_14field`; the saved lists contain 8 and 13 fields. Scores are unchanged.

The validation comparison is directionally consistent with existing constraint C5. The retained review evidence is the validation experiment itself; published test results are reference context only.

*(For narrative interpretation of all of the above, see `PRE_AUDIT.md` Investigation C01–C05.)*

---

## 17. Known Evidence Links

```text
A03 train/valid overlap        -> PRE_AUDIT.md#investigation-a03
A04 author/video redundancy    -> PRE_AUDIT.md#investigation-a04
B03 invariant users            -> PRE_AUDIT.md#investigation-b03
B05 oracle/movable gap         -> PRE_AUDIT.md#investigation-b05
B06 joint activity/list analysis -> PRE_AUDIT.md#investigation-b06--joint-activity-tier--list-length-analysis
D02 feedback/long_view assoc.  -> PRE_AUDIT.md#investigation-d02
D04/D05 historical availability -> PRE_AUDIT.md#investigation-d04d05--historical-feedback-availability-overall-and-by-activity-tier
E01 aggregation-window finding  -> PRE_AUDIT.md#investigation-e01--video-basicstatistic-feature-inventory-incl-aggregation-window-inference
A06 date-08 anomaly / decay     -> PRE_AUDIT.md#investigation-a06--temporal-interaction-volume-incl-daily-composition-earlylatevalid-comparison
F01 random-log audit            -> PRE_AUDIT.md#investigation-f01--random-exposure-log-audit
G03 Windows timeout finding     -> PRE_AUDIT.md#investigation-g03--windows-subprocess-timeout--process-tree-recovery
```
