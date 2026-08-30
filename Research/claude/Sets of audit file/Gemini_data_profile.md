# KuaiRand-Pure — Data Profile

> Purpose: compact, agent-readable snapshot of dataset and baseline measurements.
> Populate this file from the pre-audit.
>
> Keep **observations** separate from **interpretations**.

## 1. Official Dataset Summary

```text
dataset: KuaiRand-Pure
task: within-user ranking
target: long_view
metrics: GAUC, nDCG@5
primary: mean(GAUC, nDCG@5)
```

### Official splits

| Split | Dates | Rows |
|---|---|---:|
| Train | 2022-04-08 .. 2022-04-21 | 1,141,112 |
| Validation | 2022-04-22 .. 2022-04-28 | 124,909 |
| Evaluation/Test | 2022-04-29 .. 2022-05-08 | 170,588 |

Do not inspect evaluation/test labels during development.

---

## 2. Official Baseline

| Metric | Validation |
|---|---:|
| GAUC | 0.6674 |
| nDCG@5 | 0.5357 |
| Primary | 0.6016 |

Baseline model:

```text
Factorization Machine
k = 16
lr = 0.001

fields:
- user_id
- video_id
- author_id
- tab
- dur_bucket
```

Published seed std is approximately `0.0008`.

---

# 3. Entity Cardinalities

Populate from train and validation.

| Measurement | Train | Validation | Notes |
|---|---:|---:|---|
| Unique users | 26210 | 22377 | |
| Unique videos | 7538 | 5951 | |
| Unique authors | 6482 | 5315 | |
| Unique tags | 110 | 104 | |
| Unique music IDs | N/A | N/A | Not measured yet but similar order to videos |

---

# 4. Train → Validation Overlap

| Measurement | Value |
|---|---:|
| Validation users seen in train | 98.11% |
| Validation videos seen in train | 99.88% |
| Validation authors seen in train | 99.91% |
| Validation user-video pairs seen in train | 1.63% |
| Validation user-author pairs seen in train | 3.38% |
| Validation user-tag pairs seen in train | 68.14% |

---

# 5. Missingness

| File / Field | Missing % |
|---|---:|
| video_features_basic: video_duration | 3.15% |
| video_features_basic: music_type | 2.68% |
| video_features_basic: tag | 1.27% |
| user_features: onehot_feat4 | 3.20% |
| user_features: onehot_feat12-17 | 2.62% |

---

# 6. Author / Video Structure

| Measurement | Value |
|---|---:|
| Videos per author — median | 1.0 |
| Videos per author — mean | 1.16 |
| Authors with exactly 1 video | 5661 |
| % authors with exactly 1 video | 86.96% |

Interpretation belongs in `PRE_AUDIT.md`, not here.

---

# 7. User Activity

Review correction: both tables below were re-verified by direct recomputation. The train-side median
(previously 35) and p99 (previously ~250) did not reproduce and have been corrected to 31 and 207.
The previous "35" appears to have been carried over from Investigation F01's "median prior train
interactions" statistic, which is a *different* population (only validation users, restricted to their
train-side history) from "median train impressions across all train users" — the two happen to be
close but are not the same quantity. Max values are given exactly rather than as a vague lower bound.

### Train impressions per user (26,210 users)

```text
min: 1
median: 31
mean: 43.54
p90: 97
p99: 207
max: 809
```

### Validation impressions per user (22,377 users)

```text
min: 1
median: 4
mean: 5.58
p90: 12
p99: 26
max: 74
```

---

# 8. Metric Structure

### Uniform-label users — validation

| Type | Users | % Users |
|---|---:|---:|
| All negative | 6785 | 30.32% |
| All positive | 2663 | 11.90% |
| Mixed / movable | 12929 | 57.78% |

### List-length buckets — validation

| List Length | Users | nDCG@5 | Oracle nDCG@5 | GAUC Positives |
|---|---:|---:|---:|---:|
| 1 | 3917 | 0.4054 | 0.4054 | 1588 |
| 2-3 | 6218 | 0.5413 | 0.6086 | 5595 |
| 4-5 | 4119 | 0.6140 | 0.7492 | 6309 |
| 6-10 | 5225 | 0.5880 | 0.8536 | 12847 |
| 11-20 | 2346 | 0.4875 | 0.9182 | 9368 |
| 21+ | 552 | 0.4008 | 0.9420 | 3425 |

---

# 9. Activity Buckets

Define buckets from train-side activity only.

GAUC/nDCG@5 columns come from a single-seed, 12-epoch-capped FM run (not the fully-tuned 40-epoch
official baseline config) — treat as indicative of relative ordering, not precise values. See
`PRE_AUDIT.md` Investigation B01 for a list-length confound noted on review (T4's low nDCG@5 likely
reflects its longer validation lists, not activity per se).

| Tier | Users | Validation Rows | GAUC | nDCG@5 | Fixed Users % |
|---|---:|---:|---:|---:|---:|
| Cold | 422 | 1990 | 0.6741 | 0.5262 | 57.11% |
| T1 (<10) | 2897 | 8721 | 0.6475 | 0.5344 | 62.82% |
| T2 (10-49) | 11138 | 49716 | 0.6590 | 0.5444 | 45.97% |
| T3 (50-149) | 7119 | 53802 | 0.6620 | 0.5282 | 28.99% |
| T4 (150+) | 801 | 10680 | 0.6856 | 0.4069 | 25.34% |

---

# 10. Feedback Signal Profile

Use train + validation only and respect temporal order.

| Signal | Train Positive/Mean | Validation Positive/Mean | Association with long_view | Notes |
|---|---:|---:|---:|---|
| is_click | 0.4634 | 0.4438 | 0.7515 | Strong correlation |
| is_like | 0.0187 | 0.0180 | 0.0949 | Sparse |
| is_follow | 0.0010 | 0.0013 | 0.0253 | Very sparse |
| is_comment | 0.0026 | 0.0023 | 0.0587 | Very sparse |
| is_forward | 0.0010 | 0.0008 | 0.0245 | Very sparse |
| is_hate | 0.0004 | 0.0006 | -0.0038 | Extremely sparse |
| play_time_ms | 23260 | 21486 | 0.6319 | Strong correlation |
| profile_stay_time | 3.3060 | 1.8762 | -0.0005 | Uncorrelated |
| comment_stay_time | 552.94 | 460.28 | 0.1692 | Weak correlation |
| is_profile_enter | 0.0254 | 0.0195 | 0.1271 | Sparse |

---

# 11. Historical Availability

### Overall

| Measurement | Value |
|---|---:|
| % validation users with >=1 prior train interaction | 98.11% |
| >=5 | 92.85% |
| >=10 | 85.17% |
| Median prior interactions | 35 |
| Repeat-video history coverage | 1.58% |
| Repeat-author history coverage | 3.27% |

### By activity tier

| Tier | Median Prior Rows | >=1 | >=5 | >=10 | Repeat Video | Repeat Author |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| T1 | 5 | 100.00% | 59.37% | 0.00% | 0.63% | 1.16% |
| T2 | 27 | 100.00% | 100.00% | 100.00% | 1.04% | 2.31% |
| T3 | 75 | 100.00% | 100.00% | 100.00% | 1.77% | 3.78% |
| T4 | 184 | 100.00% | 100.00% | 100.00% | 4.21% | 7.43% |

---

# 12. Video Statistics

Review correction: `like_cnt` mean did not reproduce (claimed 158, actual 230.75) and has been
corrected. The two `mean:?` placeholders have been filled in from direct recomputation (median added
for context since these distributions are heavily right-skewed — see A03/G01 caution about scale).

| Statistic | Missing % | Mean | Median | Safe/Unclear | Validation Association |
|---|---:|---:|---:|---|---:|
| show_cnt | 0.0% | 10,552 | 4,519 | safe (structural stat, see G01 caution on window) | not measured |
| play_cnt | 0.0% | 7,747 | 2,560 | safe | not measured |
| long_time_play_cnt | 0.0% | 3,687 | 978 | safe | not measured |
| like_cnt | 0.0% | 230.75 | 57.54 | safe | not measured |
| comment_cnt | 0.0% | 12.93 | 2.46 | safe | not measured |
| follow_cnt | 0.0% | 17.41 | 3.80 | safe | not measured |

---

# 13. Temporal Profile

Review correction: the 3-bucket summary below understated how non-stationary daily volume actually is
and omitted that the raw train log has zero rows on 2022-04-08 (the nominal train start date — the
earliest date actually present is 2022-04-09). See `PRE_AUDIT.md` Investigation H01 for the
interpretation and `research/review_artifacts/H_temporal_review_output.txt` for the reproduced numbers.

| Date | Rows | long_view Rate | Unique Users | Unique Videos |
|---|---:|---:|---:|---:|
| 2022-04-08 | 0 | — | — | — |
| 2022-04-09 | 52,736 | 0.3362 | 13,561 | 1,933 |
| 2022-04-10 | 227,808 | 0.3409 | 21,011 | 5,258 |
| 2022-04-11 (peak) | 278,835 | 0.3330 | 21,110 | 7,090 |
| 2022-04-12 | 166,076 | 0.3322 | 20,073 | 6,380 |
| 2022-04-13 | 94,711 | 0.3171 | 18,163 | 5,691 |
| 2022-04-14 | 71,252 | 0.3195 | 16,910 | 5,189 |
| 2022-04-15 | 58,892 | 0.3634 | 16,035 | 5,019 |
| 2022-04-16 | 60,904 | 0.3770 | 15,520 | 4,906 |
| 2022-04-17 | 44,023 | 0.3644 | 14,545 | 4,603 |
| 2022-04-18 | 24,560 | 0.3308 | 11,025 | 3,953 |
| 2022-04-19 | 20,443 | 0.3229 | 9,909 | 3,670 |
| 2022-04-20 | 20,851 | 0.3114 | 10,076 | 3,632 |
| 2022-04-21 (train end) | 20,021 | 0.3146 | 9,858 | 3,669 |
| 2022-04-22..28 (validation, 7 days) | 17,844 avg/day | ~0.29-0.34 | ~8-11k/day | ~3.0-3.8k/day |

Summary buckets (kept for quick reference, but see the daily table above for the actual shape —
peak/trough ratio within train alone is 13.9x, not a flat rate):

| Period | Mean Rows/day |
|---|---:|
| Train days with data before peak (Apr 9-11) | ~186k |
| Train days after peak, still "early" (Apr 12-15) | ~90k |
| Late train (Apr 16-21) | ~31.8k |
| Validation (Apr 22-28) | ~17.8k |

---

# 14. Engineering Profile

| Stage | Time | Notes |
|---|---:|---|
| CSV load | ~5s | |
| baseline encoding | ~5s | |
| baseline training | ~50s | CPU, 12-40 epochs |
| validation evaluation | ~1s | |

### Environment

```text
OS: Windows
Python: 3.10
CPU: Used for training FM
```

---

# 15. Known Evidence Links

```text
A01 dataset cardinalities     -> PRE_AUDIT.md Investigation A01
A04 author/video redundancy   -> PRE_AUDIT.md Investigation A01
C01 missingness                -> PRE_AUDIT.md Investigation C01 (added on review)
D01 train/valid overlap        -> PRE_AUDIT.md Investigation D01 (added on review)
B01 invariant users / headroom -> PRE_AUDIT.md Investigation B01
E01 feedback signal profile    -> PRE_AUDIT.md Investigation E01
F01 history availability       -> PRE_AUDIT.md Investigation F01
G01 video statistics           -> PRE_AUDIT.md Investigation G01
H01 temporal volume profile    -> PRE_AUDIT.md Investigation H01 (added on review)
I01 random exposure log        -> PRE_AUDIT.md Investigation I01 (corrected on review)
```
