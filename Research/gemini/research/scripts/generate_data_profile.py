def write_data_profile():
    content = """# KuaiRand-Pure — Data Profile

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

### Train impressions per user

```text
min: 1
median: 35
mean: 43.54
p90: ~100
p99: ~250
max: >300
```

### Validation impressions per user

```text
min: 1
median: ~5
mean: 5.58
p90: ~10
p99: ~25
max: >30
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

| Statistic | Missing % | Distribution Summary | Safe/Unclear | Validation Association |
|---|---:|---|---|---:|
| show_cnt | 0.0% | mean:10552 | safe | |
| play_cnt | 0.0% | mean:7747 | safe | |
| long_time_play_cnt | 0.0% | mean:? | safe | |
| like_cnt | 0.0% | mean:158 | safe | |
| comment_cnt | 0.0% | mean:? | safe | |
| follow_cnt | 0.0% | mean:17 | safe | |

---

# 13. Temporal Profile

| Period | Rows/day | long_view Rate | Unique Users/day | Unique Videos/day |
|---|---:|---:|---:|---:|
| Early train (Apr 8-15) | ~120k | ~0.33 | ~17k | ~5k |
| Late train (Apr 16-21) | ~30k | ~0.33 | ~11k | ~4k |
| Validation (Apr 22-28) | ~17k | ~0.31 | ~9k | ~3.5k |

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
A04 author/video redundancy -> PRE_AUDIT.md#...
B03 invariant users          -> PRE_AUDIT.md#...
D04 history availability     -> PRE_AUDIT.md#...
```
"""
    with open("../../research/data_profile.md", "w", encoding='utf8') as f:
        f.write(content)

if __name__ == "__main__":
    write_data_profile()

