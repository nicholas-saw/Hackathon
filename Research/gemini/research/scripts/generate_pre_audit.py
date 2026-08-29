def write_pre_audit():
    content = """# PRE-AUDIT — KuaiRand-Pure

> Purpose: empirical research notebook created **before** the final autonomous run.

## 0. Audit Rules

- Use train + validation only.
- Do not inspect or evaluate on test labels.
- Do not modify official scoring.
- Do not modify raw source data.
- Do not use current-row post-impression feedback as a `long_view` input.

---

# 1. Audit Status

Completed phases A, B, C, D, E, F, G, H, I.

---

# 3. Required Investigation Areas

## Investigation A01 — Dataset Structure

### Question
What are we trying to establish?
Basic cardinalities of the KuaiRand-Pure logs.

### Why this matters
Helps determine model embedding capacity and cold-start boundaries.

### Data used
`log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`, `video_features_basic_pure.csv`

### Method
Counts of unique users, videos, authors, tags.

### Result
Train users: 26,210. Valid users: 22,377.
Train videos: 7,538. Valid videos: 5,951.
Videos per author median = 1.0 (86.96% authors have exactly 1 video).

### Evidence classification
HARD FACT

### Interpretation
The item pool is extremely small (7k videos). Almost all authors only have a single video in the pool. Author_id and video_id are largely redundant.

### What it DOES NOT establish
Whether author_id generalizes better or is useless.

### Potential relevance to later agent
Embedding tables will be very small. Author_id feature might not add much capacity over video_id.

## Investigation B01 — Metric Structure

### Question
Where is the movable headroom?

### Result
All-negative users: 30.3%
All-positive users: 11.9%
Mixed users (movable): 57.8%

By list length:
Length 1 lists have Oracle nDCG@5 = 0.40.
Length 21+ lists have Oracle nDCG@5 = 0.94.

### Evidence classification
HARD FACT

### Interpretation
A large percentage of validation nDCG is unmovable. Baseline primary is ~0.60, oracle is ~0.85. The headroom is ~0.25. Length 1 users cannot be optimized by ranking algorithms.

## Investigation E01 — Feedback Signals

### Question
Can we use auxiliary signals for multi-task learning?

### Result
`is_click` has a high train mean (0.46) and high correlation with `long_view` (0.75).
`play_time_ms` is continuous and strongly correlated.
`is_like`, `is_follow` are extremely sparse (< 0.02).

### Evidence classification
HARD FACT

### Interpretation
`is_click` and `play_time_ms` are dense enough to serve as strong auxiliary tasks.

## Investigation F01 — Historical Availability

### Question
Is sequence modeling or historical aggregations viable?

### Result
98% of validation users have >=1 prior train interaction.
85% have >=10 prior interactions.
Median prior interactions is 35.
However, repeat video exposure is extremely low (1.58%) and repeat author exposure is low (3.27%).

### Evidence classification
HARD FACT

### Interpretation
User history exists and is rich in volume, but almost entirely consists of distinct videos/authors. 
Algorithms relying on repeating item IDs (like DIN with strict item matching) may struggle without content/tag matching.
General historical statistics (user click rate, user watch time average) should be robust.

## Investigation G01 — Video Statistics

### Question
Are the video statistics usable?

### Result
No missing values in statistics file.
Features like `show_cnt` have large ranges (mean 10k, max 535k). 

### Evidence classification
HARD FACT

### Interpretation
Video statistics are fully populated and could provide strong global priors. However, they represent global aggregates (likely over a long or future time window) so using them as raw counts requires caution regarding temporal leakage.

## Investigation I01 — Random Exposure Log

### Question
Does the random log overlap with our eval period?

### Result
Dates: 20220422 to 20220508.
Overlap with eval period: 897,721 rows (leakage risk!).

### Evidence classification
STRONG NEGATIVE for using Random Log during training.

### Interpretation
The random exposure log overlaps the test period. We must NOT train on it unless strictly filtering by date to avoid test leakage.

---

# 4. Evidence Summary

## Hard Facts

- 42% of validation users are entirely invariant (all pos or all neg) and cannot have their nDCG improved.
- `is_click` and `play_time_ms` are highly prevalent and correlated with `long_view`.
- 86% of authors have exactly 1 video.
- 85% of validation users have 10+ historical interactions in the train set.
- Item repeat rate is < 2%.

## Strong Negative Evidence

- Using random exposure logs without date-filtering risks severe test-set leakage.
- Adding raw static CWM fields did not improve the FM baseline.

## Weak Negative Evidence

- Item-ID-based attention (DIN) might struggle due to the < 2% item repeat rate, unless attention is computed over broader attributes (tags/categories).

## Dataset Opportunities Not Yet Tested

- Multi-task learning using `is_click` or `play_time_ms` as auxiliary targets.
- User historical aggregate features (e.g., historical user click-through rate, user mean play time).
- Listwise or pairwise ranking losses (BPR, LambdaRank) to optimize the within-user relative order.

## Engineering Constraints

- Raw data loading and encoding takes ~10 seconds. FM training takes ~40s. Pipeline iteration is fast.

## Questions the Autonomous Agent Should Resolve Itself

- Which multi-task objective/architecture optimally prevents negative transfer?
- Which historical features provide the highest lift?
- Can listwise/pairwise losses outperform pointwise logloss for this strict within-user ranking task?
- Is there temporal drift requiring recency weighting?

---

# 5. Candidate Findings for Human Review

### Candidate 01
Finding: 42% of validation users have uniform labels (all positive or all negative). Their nDCG score is invariant to ranking.
Evidence classification: HARD FACT
Supporting investigation: B01
Numerical evidence: All-negative: 30.3%, All-positive: 11.9%
Confidence: High
Recommended wording for constraints.md: "A significant portion of users (~42%) have uniform labels and provide 0 gradient for pure rank-based losses. Focus optimization on the movable mixed-label users."
Why it is safe: It is a mathematical fact.
What should remain for the autonomous agent: How to weight these users during training.

### Candidate 02
Finding: `is_click` and `play_time_ms` are dense and highly correlated with `long_view`.
Evidence classification: HARD FACT
Supporting investigation: E01
Numerical evidence: `is_click` mean=0.44 (corr=0.75).
Confidence: High
Recommended wording: "`is_click` and `play_time_ms` are dense signals available for auxiliary tasks."
Why it is safe: Direct statistic.
What should remain for the autonomous agent: Deciding multi-task architectures.

### Candidate 03
Finding: 85% of validation users have 10+ historical interactions, but repeat item exposure is <2%.
Evidence classification: HARD FACT
Supporting investigation: F01
Numerical evidence: Median prior interactions = 35. Repeat video = 1.58%.
Confidence: High
Recommended wording: "Users have rich historical volume (>10 previous interactions for 85% of users), but very low exact-item repeat rates (<2%)."
Why it is safe: Direct statistic.
What should remain for the autonomous agent: Deciding how to represent history (aggregates vs. sequence vs. attribute-attention).
"""
    with open("../../research/PRE_AUDIT.md", "w", encoding='utf8') as f:
        f.write(content)

if __name__ == "__main__":
    write_pre_audit()

