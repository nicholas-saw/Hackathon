# PRE-AUDIT — Consolidated KuaiRand-Pure

## 0. Scope and Rules

- Use train + validation only.
- Do not inspect or evaluate on test labels.
- Do not modify official scoring or raw source data.
- Do not use current-row post-impression feedback as a `long_view` input.

## 1. Baseline Reproduction

### Question

Can we reproduce the official baseline evaluation environment?

### Data / scope

Validation logs only (124,909 rows).

### Method

Execute the unmodified baseline FM provided by the organizers on validation data.

### Result

Reproduced seed 0: GAUC 0.6671, nDCG@5 0.5358, Primary 0.6015.
Published: GAUC 0.6674, nDCG@5 0.5357, Primary 0.6016.
5-seed variance (reproduced): mean 0.60157, std 0.00032.

### Evidence classification

HARD FACT

### Interpretation

The baseline environment is successfully reproduced. The published seed standard deviation of 0.0008 is a conservative and reliable threshold for measuring improvement.

### What it DOES NOT establish

Whether the baseline hyperparameters are optimal.

### Source provenance

- Audit 2 (Claude) Section 2
- Audit 3 (GPT) Section 2

## 2. Dataset Structure

### Question

What are the basic cardinalities and missingness of the KuaiRand-Pure logs?

### Data / scope

Train and Validation datasets.

### Method

Counts of unique users, videos, authors, tags, and measurement of null values.

### Result

Train: 1,141,112 rows, 26,210 users, 7,538 videos, 6,482 authors.
Valid: 124,909 rows, 22,377 users, 5,951 videos, 5,315 authors.
Missingness: 0% in core logs and video statistics. 3.15% in `video_duration`, 1.27% in `tag`.

### Evidence classification

HARD FACT

### Interpretation

The item pool is very small (7k videos). Missingness is minimal and confined to a few static fields.

### What it DOES NOT establish

Whether missing data is missing-at-random.

### Source provenance

- Audit 1 (Gemini) A01, C01
- Audit 2 (Claude) A01, A02
- Audit 3 (GPT) A01, A04

## 3. Entity Overlap and Redundancy

### Question

How much do entities overlap between train and validation, and how redundant are authors and videos?

### Data / scope

Train and validation sets.

### Method

Set intersection of IDs and pair frequency counting.

### Result

Users/Videos/Authors generalize well (98.11% users, 99.88% videos, 99.91% authors from valid are in train).
Exact user-video pairs repeat rarely (1.63%), while user-tag pairs repeat often (68.14%).
87.05% of authors have exactly 1 video.

### Evidence classification

HARD FACT

### Interpretation

Cold-start is negligible for basic entities. Models relying on exact `(user, video)` pair repetition will struggle due to sparsity (<2% overlap), but broader generalization (e.g., tags) is dense. `author_id` and `video_id` are highly redundant.

### What it DOES NOT establish

Whether tags provide incremental predictive value over ID features alone.

### Source provenance

- Audit 1 (Gemini) D01
- Audit 2 (Claude) A03, A04, A05
- Audit 3 (GPT) A02, A03, A04

## 4. Metric Structure

### Question

Where is the movable headroom in the evaluation metrics?

### Data / scope

Validation split outcomes.

### Method

Categorizing users by label variance and computing Oracle nDCG@5.

### Result

Uniform-label users: 30.32% all-negative, 11.90% all-positive (42.22% total invariant).
Mixed/movable users: 57.78%.
Length-1 lists (17.51% of users) have 0.000 nDCG gap (Oracle=0.4054, Baseline=0.4054).

### Evidence classification

HARD FACT

### Interpretation

A large percentage of validation nDCG is unmovable. The GAUC metric inherently ignores uniform users. Over 42% of users cannot have their nDCG improved.

### What it DOES NOT establish

How an algorithm should optimally weight these users during training.

### Source provenance

- Audit 1 (Gemini) B01
- Audit 2 (Claude) B03
- Audit 3 (GPT) B01, B03

## 5. Activity / List-Length Analysis

### Question

How is performance distributed across user activity and validation list lengths?

### Data / scope

Validation dataset partitioned by train-activity tiers and validation list lengths.

### Method

Baseline vs. Oracle performance comparison.

### Result

Train-side user activity and validation list length are moderately positively associated (Spearman rho = 0.462).
The highest activity (T3/T4) and list length (6+) intersection accounts for 25.38% of validation users, 50.79% of official GAUC weight, and 51.72% of the baseline-to-oracle primary gap.
_Correction Note:_ All GAUC weight computations use the official denominator (positives from mixed-label users only: 34,592).

### Evidence classification

HARD FACT

### Interpretation

The error (headroom) is highly concentrated in active users with long validation lists. This is a diagnostic concentration, not a causal proof that activity degrades performance.

### What it DOES NOT establish

Which specific model architecture will close this gap.

### Source provenance

- Audit 2 (Claude) B01, B02, B06
- Audit 3 (GPT) B02
- Review 2 (Claude) Corrections R-02

## 6. Baseline Mechanism and Ablations

### Question

How do the core baseline features and capacity configurations perform?

### Data / scope

Validation split using the official Factorization Machine.

### Method

Field ablations and capacity scaling (k=8..64) over 3-5 random seeds.

### Result

Removing `tab` severely hurts performance (-0.0159).
Removing `video_id` or `author_id` slightly improves the baseline (+0.0011, +0.0013) over 5 paired seeds.
Adding 8/13 static CWM fields did not improve performance (0.6011, 0.5999 vs base 0.6014).
Scaling FM embedding dimension (8 to 64) is flat.

### Evidence classification

WEAK NEGATIVE for using both video and author IDs in this exact FM.
STRONG NEGATIVE for simple FM capacity scaling and the exact tested static feature expansion.

### Interpretation

The `tab` feature is critical. High redundancy between author and video causes negative interference in a simple FM. Static field stuffing does not work.

### What it DOES NOT establish

Whether author or static fields are useless in deep models (e.g., DNN, DeepFM) or multi-task setups.

### Source provenance

- Audit 2 (Claude) C01, C04, C05
- Audit 3 (GPT) C02, D01
- Review 2 (Claude) verified 5-seed statistical reliability

## 7. Post-Impression Feedback Structure

### Question

Are auxiliary signals dense and correlated enough for multi-task learning?

### Data / scope

Train and validation datasets.

### Method

Measure prevalence (mean) and same-row Pearson correlation with `long_view`.

### Result

`is_click` prevalence is 44.38% (valid) with r=0.751.
`play_time_ms` mean is 21,487ms (valid) with r=0.632.
`is_like`, `is_follow` (0.130%), `is_comment`, `is_forward` are highly sparse (<2%).

### Evidence classification

HARD FACT (for correlations and density).

### Interpretation

Clicks and play times are dense and strongly associated with long views.

### What it DOES NOT establish

Whether using them as multi-task auxiliary targets will actually yield positive transfer (avoiding negative transfer). Efficacy must be empirically tested.

### Source provenance

- Audit 1 (Gemini) E01
- Audit 2 (Claude) D01, D02
- Audit 3 (GPT) E01, E02 (Issue R-01 corrected `is_follow` to 0.130%)

## 8. Historical Information Availability

### Question

Is sequence modeling viable based on data availability?

### Data / scope

Train log compared to Validation log users.

### Method

Count of strictly prior train interactions per validation user.

### Result

98.11% of validation users have >=1 prior train interaction.
85.17% have >=10 prior interactions. Median prior interactions = 35.
However, repeat video exposure is extremely low (1.63%).

### Evidence classification

HARD FACT

### Interpretation

Users have rich historical interaction volume, but sequence models relying on exact item IDs will struggle due to sparsity. Coarser histories (tags, author, general aggregates) are strongly supported by the volume.

### What it DOES NOT establish

Which granularity of historical aggregation performs best.

### Source provenance

- Audit 1 (Gemini) F01
- Audit 2 (Claude) D04
- Audit 3 (GPT) F01

## 9. Video Basic / Statistic Features

### Question

Are the video statistics reliable and safe to use?

### Data / scope

`video_features_statistic_pure.csv`

### Method

Analysis of statistic redundancy, missingness, and scale.

### Result

Zero missingness. 54 near-duplicate stat pairs exist.
`like_cnt` mean is 230.75.
`show_cnt * counts` is nearly integral, and the reconstructed totals are massively larger (~11,465x) than the sampled train+valid traffic.

### Evidence classification

INCONCLUSIVE (regarding causal safety).
HARD FACT (for numeric summaries).

### Interpretation

The statistics represent a global, external aggregation (likely full platform traffic over an undisclosed window). Because the exact time window is not documented, its causal/leakage safety for the validation/evaluation period is uncertain.

### What it DOES NOT establish

Whether these features leak future evaluation data.

### Source provenance

- Audit 1 (Gemini) G01 (with Review 1 corrections)
- Audit 2 (Claude) E01
- Audit 3 (GPT) G01, G02

## 10. Temporal Structure

### Question

Is interaction volume stationary across the dataset?

### Data / scope

Train and validation logs grouped by date.

### Method

Row counts per day.

### Result

Train volume decays monotonically and sharply from a peak of 278,835 rows on 04-11 down to 20,021 on 04-21 (a 13.9x peak/trough drop), continuing into validation smoothly (~17.8k/day).
The source file contains zero rows for 2022-04-08 (the nominal train start date).

### Evidence classification

HARD FACT

### Interpretation

The dataset is temporally non-stationary in volume. Validation resembles late-train traffic volume.

### What it DOES NOT establish

Why the traffic spike occurred, or whether recency weighting actually improves ranking performance.

### Source provenance

- Audit 1 (Gemini) H01
- Audit 2 (Claude) A06
- Audit 3 (GPT) H01

## 11. Random-Exposure Audit

### Question

Can the random exposure log be used for model training?

### Data / scope

`log_random_4_22_to_5_08_pure.csv`

### Method

Date range analysis and overlap counting (filtering to train/validation periods).

### Result

The random log dates are 2022-04-22 to 2022-05-08.
There are 0 rows overlapping the train window (04-08 to 04-21).

### Evidence classification

STRONG NEGATIVE against training on the random log.

### Interpretation

Because every single row in the random log strictly post-dates the training cutoff, training on it violates temporal ordering and the validation/test boundaries. It may only be used as a validation-period diagnostic.

### What it DOES NOT establish

Whether it is a useful diagnostic.

### Source provenance

- Audit 1 (Gemini) I01
- Audit 2 (Claude) F01
- Audit 3 (GPT) I01
- Review 1/2 (Corrected test-leakage during validation analysis)

## 12. Engineering Feasibility

### Question

What are the runtime requirements and environment readiness?

### Data / scope

Baseline pipeline execution and codebase inventory.

### Method

Runtime profiling and file analysis.

### Result

Baseline training completes in ~1 minute (57s-78s depending on environment/epochs).
All 15 files across `harness/`, `pipeline/`, and `agent/` are empty comment-only scaffolds.

### Evidence classification

ENGINEERING CONSTRAINT

### Interpretation

Iteration speed is extremely fast, making large-scale experimentation feasible. However, the entire agent orchestration and data pipeline layer must be built from scratch before executing experiments.

### What it DOES NOT establish

How to structure the pipeline.

### Source provenance

- Audit 2 (Claude) G01
- Audit 3 (GPT) J01, J02
- Review 3 (GPT) Issue R-02 (Scope undercount fixed to 15 files)

## 13. Model / Objective Evidence

### Question

What modeling approaches are supported by the evidence?

### Data / scope

Synthesis of pre-audit findings.

### Method

Logical deduction from empirical facts.

### Result

- Strict item-matching sequence models (DIN) are poorly supported by the 1.63% item repeat rate.
- `tab` is a crucial interaction feature.
- Auxiliary targets (`is_click`, `play_time_ms`) have strong statistical support but untested transferability.

### Evidence classification

INCONCLUSIVE (pending actual model trials).

### Interpretation

The autonomous agent has a wide open field for research, particularly in multi-task learning, historical aggregation at the tag/author level, and pairwise/listwise loss formulations.

## 14. Evidence Summary

- **HARD FACTS**:
  - The dataset has extreme sparsity in repeat video interactions (<2%) but dense tag histories.
  - Missingness is minimal (<4%).
  - 87% of authors have exactly 1 video (high redundancy).
  - 42.22% of validation users are invariant (all-pos or all-neg).
  - Train volume decays 13.9x from peak to end.
  - 15 core harness/agent files require implementation.

- **STRONG NEGATIVE**:
  - Do NOT train on the random exposure log (temporal leakage).
  - Do NOT use exact static CWM feature stuffing (no baseline benefit).
  - Do NOT assume simple FM capacity scaling improves scores.

- **WEAK NEGATIVE**:
  - Retaining both author and video IDs simultaneously in a shallow FM causes negative interference.

- **INCONCLUSIVE**:
  - Causal safety of global video statistics.
  - Multi-task learning efficacy.
  - Recency weighting efficacy.
