# PRE-AUDIT — KuaiRand-Pure

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

Review note (see research/REVIEW_REPORT.md for full detail): C01, D01, and H01 had supporting raw
data already computed by the original scripts but no investigation write-up in this file — added
during review. One rule-compliance bug (I01, test-window rows touched by a diagnostic) and one crash
bug (F phase inside BEF_analysis.py, superseded by the working F_history.py) were found and fixed.
All numeric claims below were independently re-run and reproduced except where marked otherwise.

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

## Investigation C01 — Missingness

> Review note: this investigation was computed by A_dataset_structure.py (its "A02" print block)
> and the numbers already existed in data_profile.md Section 5, but no formal write-up existed in
> this file despite phase C being marked complete in the Audit Status line. Added on review; the
> underlying numbers were independently re-verified against research/scripts/A_dataset_structure.txt
> and are unchanged.

### Question
Which fields in the static feature files have missing values, and how much?

### Why this matters
Missing-value handling choices (imputation, UNK bucket, drop) affect every downstream feature that
touches these columns; silently ignoring missingness can bias encodings.

### Data used
`video_features_basic_pure.csv`, `user_features_pure.csv`

### Method
`df.isnull().mean()` per column, values > 0 reported.

### Result
Video features: `video_duration` 3.15% missing, `music_type` 2.68% missing, `tag` 1.27% missing.
User features: `onehot_feat4` 3.20% missing, `onehot_feat12`–`onehot_feat17` each 2.62% missing.
No missingness found in the interaction logs themselves or in `video_features_statistic_pure.csv`
(cross-checked against Investigation G01).

### Evidence classification
HARD FACT

### Interpretation
Missingness is low (<4%) and concentrated in a small number of optional/derived fields. This is
survivable with a simple UNK/sentinel bucket per field; it does not by itself argue for or against
using any specific field.

### What it DOES NOT establish
Whether missingness is random (MCAR) or systematically related to `long_view` (e.g., newer videos
missing `video_duration`). That would require a separate check before trusting any missingness-based
imputation strategy.

## Investigation D01 — Train → Validation Overlap

> Review note: computed by A_dataset_structure.py ("A03"/"A05" blocks); numbers already existed in
> data_profile.md Section 4 but had no formal write-up in this file. Added on review; re-verified
> against research/scripts/A_dataset_structure.txt.

### Question
How much entity- and pair-level overlap exists between train and validation?

### Data used
`log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv` (filtered to validation dates only)

### Method
Set intersection of `user_id`, `video_id`, `author_id`, and composite `user_id × {video_id, author_id, tag}`
pairs between train and validation.

### Result
- Validation users seen in train: 98.11% (21,955 / 22,377)
- Validation videos seen in train: 99.88% (5,944 / 5,951)
- Validation authors seen in train: 99.91% (5,310 / 5,315)
- Validation user-video pairs seen in train: 1.63% (1,974 / 121,337)
- Validation user-author pairs seen in train: 3.38% (4,081 / 120,885)
- Validation user-tag pairs seen in train: 68.14% (61,405 / 90,121)
- (For reference, within-train repeat rates: user-video pairs appearing >1 time in train = 4.13%,
  user-author = 5.91%, user-tag = 51.77%.)

### Evidence classification
HARD FACT

### Interpretation
Entity IDs (user/video/author) generalize almost completely from train to validation — cold-start is
a minor concern (<2% of users/videos/authors). But the exact `(user, video)` pairing essentially never
repeats (1.63%): a model cannot rely on having seen the *same* user interact with the *same* video
before. Coarser groupings generalize much better — `user × tag` pairs repeat 68% of the time — which is
consistent with F01's finding that item/author-level history is sparse but broader behavioral/attribute
history is not.

### What it DOES NOT establish
Whether `author_id` or `tag` add incremental value over `video_id` alone in a trained model (that is
an interaction/capacity question, not an overlap question).

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
A large percentage of validation nDCG is unmovable. Baseline primary is 0.6016, oracle (valid) is
0.8484 per `baseline_scores.json` — headroom is ~0.25, matching oracle - baseline directly (not just
"~0.85" rounded). Length-1 lists cannot be reranked at all (their nDCG@5 is simply the label itself);
their achieved nDCG@5 (0.4054) already equals their oracle nDCG@5 (0.4054) — this is a mathematical
identity, not evidence about model quality.

Review addition — activity-tier breakdown and a confound check:

An FM model (baseline.py's `run_fm`, capped at 12 epochs, single seed=0, not the fully-tuned
40-epoch/patience-4 official config) was evaluated per train-activity tier on validation:

| Tier | Users | Valid Rows | GAUC | nDCG@5 | Invariant Users % |
|---|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6741 | 0.5262 | 57.11% |
| T1 (<10) | 2,897 | 8,721 | 0.6475 | 0.5344 | 62.82% |
| T2 (10-49) | 11,138 | 49,716 | 0.6590 | 0.5444 | 45.97% |
| T3 (50-149) | 7,119 | 53,802 | 0.6620 | 0.5282 | 28.99% |
| T4 (150+) | 801 | 10,680 | 0.6856 | 0.4069 | 25.34% |

At first glance T4 (the most active users) has the *lowest* nDCG@5 despite the *highest* GAUC. This
should **not** be read as "the model serves power users worse." T4 users also have the longest
validation lists (highest-activity train users tend to also be highest-activity validation users), and
this dataset's length-bucket table (see below) already shows achieved nDCG@5 falling for long lists
purely from list-length structure — even oracle-independent, achieved nDCG@5 peaks at length 4-5
(0.6140) and declines for 11-20 (0.4875) and 21+ (0.4008). Activity tier and list length are confounded
here; this table alone cannot separate an "activity" effect from a "list length" effect. A future
agent that wants to claim power users are harder to rank should control for list length first.

Caveat: unlike the oracle numbers (model-independent), the achieved GAUC/nDCG@5 per bucket come from
one single-seed, capacity-limited (12-epoch) FM run, not the fully-converged official baseline
(40 epochs, patience 4). The published baseline's seed std is ~0.0008 on the *overall* metric; per-tier
subsets are smaller samples and likely carry more run-to-run noise than that. Treat the relative
ordering across tiers as indicative, not the exact decimal values.

## Investigation E01 — Feedback Signals

### Question
Can we use auxiliary signals for multi-task learning?

### Result
`is_click` mean = 0.4634 (train) / 0.4438 (validation); correlation with `long_view` = 0.7515
(computed on the validation split — see review note below).
`play_time_ms` mean = 23,260 ms (train) / 21,487 ms (validation); correlation = 0.6319.
`is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `is_profile_enter` are all sparse
(<0.03 mean); `is_hate` is essentially uncorrelated with `long_view` (-0.0038).

### Evidence classification
HARD FACT (density and correlation numbers only)

### Interpretation
`is_click` and `play_time_ms` are dense signals with strong same-row association with `long_view`
(this is expected: a "long view" is definitionally downstream of a click and of watch duration, so
part of this correlation is close to tautological rather than a novel discovery). Density and
correlation establish that these signals *exist* and are *measurable* in enough rows to be usable.

Review correction: the original wording ("dense enough to **serve as strong auxiliary tasks**")
overstates what this investigation established. Correlation between two post-impression outcomes
measured on the same row says nothing about whether an auxiliary *prediction head* for `is_click` or
`play_time_ms` would improve a shared representation for `long_view` — that is a multi-task transfer
question that has not been tested and per RULES.md must be measured empirically, not assumed. Reworded
below as a density/availability fact rather than an efficacy claim.

### What it DOES NOT establish
Whether using these as multi-task auxiliary targets actually improves `long_view` ranking (negative
transfer is a real possibility per RULES.md §5 and must be tested).

## Investigation F01 — Historical Availability

### Question
Is sequence modeling or historical aggregations viable?

### Result
98% of validation users have >=1 prior train interaction.
85% have >=10 prior interactions.
Median prior interactions is 35 (median, over the 22,377 validation users, of each user's own train-side
row count — a different population/quantity from "median train impressions across all 26,210 train
users," which is 31; see data_profile.md §7 review correction — these two numbers were previously
conflated).
However, repeat video exposure is extremely low (1.58%) and repeat author exposure is low (3.27%).

### Evidence classification
HARD FACT

### Interpretation
User history exists and is rich in volume, but almost entirely consists of distinct videos/authors. 
Algorithms relying on repeating item IDs (like DIN with strict item matching) may struggle without content/tag matching.

Review edit: the closing sentence ("General historical statistics ... should be robust") was a
plausible but untested hypothesis, not a measured result — softened accordingly. Volume alone does not
guarantee a historical aggregate (e.g. prior click rate) will be predictive or well-calibrated; that
still needs to be measured once such a feature is built.

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

Review note: data_profile.md §12 previously stated `like_cnt` mean as 158 and left `long_time_play_cnt`
and `comment_cnt` as unmeasured (`mean:?`). Direct recomputation gives `like_cnt` mean = 230.75 (not
158), `long_time_play_cnt` mean = 3,687, `comment_cnt` mean = 12.93 — corrected in data_profile.md.
This does not change G01's classification or interpretation (the caution about aggregation-window
provenance still applies to all of these fields), only the specific numbers.

## Investigation H01 — Temporal Volume Profile

> Review note: raw daily counts were already computed by A_dataset_structure.py ("A06" block, see
> research/scripts/A_dataset_structure.txt) and summarized in data_profile.md Section 13, but no
> formal investigation write-up existed despite phase H being marked complete. Added on review after
> independently reproducing the finding (research/review_scripts/H_temporal_review.py,
> research/review_artifacts/H_temporal_review_output.txt). This surfaces two facts the original
> 3-bucket summary in data_profile.md obscured.

### Question
How stationary is interaction volume across the train/validation window?

### Data used
`log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv` (validation dates only)

### Method
Row counts grouped by `date`.

### Result
1. The official train window is defined as 2022-04-08..2022-04-21, but the raw log file has **zero
   rows on 2022-04-08** — the earliest date actually present is 2022-04-09. Total train row count
   (1,141,112) is unaffected; this is purely a labeling/coverage quirk of the source file, not a
   missing-data problem to fix.
2. Daily volume is highly non-stationary within train, not a flat "~120k/day": 52,736 (04-09) →
   227,808 (04-10) → 278,835 peak (04-11) → 166,076 → 94,711 → 58,892 → 60,904 → 44,023 → 24,560 →
   20,443 → 20,851 → 20,021 (04-21). Peak/trough ratio within the train window alone is **13.9x**.
   Mean rows/day for 04-09..04-15 is 135,759 vs 31,800 for 04-16..04-21 vs 17,844 for validation
   (04-22..04-28) — a roughly monotonic decay after the 04-11 peak that continues smoothly into and
   through validation.

### Evidence classification
HARD FACT

### Interpretation
The dataset is not temporally stationary. Row volume decays by an order of magnitude from the early
peak to the end of train, and continues declining into validation at a similar rate — i.e. the
train→validation volume transition looks like a continuation of the same late-train trend, not a
regime break. This is relevant to any recency-weighting or temporal-feature hypothesis (references.md
§11): the "trend" a recency scheme would need to track is a strong, smooth decay, not noise. `long_view`
rate itself is comparatively stable across the same period (~0.29-0.38, no comparable order-of-magnitude
swing — see A06 daily table), so the volume drift is a traffic/exposure-count phenomenon, not obviously
a label-rate drift.

### What it DOES NOT establish
The *cause* of the early-train spike (e.g. a logging artifact, a promotional event, or a cohort
onboarding wave) — that is not recoverable from this file alone. It also does not establish that
recency weighting would help the scored metric; that must be tested.

## Investigation I01 — Random Exposure Log

### Question
Does the random log overlap with our eval period?

### Result
Dates: 20220422 to 20220508. Total rows: 1,186,059.
Rows falling in the test window (04-29..05-08): 897,721.
Rows falling in the validation window (04-22..04-28): 288,338.
Rows falling in the train window (04-08..04-21): **0**.

### Evidence classification
STRONG NEGATIVE for using the Random Log, unfiltered, during training.

### Interpretation
The random exposure log's date range (04-22..05-08) does not overlap the train window **at all** —
every single row in it postdates the train cutoff. The risk is broader than "the 897,721 rows that
fall in the eval/test window": using *any* row from this file as training data would mean training on
information collected strictly after the train cutoff, which breaks temporal ordering regardless of
whether the specific row's date happens to fall in the validation or test sub-range. The 288,338
validation-dated rows are not "safe to train on" — they are simply the sub-range that also isn't
provably test-contaminated. Per DATA_GUIDE.md §7, a defensible use of this file is as a
validation-time-window diagnostic/counterfactual set (e.g. for exposure-bias analysis), not as
additional training signal.

Review correction: the "Random UV pairs also in standard logs" diagnostic in
`research/scripts/GI_analysis.py` was originally computed against the **unfiltered**
`log_standard_4_22_to_5_08_pure.csv`, which includes test-window rows — a "train+validation only" rule
violation (structural/ID overlap only, no test labels were read or used). Corrected count, train+valid
only: **702 / 1,186,006 (0.06%)** — versus the original **759 (0.06%)**, of which 58 pairs came only
from the test-window contamination. The rounded headline percentage is unchanged, but the script has
been fixed (see research/scripts/GI_analysis.py and research/review_scripts/I_random_log_review.py)
so it no longer touches test-period rows.

---

# 4. Evidence Summary

## Hard Facts

- 42% of validation users are entirely invariant (all pos or all neg) and cannot have their nDCG improved.
- `is_click` and `play_time_ms` are dense signals with strong same-row correlation with `long_view`
  (interpret with care — see E01 review note on tautology risk before treating this as evidence that
  they will help as auxiliary tasks).
- 86% of authors have exactly 1 video.
- 85% of validation users have 10+ historical interactions in the train set; exact item/author repeat
  rate is < 2%, but user-tag pair overlap between train and validation is 68% (D01).
- Interaction volume decays ~14x from its early-train peak (04-11) to the end of train, and continues
  declining smoothly into validation (H01) — the dataset is not temporally stationary.
- The raw train log file has zero rows on 2022-04-08 despite that being the official train start date
  (H01) — a source-file coverage quirk to be aware of, not a bug to fix.

## Strong Negative Evidence

- Using the random exposure log as training data is unsafe: its date range does not overlap the train
  window at all (0 rows before 04-22), so any use of it in training breaks temporal ordering — not
  only the portion that falls in the test window.
- Adding raw static CWM fields did not improve the FM baseline. (Organizer-provided result; see
  constraints.md C5. Not reproduced locally, and does not need to be — it is pre-existing organizer
  evidence, not a pre-audit finding.)

## Weak Negative Evidence

- Item-ID-based attention (DIN) might struggle due to the < 2% item repeat rate, unless attention is computed over broader attributes (tags/categories).

## Dataset Opportunities Not Yet Tested

- Multi-task learning using `is_click` or `play_time_ms` as auxiliary targets (efficacy, not just
  density/correlation, remains untested).
- User historical aggregate features (e.g., historical user click-through rate, user mean play time).
- Listwise or pairwise ranking losses (BPR, LambdaRank) to optimize the within-user relative order.
- Coarser generalization keys (tag-level, author-level) as a way to work around the near-zero exact
  user-video repeat rate (D01).
- Recency weighting or explicit time features, motivated by the strong volume decay found in H01
  (not yet motivated or ruled out on the *label rate*, only on raw volume).

## Engineering Constraints

- Raw data loading and encoding takes ~5-10 seconds. A single FM training run (baseline.py's default
  config) takes on the order of 40-50 seconds on CPU depending on epoch count/early stopping; exact
  figures varied slightly between the official README's stated ~40s and this pre-audit's own measured
  ~50s (see data_profile.md Engineering Profile) — treat as an order-of-magnitude estimate, not a
  precise benchmark. Pipeline iteration is fast either way.

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
Recommended wording for constraints.md: "~42% of validation users (30.3% all-negative, 11.9% all-positive) have uniform `long_view` labels within their impression list; their nDCG@5 is mathematically fixed regardless of ranking, and they do not contribute to GAUC."
Why it is safe: It is a mathematical fact, directly following from evaluate.py's semantics.
Review note: the original wording added an action clause ("Focus optimization on the movable
mixed-label users") — removed. Per constraints.md's own stated policy, this file should record
verified facts, not strategy directives; how to weight or prioritize users is a decision for the
autonomous agent.
What should remain for the autonomous agent: How to weight or prioritize these users during training.

### Candidate 02
Finding: `is_click` and `play_time_ms` are dense signals with strong same-row statistical association with `long_view`.
Evidence classification: HARD FACT
Supporting investigation: E01
Numerical evidence: `is_click` mean = 0.4634 (train) / 0.4438 (validation), correlation with `long_view` (validation) = 0.7515. `play_time_ms` correlation (validation) = 0.6319.
Confidence: High for the density/correlation numbers themselves. Low/untested for any claim about auxiliary-task efficacy.
Recommended wording: "`is_click` and `play_time_ms` are dense signals (present on effectively all rows) with strong same-row statistical association with `long_view` in validation. This does not establish that they are effective multi-task auxiliary targets — that must be tested empirically."
Why it is safe: Direct statistic, correctly labeled as density/correlation only.
Review note: original wording said these signals are "dense enough to serve as strong auxiliary
tasks" — this conflates an availability fact with an untested efficacy claim; reworded to separate
the two per RULES.md §5 ("do not assume every auxiliary task helps").
What should remain for the autonomous agent: Deciding multi-task architectures and empirically testing for negative transfer.

### Candidate 03
Finding: 85% of validation users have 10+ historical interactions, but repeat item exposure is <2%.
Evidence classification: HARD FACT
Supporting investigation: F01
Numerical evidence: Median prior interactions = 35. Repeat video = 1.58%.
Confidence: High
Recommended wording: "Users have rich historical volume (>10 previous interactions for 85% of users), but very low exact-item repeat rates (<2%)."
Why it is safe: Direct statistic.
What should remain for the autonomous agent: Deciding how to represent history (aggregates vs. sequence vs. attribute-attention).

### Candidate 04 (added on review)
Finding: Interaction volume is highly non-stationary within the official train window and continues
decaying into validation; the raw train log also has zero rows on the nominal train-start date
2022-04-08.
Evidence classification: HARD FACT
Supporting investigation: H01
Numerical evidence: Daily rows range from 20,021 to 278,835 within train (13.9x peak/trough ratio);
mean 135,759/day (04-09..04-15) vs 31,800/day (04-16..04-21) vs 17,844/day in validation (04-22..04-28).
Confidence: High
Recommended wording: "Interaction volume decays roughly monotonically and by more than an order of
magnitude from its early-train peak (2022-04-11) through the end of train and into validation; the
raw train log file has zero rows on 2022-04-08 (the nominal train start date), with the earliest actual
date being 2022-04-09."
Why it is safe: Direct count from the raw source files, independently reproduced.
What should remain for the autonomous agent: Whether and how to use recency weighting or temporal
features, and whether `long_view` rate (as opposed to raw volume) drifts enough to matter.

### Candidate 05 (added on review)
Finding: User/video/author IDs generalize almost completely from train to validation, but the exact
(user, video) pairing essentially never repeats; coarser groupings (user × tag) repeat far more often.
Evidence classification: HARD FACT
Supporting investigation: D01
Numerical evidence: Validation users/videos/authors seen in train: 98.11% / 99.88% / 99.91%.
Validation user-video pairs seen in train: 1.63%. Validation user-author pairs seen in train: 3.38%.
Validation user-tag pairs seen in train: 68.14%.
Confidence: High
Recommended wording: "Entity IDs generalize from train to validation almost completely (>98% for
users/videos/authors), but exact (user, video) pair repetition is rare (1.63%); (user, tag) pair
repetition is far more common (68.14%)."
Why it is safe: Direct set-overlap statistic.
What should remain for the autonomous agent: Whether/how to exploit coarser (tag/author-level)
generalization keys given the near-absence of exact pair repetition.
