# PRE-AUDIT — KuaiRand-Pure

> Purpose: empirical research notebook created **before** the final autonomous run.
>
> Goal: establish reliable facts, negative evidence, feasibility constraints, and unresolved questions.
>
> Do **not** use this phase to manually solve the entire competition.
> Give the future agent evidence, not the answer.

## 0. Audit Rules

- Use train + validation only.
- Do not inspect or evaluate on test labels.
- Do not modify official scoring.
- Do not modify raw source data.
- Do not use current-row post-impression feedback as a `long_view` input.
- Multi-task auxiliary targets are permitted.
- Historical feedback must be strictly earlier than the row being represented.
- Reproduce surprising results before trusting them.

### Evidence classes

- `HARD FACT`
- `STRONG NEGATIVE`
- `WEAK NEGATIVE`
- `INCONCLUSIVE`

### Environment / methodology note

All measurements below were produced by scripts under `research/scripts/`, which load data
through `research/scripts/common.py`. That loader **never returns test-range rows**: the
shared file `log_standard_4_22_to_5_08_pure.csv` (which spans both validation and test dates)
is scanned row by row; only the date field is inspected for out-of-range rows, and complete
evaluation rows (including labels) are never stored, printed, or scored. The
one exception, `common.count_test_rows_only()`, returns only an integer row count (for split-size
confirmation) and touches no label or feature value. During the original pre-audit, the
unmodified official baseline script was mistakenly allowed to score the local test split once,
and `phase_i_random_log.py` summarized evaluation-period random-log outcomes. Both actions
violated the pre-audit's train+validation-only rule even though neither result was used for model
selection. Review correction: the local test metrics and evaluation-period outcome summaries
have been removed; the loaders/scripts now materialize only train/validation rows, and retain
only date-only evaluation row counts. All model experiments (Phase
C ablations, seed variance, lr/dimension sweeps, static-feature reproduction) use a
research-only encoder (`fm_utils.py`) built directly on `common.py`, which has no code path that
can reach test rows.

The official FM model class (`baseline.FM`, pure numpy, no data access) and the official
`evaluate()` function are imported unmodified from `source/starter-kit/`.

---

# 1. Audit Status

| ID | Investigation | Status | Evidence Class | Short Result |
|---|---|---|---|---|
| A01 | Basic dataset cardinalities | DONE | HARD FACT | 26,210 train users / 7,538 train videos / 6,482 train authors |
| A02 | Missingness profile | DONE | HARD FACT | Logs 0% missing; video_basic up to 3.15% (video_duration); user_features up to 3.20% |
| A03 | Train→validation entity overlap | DONE | HARD FACT | 98.1% user / 99.9% video overlap; only 1.6% user-video PAIR overlap |
| A04 | Author→video redundancy | DONE | HARD FACT | 87.0% of authors have exactly 1 video |
| A05 | Repeat-pair / affinity coverage | DONE | HARD FACT | user-video repeat 4.1% (train) / 2.9% (valid) of pairs |
| A06 | Temporal interaction volume | DONE | HARD FACT | 04-08 has 0 rows; 04-11 peak is 14x 04-21 volume |
| B01 | Activity buckets | DONE — REVIEW CORRECTED | HARD FACT | T4 carries 34.79% of official GAUC weight and has the largest bucket nDCG gap (0.262) |
| B02 | List-length buckets | DONE — REVIEW CORRECTED | HARD FACT | Lists of 6-10 carry largest official GAUC weight share (36.39%); gap grows with length |
| B03 | Uniform-label / invariant users | DONE | HARD FACT | 30.3% all-negative, 11.9% all-positive, 17.5% single-impression (validation) |
| B04 | GAUC weight concentration | DONE — REVIEW CORRECTED | HARD FACT | See corrected B01/B02 weight-share columns |
| B05 | Oracle/movable gap by bucket | DONE | HARD FACT | Local valid oracle primary 0.8484, matches published exactly |
| B06 | Activity-tier × list-length joint analysis | DONE — REVIEW EXTENSION | HARD FACT | Moderate positive association (Spearman ρ=0.462); T3/T4 × 6+ holds 50.79% of GAUC weight |
| C01 | Baseline field ablations | DONE — REVIEW REPRODUCED | HARD FACT + WEAK NEGATIVE | 5-seed rerun: drop-video +0.00108, drop-author +0.00132, positive 5/5 each |
| C02 | Seed variance | DONE | HARD FACT | 5-seed local std 0.00032 (primary), same order as published 0.0008 |
| C03 | Learning-rate sensitivity | DONE | HARD FACT | Clear degradation at lr≥0.003; official lr=0.001 near-optimal |
| C04 | Static feature expansion check | DONE — REVIEW CORRECTED | STRONG NEGATIVE (exact formulation) | cwm_13field -0.0015 vs base; prior artifact mislabeled field counts as 9/14 instead of 8/13 |
| C05 | FM dimension check | DONE | STRONG NEGATIVE (reproduced on valid) | k=8/16/32/64 mutually within ~1 std — flat |
| D01 | Feedback-label prevalence | DONE | HARD FACT | is_click 44.4%, is_like 1.8%, is_follow 0.13% (valid) |
| D02 | Feedback↔long_view association | DONE | HARD FACT | is_click r=0.75, play_time_ms r=0.63 (same-row, diagnostic only) |
| D03 | Inter-feedback association | DONE | HARD FACT | See correlation matrix in phase_d_feedback.json |
| D04 | Historical feedback availability | DONE | HARD FACT | 85.2% of valid users have ≥10 prior train interactions |
| D05 | Historical availability by activity | DONE | HARD FACT | Cold=0% history by construction; T2+ = 100% ≥10 prior |
| E01 | Video-statistics inventory | DONE | HARD FACT + INCONCLUSIVE | Stats are per-`counts` averages; aggregation window undocumented |
| E02 | Ratio-feature associations (safety unresolved) | DONE — REVIEW CORRECTED | HARD FACT association + INCONCLUSIVE safety | long_time_play ratio r=0.30 with long_view (valid) |
| F01 | Random-exposure date/overlap audit | DONE | HARD FACT | 1,186,059 rows; only 0.006% uv-pair overlap with standard valid |
| G01 | CSV load + encoding runtime | DONE — REVIEW RERUN | HARD FACT (run-specific) | Load 3.0s, encode 4.8s, FM train 49.7s (11 epochs) |
| G02 | Cache speedup | DONE — REVIEW RERUN | HARD FACT (run-specific) | Pickle cache: 263x faster reload, bit-identical arrays |
| G03 | Windows timeout/process recovery | DONE | HARD FACT | `subprocess.run(timeout=3)` blocked 30.1s due to inherited grandchild pipe handles |
| G04 | NaN/syntax recovery | DONE | HARD FACT | Both cleanly detected by existing official/stdlib mechanisms |

---

# 2. Investigation Template

Copy this block for every investigation.

```markdown
## Investigation <ID> — <Title>

### Question
What are we trying to establish?

### Why this matters
Why does the later autonomous agent need this fact?

### Data used
Train / validation files and columns used.

### Method
Exact computation or experiment.

### Result
Numerical result.

### Evidence classification
HARD FACT / STRONG NEGATIVE / WEAK NEGATIVE / INCONCLUSIVE

### Interpretation
What does the result establish?

### What it DOES NOT establish
State the boundary of the conclusion.

### Potential relevance to later agent
What new information does the agent now have?

### Artifacts
Scripts, JSON, tables, plots, logs.
```

---

# 3. Baseline Reproduction (pre-condition for everything below)

### Method
The original pre-audit ran the unmodified official script, which also scored test and therefore
violated the train+validation-only rule. Review reproduction instead used the official FM class
and official `evaluate()` function with the research loader/encoder, which exposes train and
validation only.

### Result

```text
{'train': 1141112, 'valid': 124909, 'test': 170588} fields=['user_id','video_id','author_id','tab','dur_bucket']
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
```

Published validation: GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016; published seed std 0.0008.

All validation differences (≤0.0003) are inside the published seed std. Row counts match exactly (1,141,112 / 124,909 / 170,588), with evaluation size verified from dates only.

### Evidence classification
HARD FACT.

### Interpretation
Environment, train/validation data, and evaluator reproduce the published validation baseline. The prior local test evaluation was unnecessary and noncompliant; it has no bearing on any retained conclusion.

### What it DOES NOT establish
Does not establish anything about feature usefulness or model quality beyond the baseline itself.

### Artifacts
Console output above; `source/starter-kit/baseline.py` (unmodified).

---

# 4. Investigations

## Investigation A01 — Basic dataset cardinalities

### Question
How many unique users/videos/authors/tags exist in train vs. validation, and what are the major categorical cardinalities?

### Why this matters
Sizes vocabulary/embedding tables and bounds what "cold-start" or "long-tail" even means numerically for this dataset.

### Data used
Train log, validation log, `video_features_basic_pure.csv`, `user_features_pure.csv`.

### Method
`nunique()` counts per split; see `research/scripts/phase_a_structure.py`.

### Result

| Measurement | Train | Validation |
|---|---:|---:|
| Unique users | 26,210 | 22,377 |
| Unique videos | 7,538 | 5,951 |
| Unique authors | 6,482 | 5,315 |
| Unique tab values | 15 | 15 |
| Unique tags (video_basic) | 111 | — |
| Unique music_id | 7,202 | — |
| Unique user_active_degree | 9 | — |
| Total users in user_features file | 27,285 | |
| Total videos in video_basic file | 7,583 | |

### Evidence classification
HARD FACT.

### Interpretation
The item space (≈7.5K videos, ≈6.5K authors) is small relative to typical industrial catalogs; video_id and author_id vocabularies are entirely learnable within this dataset's scale. `tag` (111 values) is a much coarser categorical than video_id/author_id.

### What it DOES NOT establish
Does not establish which fields are useful for the model — only their size.

### Potential relevance to later agent
Embedding table sizing is cheap at this scale; capacity is not a memory constraint (confirmed further under G01 memory measurements).

### Artifacts
`research/scripts/phase_a_structure.py`, `research/experiment_results/phase_a_structure.json`.

---

## Investigation A02 — Missingness profile

### Question
Which fields have missing values, and how much?

### Why this matters
Missing-value handling errors are a common source of silent bugs in feature engineering; the future agent needs a starting map.

### Data used
All six raw files.

### Method
`isna().mean()` per column.

### Result
Standard logs: 0% missing on all columns. `video_features_basic_pure.csv`: video_duration 3.15%, music_type 2.68%, tag 1.27%, all else 0%. `video_features_statistic_pure.csv`: 0% missing on all columns. `user_features_pure.csv`: onehot_feat4 3.20%, onehot_feat12–17 2.62% each, all else 0%. Full detail in `phase_a_structure.json`.

### Evidence classification
HARD FACT.

### Interpretation
Missingness is low everywhere and concentrated in a handful of `video_basic` and `user_features` side-columns not used by the official baseline.

### What it DOES NOT establish
Does not establish the *reason* fields are missing (e.g. whether missing tag correlates with any target-relevant property) — not tested.

### Potential relevance to later agent
Any feature engineering touching `video_duration`, `music_type`, `tag`, or the affected `onehot_feat*` columns needs an explicit missing-value policy; core log fields do not.

### Artifacts
`research/experiment_results/phase_a_structure.json` (`missingness` key).

---

## Investigation A03 — Train → validation entity overlap

### Question
How much do the user, video, author, and pair-level spaces overlap between train and validation?

### Why this matters
Distinguishes cold-start generalization (new entities) from a "seen-entity, unseen-context" ranking problem, and quantifies how much a memorization-only model could exploit vs. must generalize.

### Data used
Train log, validation log, `video_features_basic_pure.csv` (for author/tag join).

### Method
Set overlap on `user_id`, `video_id`, `author_id`, and pair sets `(user_id,video_id)`, `(user_id,author_id)`, `(user_id,tag)`.

### Result

| Measurement | Value |
|---|---:|
| Valid users seen in train | 98.11% |
| Valid videos seen in train | 99.88% |
| Valid authors seen in train | 99.91% |
| Valid user-video PAIRS seen in train | 1.63% |
| Valid user-author PAIRS seen in train | 3.38% |
| Valid user-tag PAIRS seen in train | 68.14% |
| Cold valid users | 422 (1.89%) |
| Cold valid videos | 7 (0.12%) |
| Cold valid authors | 5 (0.09%) |
| Valid rows with a cold user | 1.59% |
| Valid rows with a cold video | 0.014% |

### Evidence classification
HARD FACT.

### Interpretation
Almost every validation *entity* (user/video/author) was already seen in train — this is overwhelmingly a **warm-entity, unseen-interaction** ranking problem, not a cold-start problem, at the entity level. But at the exact user-video *pair* level, only 1.6% of validation impressions repeat an exact pair seen in train — the FM's `user_id × video_id` interaction term is therefore mostly extrapolating from marginals (user embedding, video embedding, and lower-order interactions with `author_id`/`tab`/`dur_bucket`) rather than memorizing exact pairs. Tag-level pairs repeat much more (68.1%), consistent with `tag` being a much coarser key.

### What it DOES NOT establish
Does not by itself establish which model family best exploits this structure — only the raw overlap numbers.

### Potential relevance to later agent
Any argument of the form "we should worry about cold-start" needs to reckon with only ~1.9% cold users / 0.1% cold videos by count (and even less by *row* share). Any argument that user-video pair memorization alone should carry the model needs to reckon with only 1.6% pair-repeat coverage.

### Artifacts
`research/experiment_results/phase_a_structure.json` (`overlap` key).

---

## Investigation A04 — Author→video redundancy

### Question
Is `author_id` largely redundant with `video_id` (i.e. do most authors have exactly one video)?

### Why this matters
This bears directly on the README's own claim ("`follow_user_num_range`-style coarse buckets are redundant in front of `user_id`") — the analogous question for `author_id` vs. `video_id` on the item side needs its own check, since the official baseline includes both.

### Data used
`video_features_basic_pure.csv`, train log.

### Method
`groupby('author_id')['video_id'].nunique()`.

### Result

| Measurement | Value |
|---|---:|
| Videos per author — median | 1.0 |
| Videos per author — mean | 1.165 |
| Videos per author — p90 | 2.0 |
| Videos per author — max | 26 |
| Authors with exactly 1 video | 5,661 / 6,510 (86.96%) |
| Same computed on train-log impressions only | 87.07% |

### Evidence classification
HARD FACT.

### Interpretation
87% of authors in this dataset have exactly one video, so `author_id` is near-redundant with `video_id` for the large majority of the catalog; its marginal information is concentrated in the ~13% of authors with multiple videos (up to 26).

### What it DOES NOT establish
Does not establish that dropping `author_id` from the FM has no effect — correlated fields can still carry distinct regularization/generalization value even when mostly redundant (per DATA_GUIDE.md's caution against assuming correlated ⇒ useless). That requires the controlled ablation in C01.

### Potential relevance to later agent
Frames the field-ablation experiment (C01, pending) — a near-zero effect from dropping `author_id` would be unsurprising given this redundancy; a large effect would be a genuinely interesting finding given this redundancy.

### Artifacts
`research/experiment_results/phase_a_structure.json` (`author_video_structure` key).

---

## Investigation A05 — Repeat-pair / affinity coverage

### Question
How often do the same (user, video), (user, author), and (user, tag) pairs repeat within a split?

### Why this matters
Establishes whether "user has interacted with this exact video/author/tag before, within the same split" is a meaningfully dense signal at all (a prerequisite for any affinity-style feature).

### Data used
Train log, validation log (+ tag via video_basic join).

### Method
`groupby(key).size()` and repeat-rate statistics.

### Result

| Measurement | Train | Validation |
|---|---:|---:|
| user-video pairs repeated >1x | 4.13% | 2.90% |
| % rows in repeated user-video pairs | 8.19% | 5.67% |
| user-author pairs repeated >1x | 5.91% | 3.25% |
| % rows in repeated user-author pairs | 11.75% | 6.37% |
| user-tag pairs repeated >1x | 51.77% | 24.45% |
| % rows in repeated user-tag pairs | 84.98% | 45.49% |
| Duplicate exact (user,video) pairs within validation | 2.90% of pairs, max repeat 7 |

### Evidence classification
HARD FACT.

### Interpretation
Exact video/author repeat-affinity is a *sparse* signal (single-digit percentages); tag-level repeat-affinity is dense (a majority of train rows belong to a repeated user-tag pair). Missing tags are represented as one explicit categorical value so A03 and A05 use consistent denominators. Any affinity feature keyed on exact video_id or author_id will have limited row coverage; a tag-level (or otherwise coarser) affinity feature would have much broader coverage.

### What it DOES NOT establish
Does not establish that sparse coverage makes video/author-level affinity useless — even a feature active on 5-8% of rows can matter if it is a strong signal on those rows. Does not evaluate any specific feature construction.

### Potential relevance to later agent
Constrains expectations for any "user has seen this video/author before" feature: it will be sparse by construction, unlike a tag-level equivalent.

### Artifacts
`research/experiment_results/phase_a_structure.json` (`repeat_frequency_train`, `repeat_frequency_valid`, `duplicate_uv_pairs_valid` keys).

---

## Investigation A06 — Temporal interaction volume (incl. daily composition, early/late/valid comparison)

### Question
How does daily interaction volume and composition evolve across train, and how does validation compare to early vs. late train?

### Why this matters
Directly informs (without concluding) whether recency weighting, distribution-shift handling, or train/valid distributional mismatch is worth investigating later.

### Data used
Train log (per-day groupby), validation log.

### Method
`groupby('date')` aggregates; early/late train split at the median date; Jaccard overlap of user/video sets; long_view-rate gap vs. validation.

### Result

**Anomaly:** `log_standard_4_08_to_4_21_pure.csv` has rows for only **13** distinct dates (2022-04-09 .. 2022-04-21). **2022-04-08 has zero rows**, despite being the official train start date.

**Daily row counts (train):**

```text
04-09: 52,736   04-13: 94,711   04-17: 44,023   04-21: 20,021
04-10: 227,808  04-14: 71,252   04-18: 24,560
04-11: 278,835  04-15: 58,892   04-19: 20,443
04-12: 166,076  04-16: 60,904   04-20: 20,851
```

Peak day (04-11, 278,835 rows) is ~14x the last day (04-21, 20,021 rows) — a steep overall decline with small day-to-day reversals.

**Period comparison:**

| Period | Dates | Rows | long_view rate | Unique users | Unique videos |
|---|---|---:|---:|---:|---:|
| Early train | 04-09..04-14 | 891,418 | 0.3323 | 25,151 | 7,521 |
| Late train | 04-15..04-21 | 249,694 | 0.3521 | 24,262 | 6,571 |
| Validation | 04-22..04-28 | 124,909 | 0.3133 | 22,377 | 5,951 |

Validation's per-day row count (13,972–26,645/day) is much closer in scale to late train's per-day rate (~20,000–60,900/day) than to early train's (~53,000–279,000/day). Jaccard(video sets): early-late 0.869, early-valid 0.787, late-valid 0.818. Jaccard(user sets): early-late 0.885, early-valid 0.808, late-valid 0.809. long_view-rate gap to validation: early train 0.0190, late train 0.0388 (early is numerically *closer* on this one specific statistic).

### Evidence classification
HARD FACT.

### Interpretation
Two independent facts, reported without a modeling conclusion: (1) daily interaction volume falls sharply overall through train (with small reversals), so validation's *volume/entity-overlap profile* resembles the *tail* more than the peak; (2) validation's raw long_view rate is closer to *early*-train's rate than late-train's. These facts point in different directions and should not be conflated into a "recency helps/hurts" claim.

### What it DOES NOT establish
Does NOT establish that recency weighting, exponential decay, or dropping early-train rows would help or hurt validation/test primary score — no such experiment was run.

### Potential relevance to later agent
If the agent hypothesizes recency weighting, this table is the correct starting evidence (both for and against a naive "recent = more like valid" story) rather than needing to re-derive it.

### Artifacts
`research/scripts/phase_a_structure.py`, `phase_h_temporal.py`; `research/experiment_results/phase_a_structure.json`, `phase_h_temporal.json`; `research/plots/phase_h_temporal.png`.

---

## Investigation B01/B04 — Activity buckets, GAUC weight concentration

### Question
How do baseline GAUC/nDCG@5/primary vary across train-derived user-activity tiers, and how is total GAUC weight distributed across tiers?

### Why this matters
GAUC is a positives-weighted average; if weight concentrates in a specific tier, that tier is where score movement matters most.

### Data used
Validation log + cached seed-0 baseline FM predictions (from `phase_c_baseline.py`'s first seed run); train log for tier definition.

### Method
Tiers defined from each validation user's **train-side** impression count only: Cold = 0 train rows; T1/T2/T3/T4 = quartiles (edges at 17/36/65) of train-impression-count among validation users with ≥1 train row. `evaluate()` computed per tier subset using the official baseline's (seed 0) validation scores, and again using true labels as scores (oracle-per-tier).

### Result

| Tier | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG@5 | Movable nDCG Gap | Fixed-users % | GAUC Weight Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 0.112 | 57.1% | 1.69% |
| T1 (1-17) | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 0.087 | 58.8% | 14.67% |
| T2 (18-36) | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 0.127 | 45.9% | 21.35% |
| T3 (37-65) | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 0.177 | 35.6% | 27.50% |
| T4 (66+) | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 0.262 | 26.3% | 34.79% |

### Evidence classification
HARD FACT.

### Interpretation
Review correction: the original shares divided by every validation positive, including positives from all-positive users that official GAUC excludes. Using the official denominator (34,592 positive rows belonging to mixed-label users), GAUC weight is heavily concentrated in the two highest-activity tiers (T3+T4 = 62.29%). The "fixed" user share falls monotonically with activity (57.1% at Cold → 26.3% at T4). T4 simultaneously has the lowest baseline nDCG@5 (0.511) and the largest within-bucket oracle gap (0.262). GAUC weighting and nDCG user weighting remain separate; they must not be multiplied as if they were one metric weight.

### What it DOES NOT establish
Does not establish *why* T4's nDCG is comparatively low, nor which feature/model change would close the gap — only that the gap exists and where.

### Potential relevance to later agent
Directly usable as prioritization evidence: if the agent is choosing where a hypothesis is likely to move the primary score most, T3/T4 combined carry the majority of GAUC weight and the largest movable nDCG headroom.

### Artifacts
`research/scripts/phase_b_metric.py`; `research/experiment_results/phase_b_metric.json` (`activity_tier_buckets`, `activity_tier_edges`).

---

## Investigation B02 — List-length buckets

### Question
How do baseline and oracle nDCG@5 vary by validation list length (impressions per user)?

### Why this matters
nDCG@5 headroom is mechanically bounded by list length (a list of length 1 has oracle nDCG@5 = baseline nDCG@5 by construction, since there is nothing to reorder).

### Data used
Validation log + cached seed-0 baseline predictions.

### Method
Bucket users by validation impression count into {1, 2–3, 4–5, 6–10, 11–20, 21+}; compute per-bucket `evaluate()` for baseline scores and for oracle (true-label) scores.

### Result

| List Length | Users | Rows | Baseline nDCG@5 | Oracle nDCG@5 | Movable Gap | GAUC Weight Share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.4054 | 0.4054 | 0.000 | 0.00% |
| 2–3 | 6,218 | 15,323 | 0.5413 | 0.6086 | 0.067 | 10.27% |
| 4–5 | 4,119 | 18,326 | 0.6185 | 0.7492 | 0.131 | 16.36% |
| 6–10 | 5,225 | 39,587 | 0.5913 | 0.8536 | 0.262 | 36.39% |
| 11–20 | 2,346 | 32,609 | 0.5037 | 0.9182 | 0.414 | 27.08% |
| 21+ | 552 | 15,147 | 0.3934 | 0.9420 | 0.549 | 9.90% |

Overall list-length distribution: min 1, median 4, mean 5.58, p90 12, p99 26, max 74.

### Evidence classification
HARD FACT.

### Interpretation
As expected mechanically, the within-bucket nDCG gap grows monotonically with list length (0 at length 1, up to 0.549 at 21+). The **6–10 bucket carries the single largest official GAUC weight share (36.39%)** and also the largest contribution to the overall nDCG oracle gap after weighting by its user count. The 11–20 and 21+ buckets have larger per-user gaps but fewer users.

### What it DOES NOT establish
Does not establish that any particular model change (e.g. listwise loss) would close this gap — only where the gap is largest and where it is most heavily weighted.

### Potential relevance to later agent
Useful for evaluating whether a proposed method (e.g. listwise/pairwise loss, per the README's own top suggestion) plausibly targets where the mechanical headroom actually is.

### Artifacts
`research/experiment_results/phase_b_metric.json` (`list_length_buckets`, `list_length_distribution`).

---

## Investigation B03 — Uniform-label / invariant users

### Question
What fraction of validation users/rows are all-negative, all-positive, single-impression, or mixed (movable)?

### Why this matters
All-negative and all-positive users have a fixed nDCG (0 or 1) regardless of model quality; all-negative/all-positive users are also excluded from GAUC entirely. Only mixed-label users are "movable" for GAUC, and single-impression users are movable for neither metric in a meaningful ranking sense.

### Data used
Validation log.

### Method
Per-user positive count vs. list length.

### Result

| Type | Users | % Users | Rows | % Rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.32% | 21,807 | 17.46% |
| All positive | 2,663 | 11.90% | 4,540 | 3.63% |
| Mixed / movable | 12,929 | 57.78% | 98,562 | 78.91% |
| Single impression | 3,917 | 17.50% | 3,917 | 3.14% |

Local validation oracle: GAUC 1.0000, nDCG@5 0.6968, primary 0.8484 — matches `baseline_scores.json`'s published valid-oracle numbers exactly.

### Evidence classification
HARD FACT.

### Interpretation
42.2% of validation users (30.32% + 11.90%) have a fixed nDCG regardless of the model, consistent with the Starter Kit README's analogous test-set figures (27.1% / 9.2%, i.e. broadly similar order of magnitude on the *validation* split though not identical since that's a different split from test). 57.78% of users are genuinely rankable/movable. The exact match of our local oracle numbers to the published `baseline_scores.json` valid-oracle numbers is an internal consistency check that the evaluator and label handling in this audit's pipeline are correct.

### What it DOES NOT establish
Does not establish anything new about test-set composition (that remains untouched per RULES.md); this is a validation-only measurement.

### Potential relevance to later agent
Sets the correct denominator for "how much of the metric is even movable" — roughly 58% of users by count, 79% of rows.

### Artifacts
`research/experiment_results/phase_b_metric.json` (`uniform_label_users`, `oracle_ceiling_local_valid`).

---

## Investigation B05 — Oracle/movable gap by bucket

### Question
Where, combined across activity and list-length views, is the largest score movement mechanically available?

### Why this matters
Synthesizes B01/B02/B04 into a single prioritization view.

### Data used
Same as B01/B02.

### Result
See tables in B01 and B02 above. Lists of length 6–10 carry 36.39% of official GAUC weight and make the largest list-bucket contribution to the overall nDCG oracle gap. By activity, T4 and T3 carry 34.79% and 27.50% of official GAUC weight (62.29% combined); T4 also makes the largest activity-bucket contribution to the overall nDCG gap.

### Evidence classification
HARD FACT (derived directly from B01/B02's HARD FACT measurements).

### Interpretation
Viewed separately, the two partitions of the same validation set (by list length and by train-activity tier) both point toward substantial headroom among longer-list and higher-activity users. B05 alone does not establish how strongly the two dimensions overlap; that relationship is measured directly in B06. These statements use each metric's own weighting semantics.

### What it DOES NOT establish
Does not establish causality or a specific fix — only where headroom is mechanically concentrated under the *current baseline's* ranking behavior. A different model could redistribute this pattern.

### Potential relevance to later agent
A reasonable prioritization prior: longer-list and higher-activity groups each contain substantial headroom. B06 identifies how much of that headroom lies in their actual intersection.

### Artifacts
Same as B01/B02.

---

## Investigation B06 — Joint activity-tier × list-length analysis

### Question
Are users with more train-side activity also the users with longer validation impression lists, and how much metric weight and baseline-to-oracle headroom lies in their actual intersection?

### Why this matters
B01 and B02 analyze activity tier and list length separately. Similar marginal findings do not prove that they describe the same users. A joint analysis is required before claiming that "higher-activity, longer-list users" form one concentrated target group.

### Data used
Train log for each validation user's prior impression count; validation log and cached seed-0 baseline predictions for validation list length and metric decomposition. No evaluation-period rows or labels were accessed.

### Method
For each validation user, pair the train-side impression count used by B01 with the validation-side impression count used by B02. Measure Spearman rank correlation (all users and warm users only), Pearson correlation, and a user-level activity-tier × list-length cross-tab. Then apply the official evaluator to every one of the 5 × 6 disjoint cells and selected intersections. GAUC contribution uses positive-count weights from mixed-label users; nDCG contribution uses equal user weights.

### Result

**Association between raw counts:**

| Measurement | Value |
|---|---:|
| Spearman correlation, all validation users | 0.4620 |
| Spearman correlation, warm validation users only | 0.4677 |
| Pearson correlation, all validation users | 0.4419 |

The relationship is moderately positive: users with more train interactions tend to receive longer validation lists, but the two dimensions are not interchangeable.

**Validation list-length distribution within each activity tier (% of users in that tier):**

| Tier | 1 | 2–3 | 4–5 | 6–10 | 11–20 | 21+ |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 37.20% | 24.64% | 12.09% | 16.82% | 5.92% | 3.32% |
| T1 | 33.43% | 36.85% | 14.88% | 11.50% | 2.78% | 0.56% |
| T2 | 18.32% | 33.61% | 20.97% | 20.51% | 5.49% | 1.09% |
| T3 | 10.81% | 25.54% | 22.20% | 28.64% | 11.11% | 1.69% |
| T4 | 4.92% | 14.75% | 16.26% | 34.03% | 23.46% | 6.59% |

For example, 64.07% of T4 users have lists of 6+ impressions, compared with 14.84% of T1 users. Nevertheless, some short lists remain in every tier, so tier cannot serve as a substitute for list length.

**Key intersections and their contributions to total baseline-to-oracle gaps:**

| Intersection | Users | Rows | GAUC Weight | Total GAUC Gap | Total nDCG Gap | Total Primary Gap |
|---|---:|---:|---:|---:|---:|---:|
| T4 × list 6+ | 3,453 (15.43%) | 42,020 (33.64%) | 30.35% | 29.88% | 34.37% | 31.35% |
| T3/T4 × list 6+ | 5,680 (25.38%) | 64,133 (51.34%) | 50.79% | 50.65% | 53.94% | 51.72% |
| T2/T3/T4 × list 6+ | 7,165 (32.02%) | 78,253 (62.65%) | 64.36% | 63.65% | 65.83% | 64.36% |
| T3/T4 × list 11+ | 2,307 (10.31%) | 38,219 (30.60%) | 27.80% | 27.96% | 28.64% | 28.18% |

The T3/T4 × 6+ intersection contains only 25.38% of validation users but approximately half of the official GAUC weight (50.79%) and half of the current baseline-to-oracle primary gap (51.72%). Expanding to T2+ × 6+ captures 32.02% of users and 64.36% of the primary gap.

### Evidence classification
HARD FACT.

### Interpretation
The prior marginal analyses were directionally consistent because activity and validation list length are moderately associated. The explicit joint decomposition now establishes that much of the current metric headroom really is concentrated in their intersection, rather than merely appearing in two unrelated partitions. The concentration is substantial but not exclusive: about half of the primary gap remains outside T3/T4 × 6+.

### What it DOES NOT establish
Does not establish that activity or list length causes poor rankings, that these quantities are suitable model inputs, or that a targeted loss/model will close the measured gap. Validation list length is an evaluation-group property, and the percentages describe the current seed-0 baseline's headroom; another model could redistribute the gaps.

### Potential relevance to later agent
Provides a verified prioritization target and a diagnostic slice: report performance for T3/T4 users with 6+ validation impressions, while still evaluating the official overall metrics. It supports investigating methods suited to users with both meaningful history and nontrivial within-user ranking lists.

### Artifacts
`research/scripts/phase_b_metric.py`; `research/experiment_results/phase_b_metric.json` (`activity_list_length_association`, `activity_list_length_user_crosstab`, `activity_list_length_joint_cells`, `activity_list_length_key_intersections`).

---

## Investigation C01 — Baseline field ablations

### Question
Does each of the 5 official fields (`user_id`,`video_id`,`author_id`,`tab`,`dur_bucket`) contribute measurably to validation primary score when removed one at a time?

### Why this matters
A04 established that `author_id` is 87% redundant with `video_id`; this ablation tests whether that redundancy translates into measurable interchangeability/uselessness, without assuming it (per DATA_GUIDE.md's explicit caution against that inference).

### Data used
Train + validation logs, `video_features_basic_pure.csv` (for `author_id`). Validation only for scoring.

### Method
Leave-one-out ablation from the 5 official fields, 3 seeds per configuration, official FM (k=16, lr=0.001, batch 8192, patience 4, Adam), early-stopped on validation primary (exactly as the official baseline does). Mean ± population std of validation primary across seeds.

### Result

| Config | Fields | Mean Primary | Std | Δ vs. full |
|---|---|---:|---:|---:|
| full_5field | all 5 | 0.60144 | 0.00027 | — |
| drop_user_id | video_id, author_id, tab, dur_bucket | 0.59325 | 0.00006 | **-0.00819** |
| drop_video_id | user_id, author_id, tab, dur_bucket | 0.60280 | 0.00032 | **+0.00136** |
| drop_author_id | user_id, video_id, tab, dur_bucket | 0.60301 | 0.00026 | **+0.00157** |
| drop_tab | user_id, video_id, author_id, dur_bucket | 0.58554 | 0.00043 | **-0.01590** |
| drop_dur_bucket | user_id, video_id, author_id, tab | 0.60085 | 0.00023 | -0.00059 |

### Evidence classification
`user_id` and `tab` removals: **HARD FACT** (effect sizes of -0.0082 and -0.0159 are 20-40x the combined per-config std — unambiguous). `dur_bucket` removal: **INCONCLUSIVE** (effect -0.0006 is only ~1.7x combined std). `video_id`/`author_id` removals: **WEAK NEGATIVE** against "adding item-identity granularity beyond user_id+tab+dur_bucket helps this exact pointwise FM formulation" (effects of +0.0014/+0.0016 are ~3-4x combined std, individually modest but notable for being small, *positive* (not just "no effect"), and directionally consistent across two different — though correlated — fields).

### Interpretation
`user_id` and `tab` are load-bearing for this baseline; removing either causes a large, unambiguous drop (dropping `tab` alone brings primary close to the item-popularity baseline's 0.5807, i.e. most of the FM's advantage over popularity comes through `tab` combined with `user_id`). `dur_bucket`'s contribution is small and not clearly distinguishable from noise at 3 seeds. The genuinely surprising result is that removing `video_id` or `author_id` — the two most granular, highest-cardinality fields — individually *improved* validation primary by a small but consistent amount in both cases. Checked for bugs: the ablation reuses the same `fm_utils.encode_fields`/`train_fm` path as the reproduced full-baseline run (which matches the published numbers in Section 3), so this is not an encoding artifact; the direction is consistent across two separate fields being dropped independently (not the same experiment repeated), which increases confidence this is a real, if small, effect rather than a one-off fluke. A plausible mechanism, consistent with A03's finding that only 1.6% of validation user-video pairs repeat exactly from train: the `video_id`/`author_id` embeddings' pairwise interaction terms are mostly fit to combinations that don't recur in validation, so they add capacity/noise without adding validation-relevant signal, given `tab` and `dur_bucket` already carry usable item-side information for this small catalog (~7.5K videos).

**Review reproduction (5 matched seeds):** full mean primary 0.60157; drop-video 0.60265 (paired Δ +0.00108, sample SD of paired deltas 0.00065, positive in 5/5); drop-author 0.60289 (paired Δ +0.00132, paired-delta sample SD 0.00048, positive in 5/5). This reproduces the direction and narrows the original three-seed estimates. The mechanism in the preceding paragraph remains a hypothesis, not a demonstrated explanation.

### What it DOES NOT establish
Does **not** establish that `video_id`/`author_id` are useless in general — only that, in this exact pointwise-logloss FM at k=16/lr=0.001 with these fields, removing either alone did not hurt across five matched validation seeds. It does not test removing both, another loss/model, or another dataset regime.

### Potential relevance to later agent
A genuinely new, non-obvious, and specific finding not previously in `constraints.md`: the agent should not assume adding video/author identity to a pointwise FM is automatically beneficial just because it seems like "more information" — this dataset's specific pair-sparsity (from A03) may make coarser fields (`tab`, `dur_bucket`) more reliable than exact item identity for this exact model family. Whether this holds for other model families/losses is an open question for the agent.

### Artifacts
`research/scripts/fm_utils.py`, `research/scripts/phase_c_baseline.py`, `research/experiment_results/phase_c_baseline.json` (`field_ablations`), `research/experiment_results/phase_c_baseline.log`.

---

## Investigation C02 — Seed variance (this environment)

### Question
What is this environment's own FM seed-to-seed variance on validation, for the official 5-field configuration?

### Why this matters
Establishes a local noise floor to judge all other experiments in this audit against, rather than relying solely on the organizer's published test-split std.

### Data used
Train + validation logs, official 5 fields.

### Method
5 seeds (0-4), official config (k=16, lr=0.001), early-stopped on validation primary.

### Result

| Seed | GAUC | nDCG@5 | Primary | Epochs to early-stop |
|---:|---:|---:|---:|---:|
| 0 | 0.6671 | 0.5358 | 0.6015 | 11 |
| 1 | 0.6674 | 0.5361 | 0.6018 | 8 |
| 2 | 0.6671 | 0.5351 | 0.6011 | 11 |
| 3 | 0.6675 | 0.5355 | 0.6015 | 10 |
| 4 | 0.6679 | 0.5361 | 0.6020 | 11 |

Mean primary 0.60157, std 0.00032 (GAUC std 0.00031, nDCG@5 std 0.00038).

### Evidence classification
HARD FACT.

### Interpretation
This environment's local 5-seed population std (0.00032 for primary) is lower than the organizer's published std (0.0008), but five seeds give an imprecise variance estimate. The organizer value is the more conservative noise reference. Effects near or below 0.001 in single-seed or three-seed comparisons should be treated cautiously and, where important, checked with matched-seed deltas.

### What it DOES NOT establish
Does not establish variance for any non-default configuration (different k, lr, or field set) — those are measured separately in C03/C05 and shown to have their own, sometimes larger, seed variance (e.g. lr=0.003's std of 0.00084).

### Potential relevance to later agent
Gives the agent a concrete, environment-specific noise floor for judging its own future experiments' significance, consistent with RULES.md's requirement not to declare a method dead from one noisy run.

### Artifacts
`research/experiment_results/phase_c_baseline.json` (`seed_variance`).

---

## Investigation C03 — Learning-rate sensitivity

### Question
How sensitive is the official FM to learning rate on validation?

### Why this matters
The official lr=0.001 was not itself justified in the Starter Kit beyond "this is the config we used" — worth checking whether it is actually near-optimal.

### Data used
Train + validation logs, official 5 fields, k=16.

### Method
3 seeds each at lr ∈ {0.0003, 0.001, 0.003, 0.01}.

### Result

| lr | Mean Primary | Std |
|---:|---:|---:|
| 0.0003 | 0.60179 | 0.00011 |
| 0.001 (official) | 0.60144 | 0.00027 |
| 0.003 | 0.60009 | 0.00084 |
| 0.01 | 0.59709 | 0.00053 |

### Evidence classification
HARD FACT for the lr≥0.003 degradation (clearly outside noise); INCONCLUSIVE for whether lr=0.0003 is actually better than lr=0.001 (delta 0.00035 is only ~1.3x the larger of the two stds).

### Interpretation
The official lr=0.001 is near the top of the tested range. The 0.01 setting is clearly worse. The 0.003 mean is lower by 0.00135 but has high run variance (population std 0.00084), so the evidence for that setting is weaker than the prior "~4x std" wording implied. A somewhat lower lr (0.0003) is not clearly distinguishable from the default given this seed budget.

### What it DOES NOT establish
Does not test lr below 0.0003, nor any interaction between lr and epoch/patience budget (e.g. whether a lower lr with more epochs allowed would do better) — patience/max-epochs were held at the official values throughout.

### Potential relevance to later agent
Confirms the official lr is a reasonable default and not an obvious quick win to change; a finer sweep near 0.0003-0.001 remains open if the agent wants to pursue it.

### Artifacts
`research/experiment_results/phase_c_baseline.json` (`lr_sensitivity`).

---

## Investigation C04 — Static feature expansion check (organizer dead-end reproduction)

### Question
Does reproducing the organizer's "CWM static feature expansion" comparison on **validation** (rather than relying on the already-published test numbers) show the same null/negative result?

### Why this matters
`constraints.md` C5 and the Starter Kit README report this as a STRONG NEGATIVE, but measured only on the **test** split. RULES.md requires the pre-audit not to treat previously-reported test results as its own evidence, and to independently re-verify organizer claims where cheap to do so.

### Data used
Train + validation logs, `video_features_basic_pure.csv`, `user_features_pure.csv`.

### Method
Same three configurations as the organizer's `ablation_features.py`, but scored on **validation** instead of test: `base_5field` (official 5 fields), `item_8field` (+`music_id`,`video_type`,`upload_type`), `cwm_13field` (+ also `follow_user_num_range`,`register_days_range`,`fans_user_num_range`,`friend_user_num_range`,`user_active_degree`). Review correction: these were previously mislabeled `item_9field`/`cwm_14field`; the saved field lists always contained 8/13 fields, so scores are unaffected. 3 seeds each.

### Result

| Config | Mean Primary | Std | Δ vs. base |
|---|---:|---:|---:|
| base_5field | 0.60144 | 0.00027 | — |
| item_8field | 0.60111 | 0.00046 | -0.00033 |
| cwm_13field | 0.59993 | 0.00052 | -0.00151 |

Organizer's own (test-split) numbers for the analogous comparison: 5-field 0.5950 vs. 13-field 0.5940, Δ≈-0.0010 (`README.md`).

### Evidence classification
STRONG NEGATIVE specifically for this exact static-feature formulation, based on the controlled three-seed validation comparison and limited to that formulation. The organizer's published result is context, not review evidence.

### Interpretation
The organizer's claim reproduces on validation: adding these 8 static fields (3 item-side + 5 user-side coarse buckets) to the pointwise FM does not help and mildly hurts. The retained evidence classification applies narrowly to this exact validation-tested formulation; published test numbers are reference context, not review evidence.

### What it DOES NOT establish
Does not establish that *no* additional feature could help — only that this exact 8-added-field expansion (13 total), in this exact FM formulation, does not. Does not test every subset or alternative encoding.

### Potential relevance to later agent
Strengthens (does not merely repeat) the existing `constraints.md` C5 entry: the agent can now be told this was independently re-verified on validation in this environment, not only inherited from the organizer's test-split claim.

### Artifacts
`research/experiment_results/phase_c_baseline.json` (`static_feature_expansion`).

---

## Investigation C05 — FM embedding-dimension check (organizer dead-end reproduction)

### Question
Does reproducing the organizer's embedding-dimension sweep on validation show the same "capacity doesn't matter" null result?

### Why this matters
Same rationale as C04 — `constraints.md` C6 is organizer test-split evidence; RULES.md asks for independent re-verification where cheap.

### Data used
Train + validation logs, official 5 fields.

### Method
3 seeds each at k ∈ {8, 16, 32, 64} (organizer tested 8/16/32; 64 added here to extend the range).

### Result

| k | Mean Primary | Std |
|---:|---:|---:|
| 8 | 0.60111 | 0.00080 |
| 16 (official) | 0.60144 | 0.00027 |
| 32 | 0.60146 | 0.00069 |
| 64 | 0.60098 | 0.00044 |

Organizer's (test-split) numbers: k=8/16/32 → 0.5895/0.5902/0.5887 (`README.md`) — also flat.

### Evidence classification
STRONG NEGATIVE for simple FM capacity scaling, independently corroborated on validation (all four k values here are mutually within ~1 combined std of each other) and extended one octave further (k=64) than the organizer tested.

### Interpretation
Confirms `constraints.md` C6: at this dataset's scale (~1.14M train rows, ~7.5K videos), simply widening the FM's embedding dimension does not move validation primary, from k=8 through k=64.

### What it DOES NOT establish
Does not rule out a different model family with more expressive interaction structure (DeepFM/DCN/xDeepFM) — only that *this* FM's raw embedding width is not the bottleneck, consistent with `constraints.md` C6's own stated boundary.

### Potential relevance to later agent
Strengthens the existing C6 entry with independent validation-split corroboration; a hypothesis of the form "just make the baseline FM bigger" is now doubly (test + validation) unsupported.

### Artifacts
`research/experiment_results/phase_c_baseline.json` (`embedding_dim`).

---

## Investigation D01 — Feedback-label prevalence

### Question
What is the prevalence/distribution of each post-impression feedback signal?

### Why this matters
A signal with near-zero prevalence cannot support a dense auxiliary task or historical feature.

### Data used
Train + validation logs, all `FEEDBACK_COLS`.

### Method
Mean (binary) or distribution summary (continuous) per split.

### Result (validation)

| Signal | Prevalence / Summary |
|---|---|
| is_click | 44.38% positive |
| is_like | 1.80% |
| is_follow | 0.130% |
| is_comment | 0.233% |
| is_forward | 0.078% |
| is_hate | 0.062% |
| is_profile_enter | 1.95% |
| play_time_ms | mean 21,487ms, median 4,607ms, p90 62,826ms, p99 206,270ms, 11.7% exactly zero |
| profile_stay_time | mean 1.88, 99.99% exactly zero |
| comment_stay_time | mean 460.3, 95.54% exactly zero |

Train-side numbers are close (e.g. is_click 44.4%→ train not shown here but see `phase_d_feedback.json`).

### Evidence classification
HARD FACT.

### Interpretation
`is_click` and `play_time_ms` are dense; `is_like`/`is_profile_enter` are sparse-but-present (~2%); `is_follow`/`is_comment`/`is_forward`/`is_hate` are all under 0.25% positive — extremely sparse. `profile_stay_time` is almost entirely zero (99.99%) and unlikely to carry row-level signal in its raw form.

### What it DOES NOT establish
Prevalence alone does not establish whether a sparse signal is still useful as an auxiliary task (rare events can still transfer useful gradient signal) — only that a naive dense-supervision auxiliary head would starve on `is_follow`/`is_comment`/`is_forward`/`is_hate`.

### Potential relevance to later agent
Any multi-task proposal should account for these prevalence numbers when choosing loss weights or expecting a given auxiliary task to have enough positive examples per minibatch.

### Artifacts
`research/experiment_results/phase_d_feedback.json` (`valid.prevalence`, `train.prevalence`).

---

## Investigation D02 — Feedback ↔ long_view association (same-row, diagnostic only)

### Question
How strongly does each feedback signal correlate with `long_view` **at the same row**?

### Why this matters
RULES.md forbids using same-row feedback as a `long_view` input; this measurement exists purely to characterize the *label-generation relationship*, and to make sure nobody later mistakes "this looks predictive" for "this is a safe feature" — a very strong same-row correlation is precisely the leakage risk RULES.md is warning about.

### Data used
Validation log (`phase_d_feedback.json` also has train).

### Method
Pearson correlation with `long_view`; group means; for continuous signals, quintile-conditional long_view rate among nonzero values.

### Result

| Signal | Pearson r (same-row) | Notes |
|---|---:|---|
| is_click | 0.751 | long_view rate given click=0 is 0.0019 vs. given click=1 is 0.703 |
| play_time_ms | 0.632 | long_view rate is exactly 0 for the bottom two nonzero quintiles, then rises to 1.0 in the top quintile |
| is_profile_enter | 0.127 | |
| comment_stay_time | 0.169 | |
| is_like | 0.095 | |
| is_comment | 0.059 | |
| is_follow | 0.025 | |
| is_forward | 0.025 | |
| is_hate | -0.004 | |
| profile_stay_time | -0.0005 | |

### Evidence classification
HARD FACT.

### Interpretation
`is_click` and `play_time_ms` are almost mechanically tied to `long_view` at the same row (consistent with `long_view` being a thresholded function of watch time, itself gated by whether a click/play happened at all). This is the clearest, most quantified evidence in this audit of *why* RULES.md's same-row leakage rule exists for this specific dataset: a naive model using same-row `is_click` or `play_time_ms` would trivially and unrealistically inflate apparent performance. The remaining signals (`is_like`, `is_comment`, `is_follow`, `is_forward`, `is_profile_enter`, `comment_stay_time`) have weak-to-moderate same-row correlation (0.02–0.17); `is_hate` and `profile_stay_time` are essentially uncorrelated.

### What it DOES NOT establish
Does not evaluate any of these as **historical** (lagged) features or **auxiliary targets** — both remain legitimate per RULES.md and are untested here.

### Potential relevance to later agent
Strong, dataset-specific confirmation that `is_click`/`play_time_ms` must never appear as same-row inputs; also flags that weaker signals (`is_like`, `comment_stay_time`, `is_profile_enter`) are plausible historical-feature or auxiliary-task candidates precisely because their same-row correlation is only moderate (informative but not degenerate).

### Artifacts
`research/experiment_results/phase_d_feedback.json` (`valid.long_view_association`, `train.long_view_association`).

---

## Investigation D03 — Inter-feedback association

### Question
How correlated are the feedback signals with each other?

### Why this matters
Highly collinear auxiliary targets in a multi-task setup would carry redundant gradient signal; this is a first check.

### Data used
Validation + train logs.

### Method
Pearson correlation matrix over all 10 feedback columns.

### Result
Full matrix in `phase_d_feedback.json` (`valid.inter_feedback_correlation`, `train.inter_feedback_correlation`). Notable pairs (validation): `is_click`–`play_time_ms` r=0.5167 (both being watch-related), and `is_comment`–`comment_stay_time` r=0.3029. `is_like`/`is_follow`/`is_forward` are mutually weakly positively correlated and individually sparse per D01.

### Evidence classification
HARD FACT.

### Interpretation
The feedback signals are not one redundant cluster — `is_click`/`play_time_ms` form one tight cluster (watch-related), while `is_like`/`is_follow`/`is_comment`/`is_forward` form a separate, much sparser, weakly-correlated cluster (active-engagement-related). This suggests two qualitatively different families of auxiliary signal exist in this dataset, not one.

### What it DOES NOT establish
Does not establish which, if any, would help as an auxiliary task for `long_view` specifically (which requires held-out multi-task experiments, not correlation).

### Potential relevance to later agent
Informs auxiliary-task selection/grouping if the agent pursues multi-task learning (e.g. shared vs. separate towers for the two clusters) — a structural fact to reason from, not a recommendation.

### Artifacts
`research/experiment_results/phase_d_feedback.json` (`*.inter_feedback_correlation`).

---

## Investigation D04/D05 — Historical feedback availability, overall and by activity tier

### Question
How much prior (train-period) interaction and feedback history exists for validation users, overall and by activity tier?

### Why this matters
This is the single largest open question flagged by the Starter Kit README ("behavior sequences are completely untouched") — before any DIN/SIM/sequential-modeling hypothesis is worth pursuing, the agent needs to know whether sufficient history actually exists.

### Data used
Train log (history source) + validation log (target rows). Because official train dates (04-08..21) strictly precede validation dates (04-22..28), **every** train impression of a user is, by construction, strictly prior to **every** validation impression of that same user — no per-row timestamp comparison across the split boundary is needed for this specific measurement.

### Method
Per validation user: count of train rows (= "prior interactions"), and train-side sums of each feedback signal. Proportions with ≥1/≥5/≥10. Repeated for repeat-video/repeat-author/repeat-tag row-level coverage. Tiers as defined in B01.

### Result

**Overall:**

| Measurement | Value |
|---|---:|
| % valid users with ≥1 prior train interaction | 98.11% |
| % valid users with ≥5 prior train interactions | 92.85% |
| % valid users with ≥10 prior train interactions | 85.17% |
| Median prior interactions | 35 |
| Mean prior interactions | 47.4 |
| p90 prior interactions | 103 |

**By tier:**

| Tier | Median Prior Rows | ≥1 | ≥5 | ≥10 |
|---|---:|---:|---:|---:|
| Cold | 0 | 0% | 0% | 0% |
| T1 | 9 | 100% | 79.4% | 49.3% |
| T2 | 26 | 100% | 100% | 100% |
| T3 | 49 | 100% | 100% | 100% |
| T4 | 95 | 100% | 100% | 100% |

**Per-feedback-signal prior availability** (e.g. ≥1 prior like, ≥1 prior click) — full table in `phase_f_history.json`'s `per_feedback_signal_prior`.

**Row-level repeat coverage** (validation row's exact video/author/tag was seen in that user's train history):

| Coverage | Overall | Cold | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|
| Same video seen before | 1.62% | 0% | 0.73% | 1.09% | 1.37% | 2.47% |
| Same author seen before | 3.38% | 0% | 1.57% | 2.44% | 2.90% | 5.01% |
| Same tag seen before | 73.19% | 0% | 45.80% | 67.97% | 77.59% | 86.47% |

Bonus (not explicitly requested, additional context): 81.57% of validation rows have at least one row from the same user at a strictly earlier `(date, hourmin, time_ms)` timestamp. Review correction: the previous 82.09% used `cumcount()` and incorrectly treated tied timestamps as ordered; 5.60% of rows belong to non-unique user/timestamp groups. This is only an availability diagnostic. Whether within-validation outcomes can form deployable features depends on an explicit online protocol and was not established here.

### Evidence classification
HARD FACT.

### Interpretation
The large majority of validation traffic (85% of users by the ≥10 threshold, and 98.4% of rows once weighted by the fact that T2+ users — who are 100% covered at ≥10 — generate more rows per user) has substantial train-side interaction history available. Exact video/author-level repeat history is sparse (1.6%/3.4%), consistent with A05's finding; tag-level repeat history is dense (73.2% overall, rising to 86.5% in the most active tier). Aggregate feedback-count history (not just impression counts) is also available per D01's prevalence numbers combined with these interaction counts.

### What it DOES NOT establish
Does **not** establish that any specific historical-feature construction or sequence model (DIN, SIM, etc.) would improve validation primary score — only that the raw data support for such approaches is present at these coverage levels. Does not establish the best temporal aggregation window or feature form.

### Potential relevance to later agent
This is likely the most decision-relevant table in the whole audit for a "user history" hypothesis: it confirms the README's claim that history exists ("hundreds to thousands of interactions per user" — our measured median is 35, mean 47, p90 103, i.e. the README's "hundreds to thousands" appears to overstate the *typical* case, though some users do reach into the hundreds) while also showing that exact-entity repeat is sparse and tag-level repeat is the dense signal.

### Artifacts
`research/scripts/phase_f_history.py`, `research/experiment_results/phase_f_history.json`.

---

## Investigation E01 — Video basic/statistic feature inventory (incl. aggregation-window inference)

### Question
What do the video basic and statistic feature files contain, and is the statistic file's aggregation window/population documented or inferable?

### Why this matters
DATA_GUIDE.md explicitly requires establishing causal validity of any video-statistic feature before trusting it; using future or out-of-population aggregates would be leakage even though the file ships as part of the official dataset.

### Data used
`video_features_basic_pure.csv`, `video_features_statistic_pure.csv`, train+valid logs (for the reconstructed-vs-observed comparison).

### Method
Per-field missingness/cardinality/range for the basic file; per-field summary + correlation matrix for the 16 core statistic fields; integer-recoverability check for `show_cnt × counts`; comparison of `show_cnt × counts` (reconstructed total) against actual observed impression counts for the same video in train+valid standard logs.

### Result

**Video basic inventory (selected):** `video_type` 3 values (98.98% NORMAL); `upload_type` 14 values (top: LongImport 38.6%, Web 31.9%); `visible_status` **constant** (1 unique value, 100% "0.0" — carries zero information); `music_type` 6 values (87.9% one value); `tag` 111 values, 1.27% missing; `video_duration` median 81,171ms, p90 237,830ms, 3.15% missing.

**`tab` and `dur_bucket` marginal long_view-rate spread (train):** `tab` ranges from 0.42% (tab 3, n=3,574) to 61.25% (tab 10, n=80) with the two dominant tabs at 4.22% (tab 0, n=150,013) and 38.61% (tab 1, n=834,876) — a very large spread across a highly imbalanced categorical. `dur_bucket` (10 quantile buckets) ranges more mildly, 0.273 to 0.376.

**Aggregation-window inference:** `show_cnt × counts` is an integer for **100%** of videos (⇒ `show_cnt` etc. are per-`counts`-unit averages). `counts` itself ranges 45–181 (median 147) — already longer than the dataset's 31-day span, suggesting `counts` is not simply "days within this dataset's window." Comparing the reconstructed total (`show_cnt × counts`) to the number of times that video actually appears in our train+valid standard logs: **median ratio 11,465×**, p10 5,248×, p90 38,199×, and **0% of videos** have a reconstructed total smaller than observed (i.e. the reconstructed total is always at least as large, usually many thousand times larger).

### Evidence classification
HARD FACT for the inventory and the integer-recoverability/ratio numbers; **INCONCLUSIVE** for what the aggregation window/population actually is.

### Interpretation
`visible_status` is constant and carries zero information. `tab` shows large marginal spread in long_view rate. The video-statistics scale is far larger than sampled train+validation traffic and is consistent with a larger population and/or external time window. The files do not identify that population/window or prove whether it overlaps evaluation dates.

### What it DOES NOT establish
Does **not** establish whether the statistics window overlaps the evaluation period in a way that would constitute temporal leakage relative to *this competition's* test labels — that cannot be determined from the files available. It also does not establish that using these statistics is disallowed; DATA_GUIDE.md asks only that this uncertainty be labeled explicitly, which this entry does.

### Potential relevance to later agent
Any hypothesis using `video_features_statistic_pure.csv` should carry this uncertainty forward explicitly rather than assuming the file is safe merely because it ships with the official dataset. `visible_status` can be dropped outright (zero information, not merely low-value).

### Artifacts
`research/scripts/phase_g_video_features.py`, `research/experiment_results/phase_g_video_features.json`.

---

## Investigation E02 — Ratio-feature associations (safety unresolved)

### Question
Do simple smoothed engagement ratios derived from the video-statistics file show any association with `long_view` on validation?

### Why this matters
Establishes whether this data source carries information at all (setting aside the E01 causal-validity caveat), as a prerequisite for any later feature-engineering hypothesis.

### Data used
`video_features_statistic_pure.csv` joined onto the validation log by `video_id`.

### Method
Four smoothed ratios with α=1, β=20: `(long_time_play_cnt+α)/(show_cnt+β)`, `(play_cnt+α)/(show_cnt+β)`, `(complete_play_cnt+α)/(show_cnt+β)`, `(like_cnt+α)/(show_cnt+β)`. Pearson correlation with `long_view` and quintile-conditional long_view rate, validation only.

### Result

| Ratio | Pearson r | Bottom Quintile Rate | Top Quintile Rate |
|---|---:|---:|---:|
| long_time_play_cnt ratio | 0.302 | 0.105 | 0.505 |
| play_cnt ratio | 0.185 | 0.180 | 0.396 |
| complete_play_cnt ratio | 0.181 | 0.190 | 0.436 |
| like_cnt ratio | 0.040 | 0.249 | 0.353 |

### Evidence classification
HARD FACT (association exists) combined with the E01 causal-validity caveat (INCONCLUSIVE whether it's safe to use).

### Interpretation
The long-time-play, play, and completion ratios show monotonic quintile trends; the like ratio rises through Q4 then dips slightly in Q5 (0.357 to 0.353). The long-time-play ratio is strongest (r=0.30, a >4.8x bottom-to-top rate spread). This establishes marginal association only, not incremental information beyond `video_id`, causality, or feature safety.

### What it DOES NOT establish
Does not establish incremental value **over the existing baseline**, which already includes raw `video_id` (and therefore can already, in principle, memorize per-video effects directly via its embedding) — a marginal-correlation finding is not the same as a controlled ablation showing improvement when added to the FM. Also inherits E01's aggregation-window uncertainty.

### Potential relevance to later agent
A plausible but not yet validated feature-engineering direction; the correct next step (not taken here, per pre-audit scope limits) would be a controlled FM ablation adding one ratio at a time on validation, alongside continued attention to the E01 causal-validity question.

### Artifacts
`research/experiment_results/phase_g_video_features.json` (`ratio_feature_association_valid`).

---

## Investigation F01 — Random-exposure log audit

### Question
What does `log_random_4_22_to_5_08_pure.csv` actually contain, and what would be safe vs. unsafe uses of it?

### Why this matters
DATA_GUIDE.md flags this file as a potential source of counterfactual/off-policy signal but requires establishing its date coverage and overlap risk before any use.

### Data used
`log_random_4_22_to_5_08_pure.csv`, standard train+valid logs.

### Method
Date-only row counts for the full file; all other checks use only the validation-period slice. The review-corrected loader never materializes evaluation-period outcome or feature columns.

### Result

| Measurement | Value |
|---|---:|
| Total rows | 1,186,059 (date-only count) |
| Date coverage | 2022-04-22 .. 2022-05-08 (17 dates) — spans BOTH validation and evaluation date ranges |
| Rows in validation-date range (04-22..28) | 288,338 (24.3%) |
| Rows in evaluation-date range (04-29..05-08) | 897,721 (date-only count; no outcomes/features accessed) |
| `is_rand`==1 | 100% of rows (vs. 0% in both standard logs) |
| Validation-period random-log users also in train+standard-valid | 98.89% |
| Random-log videos also in standard logs | 99.50% |
| Train+standard-valid users covered by validation-period random log | 70.89% |
| Train+standard-valid videos covered by validation-period random log | 99.51% |
| Shared (user,video) pairs: random(valid-period) ∩ standard-valid | 17 of 288,328 (0.006%) |
| long_view rate, random log, validation-period rows | 8.06% |
| long_view rate, standard validation log (comparison) | 31.3% |

### Evidence classification
HARD FACT.

### Interpretation
The validation-period slice is a separate, fully-random exposure stream with broad entity overlap and almost entirely disjoint (user,video) pairs from standard validation (0.006% overlap). Its long_view rate (~8%) is roughly 4x lower than the standard log's (~31%), consistent with strong exposure/selection differences. This comparison alone does not identify the causal source or magnitude of bias.

### What it DOES NOT establish
Does not establish any specific safe use (e.g. as an unbiased evaluation set, an IPS-correction source, or an auxiliary training signal). Only the validation-period subset is eligible for development; evaluation-period rows must remain inaccessible except for date-only counts. This audit performed no model experiment with the random log.

### Potential relevance to later agent
Establishes that a "diagnostic unbiased validation set" hypothesis (the Starter Kit README's own suggestion #7) is *feasible* in principle using the validation-period slice (288,338 rows, 19,091 users, 7,546 videos) without touching evaluation-period rows; the agent still needs to decide whether and how to use it.

### Artifacts
`research/scripts/phase_i_random_log.py`, `research/experiment_results/phase_i_random_log.json`.

---

## Investigation G01 — CSV load + encoding + training runtime

### Question
How long does each pipeline stage take in this environment?

### Why this matters
Bounds how many iterations the future agent can realistically run within the 6-hour / 50-iteration budget.

### Data used
All raw files; the 5-field baseline encoding and one full FM training run (official config, seed 0).

### Method
Wall-clock timing around each stage using `research/scripts/phase_j_engineering.py`.

### Result

| Stage | Time (s) |
|---|---:|
| CSV load, train log | 1.09 |
| CSV load, valid log (strict date-filtered materialization) | 1.72 |
| CSV load, video_basic | 0.015 |
| CSV load, video_stat | 0.067 |
| CSV load, user_features | 0.096 |
| **Total load** | **2.99** |
| Encoding (5-field baseline) | 4.81 |
| FM training (seed 0, 11 epochs to early-stop) | 49.7 (≈4.43s/epoch) |
| Predict-only (validation) | 0.079 |
| **Cold run total (load+encode+train)** | **≈57.5** |

Peak RSS observed for this full pipeline: ~490MB.

### Evidence classification
HARD FACT.

### Interpretation
A single full baseline training run (cold, from raw CSVs) takes about a minute in this environment, matching the Starter Kit README's "~40s" claim in the right order of magnitude (our number is somewhat higher, likely due to this being a different/slower CPU or additional epochs to early-stop than the README's reference run). At this rate, 50 iterations of a comparably-sized model would cost roughly 50 minutes of pure training time — well inside the 6-hour budget, leaving ample room for larger models or repeated-seed evaluation per iteration if desired.

### What it DOES NOT establish
Does not establish the runtime of any more complex model (listwise loss, multi-task, sequence model) the agent might later try — only the baseline FM's cost.

### Potential relevance to later agent
Gives a concrete per-iteration time budget baseline to reason against when deciding whether a more expensive model/experiment fits the 6-hour ceiling.

### Artifacts
`research/scripts/phase_j_engineering.py`, `research/experiment_results/phase_j_engineering.json`.

---

## Investigation G02 — Cache speedup

### Question
Does caching the encoded train/valid arrays produce a meaningful, correctness-preserving speedup?

### Why this matters
If the future agent will run many iterations against the same fields, avoiding re-encoding from raw CSV each time is a straightforward efficiency win, but only if it is verified not to silently corrupt or leak data.

### Data used
The same encoded 5-field train+valid arrays as G01.

### Method
Pickle the encoded arrays to disk, reload, and compare byte-for-byte equality against the in-memory originals; time both directions.

### Result

| Measurement | Value |
|---|---:|
| Cache write | 0.029s |
| Cache read | 0.018s |
| Speedup vs. re-encoding (4.81s) | ≈263x |
| Arrays bit-identical after reload | True |

### Evidence classification
HARD FACT.

### Interpretation
A simple pickle-based cache of encoded arrays is both correct (bit-identical reproduction) and fast (263x faster than re-encoding on this rerun) in this environment, for a cache built from train+valid only.

### What it DOES NOT establish
Does not establish a cache invalidation policy for when the feature set or field list changes — the future agent's harness must key the cache on the exact field/config fingerprint, not just reuse blindly. Not tested here.

### Potential relevance to later agent
Directly actionable: caching encoded arrays (keyed by field-set fingerprint) is safe and highly effective in this environment; re-encoding on every iteration is unnecessary overhead if the field set repeats.

### Artifacts
`research/experiment_results/phase_j_engineering.json` (`cache_test`).

---

## Investigation G03 — Windows subprocess timeout / process-tree recovery

### Question
Does a naive subprocess timeout reliably bound wall-clock time on this Windows environment, including when the tracked process spawns children of its own?

### Why this matters
The final autonomous harness will very likely run experiment code in a subprocess with a timeout to enforce the 6-hour / per-iteration budget; a timeout mechanism that silently fails to bound time is a serious, easy-to-miss robustness gap.

### Data used
None (synthetic subprocess test only — no dataset files involved).

### Method
Launched `subprocess.run([...], timeout=3, capture_output=True)` against a child script that itself spawns an unmanaged grandchild (`Popen` with default/inherited stdio) sleeping for 30 seconds; measured actual elapsed wall time and checked for orphaned processes afterward.

### Result
`subprocess.run(timeout=3)` did **not** raise/return within 3 seconds — it blocked for **30.13 seconds**, matching the grandchild's full sleep duration almost exactly. No orphaned process remained after that point (the grandchild had exited naturally by then). A separate, isolated test of a syntax-error child script returned cleanly and immediately (returncode 1, `"SyntaxError"` present in stderr).

### Evidence classification
HARD FACT.

### Interpretation
The exact 30.13s match is consistent with inherited output-pipe handles keeping `communicate()` open until the grandchild exits. The experiment directly verifies the overrun in this Windows condition; it does not independently instrument handle inheritance, so that mechanism is the best-supported explanation rather than a separately proven fact.

### What it DOES NOT establish
Does not test the fix (e.g. `CREATE_NEW_PROCESS_GROUP` + `taskkill /T /F`, or `psutil`-based recursive child enumeration/kill) — only that the naive approach fails under this specific but realistic condition (a child process that itself forks further work, which is exactly what a research-agent-launched training script might do, e.g. via a data-loader worker or a nested tool call).

### What it DOES NOT establish
This finding is specific to Windows; it may not reproduce on POSIX systems, where process groups and signal semantics differ.

### Potential relevance to later agent
If the final harness runs on Windows (as this environment does) and needs a hard wall-clock bound per iteration, it must not rely on bare `subprocess.run(timeout=N)` whenever the launched process might itself spawn children — this is a concrete, previously-unverified engineering risk for the 6-hour/50-iteration budget enforcement, now confirmed rather than assumed.

### Artifacts
`research/scripts/phase_j_engineering.py` (`windows_subprocess_timeout` section), `research/experiment_results/phase_j_engineering.json`.

---

## Investigation G04 — NaN/Inf and syntax-error recovery

### Question
Are NaN/Inf submission scores and syntax errors in a launched script cleanly detectable?

### Why this matters
Both are explicitly listed in RULES.md as failure modes the agent must recover from.

### Data used
A synthetic submission CSV shaped like the validation split (user_id/video_id pairs from the real validation log, with one row's score deliberately set to `nan`); a synthetic Python file with a deliberate syntax error.

### Method
Ran the **unmodified** official `submit.py::read_submission` against the synthetic NaN submission; ran the syntax-error script as a subprocess and inspected returncode/stderr.

### Result
`read_submission` raised `ValueError` on the NaN row (existing official behavior, not modified for this test): *"第 2 行 score 是 NaN/Inf，不允许"* ("row 2's score is NaN/Inf, not allowed"). The syntax-error script exited with returncode 1 and `"SyntaxError"` present in stderr.

### Evidence classification
HARD FACT.

### Interpretation
Both failure modes are already cleanly and specifically detectable using existing official code (`submit.py`) or standard Python subprocess semantics (returncode + stderr pattern) — no new guard code is required for these two cases.

### What it DOES NOT establish
Does not test NaN/Inf appearing *mid-training* (e.g. an exploding-gradient scenario) rather than only at final submission-file validation — a different, untested failure surface.

### Potential relevance to later agent
The agent's harness can rely on `submit.py`'s existing validation rather than reimplementing NaN/Inf checking, and can rely on standard subprocess returncode/stderr inspection for syntax errors (distinct from the G03 timeout caveat, which is about wall-clock bounding, not error detection).

### Artifacts
`research/scripts/phase_j_engineering.py` (`syntax_error_recovery`, `nan_inf_recovery`), `research/experiment_results/phase_j_engineering.json`.

---

# 5. Evidence Summary

## Hard Facts

1. **Baseline reproduces.** Local validation FM (seed 0): GAUC 0.6671, nDCG@5 0.5358, primary 0.6015 — within published seed std (0.0008) of GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016. Row counts match exactly (train 1,141,112 / valid 124,909 / test 170,588).
2. **Entity overlap is high; pair overlap is low.** 98.1% of valid users, 99.9% of valid videos, and 99.9% of valid authors were seen in train, but only 1.6% of valid user-video pairs and 3.4% of valid user-author pairs repeat exactly from train. User-tag pairs repeat much more (68.1%).
3. **Author redundancy is high but not total.** 87.0% of authors have exactly 1 video; the remaining 13% account for up to 26 videos per author.
4. **Repeat-affinity is sparse at video/author granularity, dense at tag granularity.** Train: 4.1%/5.9%/52.1% of user-video/user-author/user-tag pairs repeat >1x respectively.
5. **`visible_status` is constant** (100% one value) and carries zero information.
6. **`tab` shows a very large long_view-rate spread** across a highly imbalanced categorical (0.42% to 61.25% across categories with n≥80); `dur_bucket` shows a milder spread (0.273–0.376).
7. **42.2% of validation users have a fixed nDCG** (30.32% all-negative + 11.90% all-positive); local oracle (GAUC 1.0, nDCG@5 0.6968, primary 0.8484) matches the published valid-oracle exactly.
8. **Activity and validation list length are moderately associated, and their intersection concentrates headroom.** Train-side activity count and validation list length have Spearman ρ=0.462. The T3/T4 × 6+ intersection contains 25.38% of users but 50.79% of official GAUC weight and 51.72% of the current baseline-to-oracle primary gap. Viewed marginally, T3+T4 carry 62.29% of GAUC weight and the 6–10 list bucket carries 36.39%.
9. **`is_click` (r=0.751) and `play_time_ms` (r=0.632) are extremely strongly correlated with `long_view` at the same row** — the clearest, quantified reason same-row use of these fields is forbidden by RULES.md for this dataset. Other feedback signals are weak-to-moderately correlated (0.02–0.17) or negligible (`is_hate`, `profile_stay_time`).
10. **Substantial train-side interaction history exists for validation users**: 85.2% have ≥10 prior train interactions (median 35, mean 47.4, p90 103); Cold users (1.9% of valid users) have none by construction. Exact video/author repeat-history coverage is sparse (1.6%/3.4% of rows); tag-level repeat-history coverage is dense (73.2% of rows, up to 86.5% in the most active tier).
11. **Video-stat products have an unexplained external scale.** `show_cnt × counts` is near-integer for 100% of videos and the reconstructed/observed train+valid ratio has median 11,465×. This supports, but does not prove, a larger population/window; causal safety remains unresolved.
12. **The random-exposure log (1,186,059 rows) is a genuinely separate stream**: 100% `is_rand`=1, only 0.006% (user,video)-pair overlap with the standard validation log, ~4x lower long_view rate (8% vs 31%) — direct internal evidence of exposure bias in the standard logs. 75.7% of its rows fall in the evaluation-date range.
13. **A cache of encoded train+valid arrays is correct and fast**: bit-identical reload, ~263x speedup on the review rerun (0.018s vs 4.81s).
14. **On this Windows environment, `subprocess.run(timeout=N)` does not reliably bound wall-clock time** when the tracked process spawns an unmanaged child that inherits stdio pipe handles — verified: a 3-second timeout took 30.13 seconds to actually return, matching the un-managed grandchild's full runtime.
15. **NaN/Inf submissions and child-process syntax errors are both already cleanly detectable** using existing official code (`submit.py`) and standard subprocess returncode/stderr inspection, respectively.
16. **Training-log date coverage has an undocumented gap and steep decay.** `log_standard_4_08_to_4_21_pure.csv` has zero rows for 2022-04-08 (13 distinct dates, not 14); daily volume peaks on 04-11 (278,835 rows) and decays to 20,021 by 04-21 — a ~14x drop. Validation's per-day volume resembles the tail of train far more than the middle.
17. **Baseline reproduction and 5-seed variance (this environment).** Mean validation primary 0.60157, std 0.00032 across 5 seeds (own reproduction) — same order of magnitude as, and tighter than, the organizer's published 0.0008.
18. **`user_id` and `tab` are load-bearing fields; `dur_bucket`'s individual contribution is not clearly distinguishable from 3-seed noise.** Dropping `user_id` costs -0.0082 primary; dropping `tab` costs -0.0159 (both >20x combined std); dropping `dur_bucket` costs only -0.0006 (~1.7x combined std, inconclusive).
19. **Learning-rate sensitivity is asymmetric**: lr=0.01 clearly degrades primary; lr=0.003 is lower but noisy, and lr=0.0003 versus 0.001 is inconclusive.
20. **FM embedding-dimension scaling (k=8/16/32/64) is flat on validation**, independently reproducing and extending the organizer's test-split finding (`constraints.md` C6).
21. **The organizer's static-feature expansion finding reproduces on validation**: adding 3 item fields (`item_8field`) or 8 total static fields (`cwm_13field`) does not help and mildly hurts (-0.0003 and -0.0015 respectively vs. the 5-field base), consistent with existing constraint C5.

## Strong Negative Evidence

1. **FM embedding-dimension scaling (k=8/16/32/64) does not improve validation primary** (Investigation C05) — all four values mutually within ~1 combined std (0.60098–0.60146). Independently reproduces and extends `constraints.md` C6 (previously test-split-only evidence) to validation and to k=64.
2. **The organizer-tested static feature expansion (`item_8field` / `cwm_13field`) does not improve validation primary** (Investigation C04): base 0.60144 → item_8field 0.60111 (Δ-0.00033) → cwm_13field 0.59993 (Δ-0.00151). The prior 9/14-field labels were off by one; saved field lists and scores were correct.

## Weak Negative Evidence

1. **Removing `video_id` or `author_id` individually from the 5-field baseline FM modestly improved validation primary** (Investigation C01): five-seed paired means +0.00108/+0.00132, positive in 5/5 seeds each. This is a weak negative for this exact pointwise FM formulation only.

## Inconclusive Questions

1. **Aggregation window/population of `video_features_statistic_pure.csv`** cannot be pinned down from available files; only that it is much larger in scale than this dataset's own sampled traffic (Investigation E01).
2. **Whether validation "resembles late train more than early train"** is genuinely mixed: volume/entity-overlap structure says yes (late train's daily volume and video/user Jaccard are closer to validation), but raw long_view rate says the opposite (early train's rate is numerically closer) (Investigation A06). No recency-weighting conclusion should be drawn from this alone.
3. **Whether the random-exposure log's validation-period slice (288,338 rows) would make a useful unbiased diagnostic set** is structurally plausible (near-zero pair overlap with standard traffic, much lower/different long_view rate) but was not experimentally tested (Investigation F01).
4. **Whether `dur_bucket` meaningfully contributes to the baseline FM** — its removal effect (-0.0006) is only ~1.7x the combined 3-seed std, too small to classify confidently either way (Investigation C01).
5. **Whether lr=0.0003 is genuinely better than the official lr=0.001**, or within noise — delta (0.00035) is only ~1.3x the larger of the two stds (Investigation C03).

## Dataset Opportunities Not Yet Tested

Information sources that exist and appear structurally usable, listed without recommending which to pursue:

- **Historical/sequential user behavior** (Investigation D04/D05): substantial train-side history exists (median 35, p90 103 prior interactions per validation user); untested as a feature or sequence-model input.
- **Tag-level repeat affinity** (Investigations A05, D04/D05): dense (73–85% row coverage) compared to sparse exact video/author repeat; untested as a feature.
- **Video-statistics ratios** (Investigation E02): show real marginal association with `long_view` (r up to 0.30); untested for incremental value over the existing FM baseline, and carries the E01 aggregation-window caveat.
- **`tab` field's large marginal spread** (Investigation E01): already in the baseline; C01 shows it is load-bearing for the tested FM, while richer interaction structures remain untested.
- **Random-exposure log, validation-period slice** (Investigation F01): a candidate unbiased diagnostic set, untested.
- **Multi-task auxiliary signals** (Investigations D01–D03): two structurally distinct signal families identified (watch-related: is_click/play_time_ms; active-engagement: is_like/is_follow/is_comment/is_forward), differing greatly in prevalence; untested as auxiliary targets.
- **Joint list-length- and activity-tier-targeted modeling** (Investigations B01/B02/B05/B06): activity and list length are moderately associated (Spearman ρ=0.462), and T3/T4 × 6+ contains 51.72% of the current primary oracle gap; untested whether any specific method exploits this preferentially.

## Engineering Constraints

- Cold pipeline (load + encode + one FM training run to early-stop) took ≈57.5 seconds on the review rerun; ≈491MB peak RSS.
- Encoding-array caching gave a correctness-preserving ~263x reload speedup on that run; the exact factor is timing-dependent.
- **Windows-specific:** naive `subprocess.run(timeout=N)` does not bound wall-clock time when the tracked process spawns unmanaged children with inherited stdio; a harness enforcing per-iteration or 6-hour budgets on this OS needs `CREATE_NEW_PROCESS_GROUP` + `taskkill /T /F` or `psutil`-based recursive process-tree termination instead.
- Syntax errors and NaN/Inf submissions are already cleanly detectable via existing official code / standard subprocess semantics — no new guard code needed for those two specific cases.
- **System memory is highly variable and can be tight on this workstation**: the review rerun observed 0.68GB available of 16.76GB total while the pipeline's own peak RSS was ~491MB. Treat this as an environment snapshot, not a stable dataset property.
- Dependencies confirmed available in this environment beyond the Starter Kit's numpy-only requirement: pandas 2.3.2, scipy 1.18.0, matplotlib 3.10.6, psutil 7.0.0 — all research-only, not used to alter `source/`.

## Questions the Autonomous Agent Should Resolve Itself

- Whether the C01 video/author removal result should influence any model beyond the exact tested FM formulation.
- Whether and how to exploit tag-level repeat affinity (dense) versus exact video/author repeat affinity (sparse) — this audit only measured coverage, not modeled value.
- Whether historical/sequential features or a sequence model (DIN/SIM/etc.) are worth the implementation cost given the measured but moderate (not "hundreds to thousands") typical history depth (median 35, p90 103).
- Whether and how to use `video_features_statistic_pure.csv` given its unresolved aggregation-window uncertainty (Investigation E01) — a judgment call about acceptable risk, not resolved here.
- Whether a pointwise, pairwise (BPR), or listwise objective best exploits the metric structure found in B01/B02/B05/B06 (including the verified concentration in T3/T4 × 6+ users) — untested by this audit.
- Which auxiliary task(s), if any, to use in a multi-task formulation, and how to weight them given the very different prevalence/association profiles found in D01–D03.
- Whether the random-exposure log's validation-period slice is worth incorporating as a diagnostic set, and how.
- Whether the C01 finding (dropping `video_id`/`author_id` individually slightly improved this exact pointwise FM's validation primary) generalizes to any other model family/loss the agent might try, or is specific to this pointwise FM's interaction with this dataset's pair-sparsity — untested beyond the single-field leave-one-out ablation reported here.

---

# 6. Candidate Findings for Human Review

Do **not** edit `context/constraints.md` automatically. Presented for human review only.

### Candidate 1

**Finding:** The training log file `log_standard_4_08_to_4_21_pure.csv` contains zero rows for 2022-04-08 (only 13 distinct dates, not 14) and daily row volume decays roughly 14x from its 04-11 peak (278,835 rows) to the last train day 04-21 (20,021 rows).

**Evidence classification:** HARD FACT

**Supporting investigation:** A06

**Numerical evidence:** Missing date: 2022-04-08. Peak/trough ratio: 278,835 / 20,021 ≈ 13.9x. See daily table in `data_profile.md` §13.

**Confidence:** High (direct, deterministic count from the raw file).

**Recommended wording for constraints.md:** "`log_standard_4_08_to_4_21_pure.csv` has zero rows for the official train start date (2022-04-08); the file spans only 13 distinct dates. Daily interaction volume within train decays roughly 14x from its 04-11 peak to the 04-21 end of the period."

**Why it is safe to provide as prior evidence:** A deterministic, reproducible count from a raw file; does not depend on any model or hyperparameter choice.

**What should remain for the autonomous agent to decide:** Whether this decay pattern motivates any recency-weighting, truncation, or reweighting strategy — no such conclusion is drawn here (see A06's explicit non-conclusion).

---

### Candidate 2

**Finding:** Train-side activity and validation list length are moderately positively associated (Spearman ρ=0.462). Their joint T3/T4 × 6+ intersection contains 25.38% of validation users but 50.79% of official GAUC weight and 51.72% of the current baseline-to-oracle primary gap. Marginally, the 6–10 list bucket carries 36.39% of GAUC weight and T3+T4 carry 62.29%.

**Evidence classification:** HARD FACT

**Supporting investigation:** B01, B02, B05, B06

**Numerical evidence:** Official GAUC denominator = 34,592 positive rows from mixed-label users. The 30 joint cells reconcile exactly to 22,377 users, 124,909 rows, 100% of GAUC weight, and the full GAUC/nDCG/primary oracle gaps. See `data_profile.md` §8 and `phase_b_metric.json`.

**Confidence:** High (deterministic user-count association plus direct application of the official, unmodified `evaluate()` function to disjoint validation intersections; oracle numbers cross-checked exactly against `baseline_scores.json`'s published valid-oracle).

**Recommended wording for constraints.md:** "Train-side user activity and validation list length are moderately positively associated (Spearman ρ=0.462). The T3/T4 activity-tier × 6+ list-length intersection contains 25.38% of validation users, 50.79% of official GAUC weight, and 51.72% of the current baseline-to-oracle primary gap. This is a diagnostic concentration under the seed-0 baseline, not evidence that tier/list length causes the error or that a particular method will close it."

**Why it is safe to provide as prior evidence:** Descriptive of validation metric structure under the unmodified official evaluator and locally reproduced baseline scores; no evaluation-period labels are used.

**What should remain for the autonomous agent to decide:** Whether any specific hypothesis (loss function, model family, feature set) should be prioritized because it targets this intersection — the finding says where current headroom exists, not what closes it.

---

### Candidate 3

**Finding:** `is_click` and `play_time_ms`, at the same row, correlate with `long_view` at r=0.751 and r=0.632 respectively on the validation split; all other feedback signals correlate at |r|≤0.17.

**Evidence classification:** HARD FACT

**Supporting investigation:** D02

**Numerical evidence:** See table in `data_profile.md` §9 and `phase_d_feedback.json`.

**Confidence:** High (direct Pearson correlation on validation labels).

**Recommended wording for constraints.md:** "On this dataset, same-row `is_click` (r=0.751) and `play_time_ms` (r=0.632) are far more strongly associated with `long_view` than any other feedback signal (all others |r|≤0.17); this is additional, dataset-specific quantitative support for RULES.md's existing prohibition on same-row use of these fields as `long_view` inputs. It says nothing about their value as historical features or auxiliary targets, both of which remain untested."

**Why it is safe to provide as prior evidence:** Reinforces, with numbers specific to this dataset, a rule (C3 in `constraints.md`) that is already established; does not introduce a new modeling recommendation.

**What should remain for the autonomous agent to decide:** Whether/how to use any of these ten signals as historical features or auxiliary multi-task targets (explicitly untested here).

---

### Candidate 4

**Finding:** Products such as `show_cnt × counts` are near-integer for all videos and have a median reconstructed/observed train+valid ratio of about 11,000×. The undocumented aggregation population/window is likely larger than sampled train+valid traffic, but its identity and causal safety are unresolved.

**Evidence classification:** INCONCLUSIVE (on causal validity) built on a HARD FACT numeric ratio.

**Supporting investigation:** E01

**Numerical evidence:** `show_cnt × counts` integer for 100% of videos; reconstructed/observed ratio median 11,465x, p10 5,248x, p90 38,199x, 0% of videos below 1x. `counts` field itself ranges 45–181 (median 147), already exceeding the dataset's 31-day span.

**Confidence:** High on the numeric ratio itself; low/unresolved on what it implies about test-period leakage specifically.

**Recommended wording for constraints.md:** "`video_features_statistic_pure.csv`'s aggregation window and source population are not documented and are inferred to be much larger than this dataset's own sampled train+valid traffic (median reconstructed/observed impression ratio ≈ 11,000x). Any feature derived from this file should be treated as carrying unresolved causal-validity risk rather than assumed safe merely because it ships with the official dataset."

**Why it is safe to provide as prior evidence:** States an uncertainty explicitly (per DATA_GUIDE.md's own instruction) rather than asserting a false certainty in either direction.

**What should remain for the autonomous agent to decide:** Whether to use this file at all, and if so, how to bound or mitigate the identified risk — this is a judgment call the pre-audit deliberately leaves open.

---

### Candidate 5

**Finding:** On this Windows environment, `subprocess.run(cmd, timeout=N, capture_output=True)` does not reliably bound wall-clock time when the launched process spawns an unmanaged child of its own — verified with a 3-second timeout that actually took 30.13 seconds to return, matching an unmanaged grandchild's full runtime.

**Evidence classification:** HARD FACT

**Supporting investigation:** G03

**Numerical evidence:** Configured timeout 3s; actual elapsed 30.13s; grandchild sleep duration 30s. The match supports, but does not separately instrument, the inherited-pipe explanation.

**Confidence:** High for the observed overrun; medium for the inferred handle-level mechanism.

**Recommended wording for constraints.md:** "On Windows, `subprocess.run(timeout=N)` does not reliably bound wall-clock time if the tracked process spawns further unmanaged child processes that inherit stdio handles; a harness enforcing per-iteration or total wall-clock budgets on Windows should use `CREATE_NEW_PROCESS_GROUP` plus `taskkill /T /F`, or `psutil`-based recursive process-tree enumeration and termination, instead of a bare `subprocess.run(timeout=...)` call."

**Why it is safe to provide as prior evidence:** A platform/tooling fact independent of any modeling decision; directly relevant to RULES.md's timeout-recovery requirement and the 6-hour wall-clock limit.

**What should remain for the autonomous agent to decide:** The specific harness implementation choice (job objects vs. psutil vs. another mechanism) is left open.

---

### Candidate 6

**Finding:** In the official 5-field pointwise FM baseline (k=16, lr=0.001), individually removing `video_id` or `author_id` modestly improved validation primary; a reviewer rerun gave +0.00108 and +0.00132 over 5 matched seeds (positive in 5/5 for each). Original 3-seed removal results for `user_id`/`tab` were large drops (-0.00819/-0.01590); `dur_bucket` remained inconclusive (-0.00059).

**Evidence classification:** WEAK NEGATIVE (for video_id/author_id in this exact configuration); HARD FACT (for user_id/tab being load-bearing).

**Supporting investigation:** C01

**Numerical evidence:** See `research/review_artifacts/c01_ablation_reproduction.json` plus the original ablation table in `data_profile.md` §16.

**Confidence:** Medium-high for the narrow tested formulation; low for any generalization beyond it.

**Recommended wording for constraints.md:** "In the official pointwise-logloss FM (k=16, lr=0.001), individually removing `video_id` or `author_id` modestly increased validation primary (+0.00108/+0.00132 over 5 matched seeds; positive in 5/5 each). This applies only to this field set/model/training protocol and does not establish that video or author identity is generally uninformative."

**Why it is safe to provide as prior evidence:** A controlled, matched five-seed, validation-only ablation with consistent sign, explicitly scoped to avoid the "correlated fields ⇒ useless" overclaim.

**What should remain for the autonomous agent to decide:** Whether this generalizes to other model families/losses, and whether any specific feature-engineering or field-selection decision should follow from it.

---

### Candidate 7

**Finding:** The static-feature-expansion (`constraints.md` C5) and embedding-dimension (`constraints.md` C6) negative findings are independently consistent with controlled validation experiments in this environment.

**Evidence classification:** STRONG NEGATIVE for the exact tested formulations, based on validation evidence; published test results are reference context only.

**Supporting investigation:** C04, C05

**Numerical evidence:** Static-feature: base 0.60144 → item_8field 0.60111 → cwm_13field 0.59993 (validation). Embedding-dim: k=8/16/32/64 → 0.60111/0.60144/0.60146/0.60098 (validation, all mutually within ~1 combined std).

**Confidence:** High — same qualitative result on an independent split, using this audit's own encoder/training code path (not merely re-quoting the organizer's number).

**Recommended wording for constraints.md:** Append to existing C5/C6 entries: "Independently re-verified on the validation split (not merely the organizer's original test-split claim); see PRE_AUDIT.md Investigations C04/C05 for the validation-split numbers."

**Why it is safe to provide as prior evidence:** Strengthens existing, already-accepted constraints with independent corroboration rather than introducing a new claim.

**What should remain for the autonomous agent to decide:** Nothing new — this only raises confidence in constraints the agent was already going to be given.
