# PRE-AUDIT — Consolidated KuaiRand-Pure

> Purpose: empirical research notebook created **before** the final autonomous run, consolidated
> from three independently-produced pre-audits ("Claude", "Gemini", "GPT") plus their own
> independent review passes. This file gives the future autonomous agent evidence, not the answer.
>
> Every source PRE_AUDIT.md had already been through its own reviewer pass before this
> consolidation began; every "Review correction:" noted in a source file is preserved here. This
> merge adds one further cross-audit consistency pass (see `research/consolidated/MERGE_WORKLOG.md`
> for the full crosswalk and reasoning).

## 0. Scope and Rules

- Use train (2022-04-08..2022-04-21) + validation (2022-04-22..2022-04-28) only.
- Do not inspect or evaluate on evaluation/test labels.
- Do not modify official scoring (`evaluate.py`), official date splits, or raw source data.
- Do not use current-row post-impression feedback as a `long_view` input.
- Multi-task auxiliary targets are permitted; do not assume any auxiliary task helps.
- Historical feedback must be strictly earlier than the row being represented.
- Reproduce surprising results before trusting them.

### Evidence classes

- `HARD FACT` — deterministic dataset statistic, official rule/code, or mathematical consequence.
- `STRONG POSITIVE EVIDENCE` — controlled, repeated experiments consistently support a formulation.
- `WEAK POSITIVE EVIDENCE` — a formulation worked, but on limited/single-run evidence.
- `STRONG NEGATIVE EVIDENCE` — controlled repeated experiments convincingly argue against a tested formulation.
- `WEAK NEGATIVE EVIDENCE` — a tested formulation performed poorly, but the broader idea remains plausible.
- `INCONCLUSIVE` — evidence is insufficient, noisy, or conflicting.
- `ENGINEERING CONSTRAINT` — a fact about the runtime/tooling environment, not the dataset or model.
- `INVALID / FORBIDDEN` — a use that the competition rules explicitly disallow.

(Source audits additionally used `HARD FACT`/`STRONG NEGATIVE`/`WEAK NEGATIVE`/`INCONCLUSIVE`
only; this merge maps their labels onto the fuller vocabulary above where the fuller label is a
strictly more precise fit. No source finding was re-classified to a *stronger* class than its
originating audit assigned; several were kept conservative or annotated with the scope that
justifies the class.)

### Compliance status (verified across all three merged audits)

- **No evaluation/test-label leakage.** All three source PRE_AUDIT.md files are already
  post-review, corrected text. Each source's own review report documents and fixes an integrity
  issue found in that audit's *original* (pre-review) work: Claude's original pass locally scored
  test labels once and summarized evaluation-period random-log outcomes (both removed on review);
  Gemini's `GI_analysis.py` built one diagnostic's comparison set from unfiltered
  train+validation+test rows, touching test-period row *identities* only, no labels (corrected,
  759→702 pairs); GPT's audit avoided this class of issue from the start (date-filtered loaders
  throughout, and explicitly declined to run the organizer's `ablation_features.py` because it
  evaluates test labels). No evaluation/test-derived number appears anywhere in this consolidated
  package.
- **GAUC weight-share denominator.** Official `evaluate.py` semantics: GAUC is computed only over
  mixed-label users, weighted by each such user's positive-row count. Claude's original pass used
  the wrong denominator (all validation positives, including uniform-label users) for its
  activity/list-length "GAUC weight share" tables; this was corrected on review (34,592 positives
  from mixed-label users only). GPT's equivalent tables used the correct denominator from the
  start. Every GAUC-weight-share figure in this consolidated package uses the corrected
  denominator.
- **Official / structural facts already in `context/constraints.md` (C1-C6)** are treated as
  higher authority than any pre-audit interpretation and are not re-derived here except where a
  pre-audit independently re-verified them on validation (noted explicitly in §6).

---

## 1. Baseline Reproduction

### Question
Does the local environment reproduce the officially published validation baseline closely enough
to trust deeper analysis built on it?

### Data / scope
Train + validation standard logs, `video_features_basic_pure.csv`, the unmodified official `FM`
class and `evaluate()` function from `source/starter-kit/`. No test label was accessed by any of
the three audits for this measurement.

### Method
All three audits independently built a research-only loader/encoder around the same five official
fields (`user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`) and the official FM
(k=16, lr=0.001, batch 8192, Adam, max 40 epochs, patience 4, early-stopped on validation primary),
then called the unmodified official `evaluate()`.

### Result

| Source | GAUC | nDCG@5 | Primary | Notes |
|---|---:|---:|---:|---|
| Published (organizer) | 0.6674 | 0.5357 | 0.6016 | seed std ≈ 0.0008 |
| GPT (seed 0) | 0.667133 | 0.535806 | 0.601470 | best epoch 7, stop epoch 11; cold run 78.52s |
| Claude (seed 0) | 0.6671 | 0.5358 | 0.6015 | matches GPT to reported precision |
| Claude (5-seed mean) | 0.66740 | 0.53574 | 0.60157 | own-environment std 0.00031/0.00038/0.00032 |
| Gemini | not independently reproduced as a standalone check; oracle nDCG@5 (0.6969) cross-validated to 4 decimals against organizer's published valid-oracle (0.6968) | | | |

All differences are inside the published seed std. Train/validation/(evaluation date-only) row
counts match exactly across all three audits: 1,141,112 / 124,909 / 170,588.

### Evidence classification
`HARD FACT`.

### Interpretation
Environment, data loading, encoding, and evaluator logic reproduce the published baseline in all
three independently-built codebases. This is a precondition for trusting every downstream
investigation in this document: three separate implementations converge on the same baseline
numbers.

### What it DOES NOT establish
Nothing about feature usefulness or model quality beyond the baseline itself.

### Source provenance
- Claude (unlabeled §3 baseline-reproduction section; C02 seed variance)
- GPT R00
- Gemini (B01's oracle cross-check only, no standalone reproduction)

---

## 2. Dataset Structure

### 2.1 Entity cardinalities

| Measurement | Train | Validation |
|---|---:|---:|
| Unique users | 26,210 | 22,377 |
| Unique videos | 7,538 | 5,951 |
| Unique authors | 6,482 | 5,315 |
| Unique `tab` values | 15 | 15 |
| Rows | 1,141,112 | 124,909 |

Side-table totals: `user_features_pure.csv` has 27,285 users (100% coverage of train/valid users);
`video_features_basic_pure.csv` / `video_features_statistic_pure.csv` each have 7,583 videos (100%
coverage of train/valid videos).

Tag cardinality: 110 unique tag strings (GPT, full `video_basic` file, nonmissing) / 110-104
train-valid log-observed tags (Gemini). Claude's independent count reports 111 for the same field;
the exact scope difference was not resolved during this merge (see MERGE_WORKLOG row 3) — treat
110 as the primary figure (2/3 agreement, more precisely documented scope) and 111 as an
unreconciled ±1 discrepancy of no material consequence. GPT separately distinguishes 46 *parsed*
tag tokens (comma-split values within the tag string) from the 110 raw tag strings — this
distinction matters for §3's tag-overlap figures below.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Claude A01, Gemini A01, GPT A01 — exact match on all counts except the
110-vs-111 tag-string discrepancy noted above.

### 2.2 Missingness

| File / field | Missing % |
|---|---:|
| `video_features_basic_pure.csv`: `video_duration` | 3.152% |
| `video_features_basic_pure.csv`: `music_type` | 2.677% |
| `video_features_basic_pure.csv`: `tag` | 1.266% |
| `user_features_pure.csv`: `onehot_feat4` | 3.203% |
| `user_features_pure.csv`: `onehot_feat12`–`onehot_feat17` | 2.617% each |
| Standard interaction logs (all columns) | 0% |
| `video_features_statistic_pure.csv` (all columns) | 0% |

**Evidence classification:** `HARD FACT`.

**Interpretation:** Missingness is low (<3.3%) everywhere and concentrated in a small set of
side-feature columns not used by the official baseline. It is survivable with a simple UNK/
sentinel bucket per field. Whether missingness is systematically related to `long_view` (rather
than missing at random) was not tested by any of the three audits.

**Source provenance:** Claude A02, Gemini C01 (review-added), GPT A01 — exact match to available
precision across all three.

### 2.3 `visible_status` is constant

`video_features_basic_pure.csv`'s `visible_status` field has cardinality 1 (100% one value) across
the full file. Claude explicitly interprets this as carrying zero information and droppable
outright; GPT independently notes the cardinality-1 fact; Gemini does not flag it.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Claude E01 (explicit interpretation), GPT A01/G01 (noted).

---

## 3. Entity Overlap and Redundancy

### 3.1 Train → validation entity and pair overlap

| Measurement | Value | Source agreement |
|---|---:|---|
| Validation users seen in train | 98.11%–98.114% | exact match, 3/3 |
| Validation videos seen in train | 99.88%–99.882% | exact match, 3/3 |
| Validation authors seen in train | 99.906%–99.91% | exact match, 3/3 |
| Validation user-video PAIRS seen in train | 1.627%–1.63% | exact match, 3/3 |
| Validation user-author PAIRS seen in train | 3.376%–3.38% | exact match, 3/3 |
| Cold validation users | 422 (1.89%) | Claude, GPT |
| Cold validation videos | 7 (0.12%) | Claude, GPT |

**Evidence classification:** `HARD FACT`.

**Interpretation:** This is overwhelmingly a warm-entity, unseen-interaction ranking problem, not
a cold-start problem, at the user/video/author level (>98% overlap by count, and cold rows are an
even smaller share of validation rows than cold entities are of validation entities). But at the
exact `(user, video)` pair level, only ~1.6% of validation impressions repeat an exact pair seen in
train — any model term that memorizes exact `(user, video)` pairs will have very limited direct
coverage; the FM's `user_id × video_id` interaction is therefore mostly extrapolating from
marginals and coarser interactions, not memorizing.

**What it does NOT establish:** Which model family best exploits this structure.

**Source provenance:** Claude A03, Gemini D01, GPT A02.

### 3.2 Train → validation user-tag pair overlap — scope-dependent figures (not a conflict)

Two genuinely different tag-pair definitions were used across the three audits:

| Definition | Value | Source |
|---|---:|---|
| Tag treated as one opaque categorical string (raw `tag` field value) | 68.14% | Claude A03, Gemini D01 |
| Tag treated as parsed, comma-separated tokens (a video can match on any shared token) | 71.913% (68,316 / ~95,010 pairs) | GPT A02/A03 |

These are **not conflicting measurements of the same quantity** — they measure user-tag overlap
under two different definitions of "same tag," and the parsed-token definition is mechanically
looser (any shared token counts, not the whole string matching), which explains why it reports a
higher overlap rate. Both are valid HARD FACTs under their own stated definition. A downstream
agent choosing to build a tag-based feature must decide, and record, which of these two
granularities it is using.

**Evidence classification:** `HARD FACT` (each figure, under its own stated scope).

**Source provenance:** Claude A03, Gemini D01, GPT A02/A03/A04 (which explicitly documents the
110-string / 46-token distinction that explains the gap).

### 3.3 Author → video redundancy

| Measurement | Value | Scope | Source |
|---|---:|---|---|
| Authors with exactly 1 video | 86.959%–86.96% (5,661/6,510) | full `video_basic` file | Claude A04, Gemini A01, GPT A04 |
| Authors with exactly 1 video, restricted to train/valid-observed authors | 87.05%–87.07% (5,647/6,487, or 87.07% train-only) | observed-only (train-only for Claude; train+valid for GPT) | Claude A04, GPT A04 |
| Videos/author — median | 1.0 | both scopes | all three |
| Videos/author — mean | 1.16–1.165 | full-table | all three |
| Videos/author — p90 | 2 (full table) / 3 (observed-only) | scope-dependent | Claude, GPT |
| Videos/author — max | 26 (full table) / 24 (observed-only) | scope-dependent | Claude, GPT |
| Video → author functional mapping | 100% (every observed video maps to exactly one author) | observed-only | GPT A04 |

The p90/max discrepancy (2 vs 3; 26 vs 24) is a direct, reconcilable consequence of the
full-video_basic-table vs. observed-in-logs-only scope difference, not a data error.

**Evidence classification:** `HARD FACT`.

**Interpretation:** ~87% of authors have exactly one video regardless of scope; `author_id` is
near-redundant with `video_id` for the large majority of the catalog, with marginal information
concentrated in the ~13% of authors with multiple videos (up to 24-26 videos for the most prolific
author, depending on scope).

**What it does NOT establish:** Whether dropping `author_id` from a trained model changes its
output — correlated fields can still carry distinct regularization/generalization value even when
mostly redundant. Tested directly in §6.2.

**Source provenance:** Claude A04, Gemini A01 / data_profile §6, GPT A04.

### 3.4 Repeat-pair frequency within TRAIN

| Pair type | % pairs repeated >1x | % rows in repeated pairs | Source agreement |
|---|---:|---:|---|
| user-video | 4.13%–4.130% | 8.19%–8.194% | exact match, Claude/GPT (Gemini quotes the same figure as cross-reference) |
| user-author | 5.91%–5.913% | 11.75%–11.750% | exact match, Claude/GPT |
| user-tag (raw string) | 51.77% (Claude, review-corrected) | 84.98% | Claude only |
| user-tag (parsed tokens) | 55.250% | 87.819% | GPT only — same tag-granularity scope difference as §3.2 |

**Evidence classification:** `HARD FACT`.

**Interpretation:** Exact item/author repeat-affinity within train is a sparse, single-digit-percent
signal; tag-level repeat-affinity (under either tag definition) is dense, covering a majority of
train rows. This constrains expectations for any "user has seen this video/author before" feature:
it will be sparse by construction, unlike a tag-level equivalent.

**Source provenance:** Claude A05 (with its own review-corrected missing-tag handling), GPT A03.

### 3.5 Row-level repeat coverage in VALIDATION (user's prior TRAIN history contains the same entity)

| Coverage | Claude | GPT | Gemini | Agreement |
|---|---:|---:|---:|---|
| Same video seen before | 1.62% | 1.624% | 1.58% | Claude/GPT match near-exactly; Gemini close but ~0.04pp lower, reason not established |
| Same author seen before | 3.38% | 3.381% | 3.27% | Claude/GPT match near-exactly; Gemini close but ~0.11pp lower, reason not established |
| Same tag seen before (raw string) | 73.19% | — | — | Claude only |
| Same tag seen before (parsed tokens) | — | 78.413% | — | GPT only — same tag-granularity scope difference as §3.2 |

**Evidence classification:** `HARD FACT` for the Claude/GPT-matching video and author figures;
Gemini's close-but-different values are reported as a minor, unresolved discrepancy rather than
folded into the primary figure.

**Source provenance:** Claude D04/D05, Gemini F01 / data_profile §11, GPT A02/F01.

---

## 4. Metric Structure

### 4.1 Uniform-label / invariant validation users

| Type | Users | % Users | Rows | % Rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.32%–30.321% | 21,807 | 17.46%–17.458% |
| All positive | 2,663 | 11.90%–11.901% | 4,540 | 3.63%–3.635% |
| Mixed / movable | 12,929 | 57.78%–57.778% | 98,562 | 78.91%–78.907% |
| Single impression (subset of the above) | 3,917 | 17.50%–17.505% | 3,917 | — |

**Evidence classification:** `HARD FACT`.

**Interpretation:** 42.2% of validation users (30.32% all-negative + 11.90% all-positive) have a
`long_view` nDCG@5 that is mathematically fixed regardless of ranking, and all-negative/all-positive
users are entirely excluded from GAUC. Only the 57.78% mixed-label users are "movable" for GAUC;
mixed-label users account for 78.91% of validation rows.

**What it does NOT establish:** Any training reweighting or sampling rule — this states the
correct denominator for reasoning about metric headroom, nothing more.

**Source provenance:** Claude B03, Gemini B01, GPT B01 — exact triplicate match.

### 4.2 Oracle vs. baseline nDCG@5 by validation list length

Oracle nDCG@5 (labels used as scores; model-independent, purely a function of list length and
label composition) — exact match across all three audits, and independently cross-validated by
Gemini against the organizer's published `baseline_scores.json` valid-oracle (0.6969 recomputed vs
0.6968 published):

| List length | Users | Rows | Oracle nDCG@5 | Baseline nDCG@5 (Claude/GPT, official-config seed 0) | Baseline nDCG@5 (Gemini, 12-epoch-capped single seed — lower confidence) | GAUC weight share |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 3,917 | 0.4054 | 0.4054 | 0.4054 | 0.00% |
| 2–3 | 6,218 | 15,323 | 0.6086 | 0.5413 | 0.5413 | 10.27% |
| 4–5 | 4,119 | 18,326 | 0.7492 | 0.6185 | 0.6140 | 16.36% |
| 6–10 | 5,225 | 39,587 | 0.8536 | 0.5913 | 0.5880 | **36.39%** |
| 11–20 | 2,346 | 32,609 | 0.9182 | 0.5037 | 0.4875 | 27.08% |
| 21+ | 552 | 15,147 | 0.9420 | 0.3934 | 0.4008 | 9.90% |

Overall validation list-length distribution: min 1, median 4, mean 5.58, p90 12, p99 26, max 74
(all three audits agree).

**Evidence classification:** `HARD FACT` for the oracle column and list-length distribution
(deterministic, triplicate). `HARD FACT` for the Claude/GPT baseline column (independently matches
to 4 decimals — both ran the fully-converged official-config seed-0 baseline). Gemini's baseline
column is retained as a separately-labeled, lower-confidence variant (12-epoch cap, single seed,
explicitly caveated by Gemini's own review as non-official-config) — its qualitative shape agrees
(headroom grows with list length, longer lists have lower baseline nDCG@5) but its exact values on
longer-list buckets diverge from the fully-converged run.

**Interpretation:** The 6-10 length bucket carries the single largest official GAUC weight share
(36.39%) and, weighted by its user count, the largest contribution to the aggregate nDCG oracle
gap. The 11-20 and 21+ buckets have larger *per-user* gaps but far fewer users. As a mechanical
identity, length-1 lists have oracle nDCG@5 exactly equal to baseline nDCG@5 (nothing to reorder).

**What it does NOT establish:** That any particular model change (e.g. a listwise loss) would
close this gap — only where the gap is largest and how it is weighted under the official metric.

**Source provenance:** Claude B02, Gemini B01, GPT B03 — oracle/GAUC-weight columns triplicate
(Claude/GPT exact match, absent from Gemini for GAUC weight share); baseline column Claude/GPT
exact match, Gemini separate lower-confidence variant.

### 4.3 Activity-tier bucket analysis — three non-comparable tier schemes

All three audits partition validation users by train-side activity and evaluate GAUC/nDCG@5/GAUC
weight per tier, but **each audit used a different tier definition**. These are not the same
measurement and their numeric values must not be merged or averaged.

**Claude** (tier edges at train-impression-count 17/36/65, quartiles among validation users with
≥1 train row; Cold = 0 train rows):

| Tier | Users | Rows | GAUC | nDCG@5 | Oracle nDCG@5 | Movable nDCG Gap | GAUC Weight Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 0.112 | 1.69% |
| T1 (1–17) | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 0.087 | 14.67% |
| T2 (18–36) | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 0.127 | 21.35% |
| T3 (37–65) | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 0.177 | 27.50% |
| T4 (66+) | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 0.262 | **34.79%** |

**GPT** (tier edges = train-count quartiles 13/31/59 over all train users; Cold = 0):

| Tier | Users | Rows | GAUC | nDCG@5 | Primary | Invariant users | GAUC weight | Overall nDCG-gap contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6091 | 57.11% | 1.69% | 0.0021 |
| T1 (1–13) | 4,351 | 13,437 | 0.6550 | 0.5374 | 0.5962 | 61.25% | 10.55% | 0.0155 |
| T2 (14–31) | 5,582 | 23,310 | 0.6686 | 0.5409 | 0.6047 | 48.01% | 19.95% | 0.0298 |
| T3 (32–59) | 5,791 | 32,052 | 0.6624 | 0.5521 | 0.6073 | 37.70% | 27.80% | 0.0429 |
| T4 (60+) | 6,231 | 54,120 | 0.6720 | 0.5154 | 0.5937 | 26.95% | **40.01%** | 0.0707 |

**Gemini** (fixed absolute thresholds Cold=0/T1<10/T2 10-49/T3 50-149/T4 150+; single-seed,
12-epoch-capped FM — lower confidence than Claude/GPT's fully-converged runs):

| Tier | Users | Valid Rows | GAUC | nDCG@5 | Invariant Users % |
|---|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6741 | 0.5262 | 57.11% |
| T1 (<10) | 2,897 | 8,721 | 0.6475 | 0.5344 | 62.82% |
| T2 (10–49) | 11,138 | 49,716 | 0.6590 | 0.5444 | 45.97% |
| T3 (50–149) | 7,119 | 53,802 | 0.6620 | 0.5282 | 28.99% |
| T4 (150+) | 801 | 10,680 | 0.6856 | 0.4069 | 25.34% |

**Evidence classification:** `HARD FACT` for each audit's own numbers, under its own stated tier
definition.

**Interpretation:** Despite using three different tier boundaries and (in Gemini's case) a
different, capacity-limited baseline run, all three independently reach the same *qualitative*
conclusion: GAUC weight and movable nDCG headroom both concentrate in the highest train-activity
tier, and the invariant-user share falls roughly monotonically as activity rises. The exact
percentage of GAUC weight attributed to "the top tier" (34.79% Claude vs. 40.01% GPT) is not
comparable as a single number because the two audits' top tiers cover different population
fractions and different absolute activity thresholds — this is a genuine scope difference, not a
disagreement about the underlying data.

Gemini separately flags a confound worth preserving: its T4 has the *lowest* nDCG@5 despite the
*highest* GAUC, and Gemini shows this is explained by list-length structure (T4 users also have
the longest validation lists, and §4.2 already shows achieved nDCG@5 falling for long lists purely
from list-length structure) rather than being an independent "high-activity users are harder"
effect. Claude's joint analysis (§5) directly measures this activity/list-length confound rather
than only flagging it.

**What it does NOT establish:** Why any tier's nDCG is comparatively low, nor which
feature/model change would close a given tier's gap — only that gaps exist and roughly where.

**Source provenance:** Claude B01/B04, Gemini B01 (+ review addendum), GPT B02.

---

## 5. Activity / List-Length Analysis

### 5.1 Joint activity-tier × list-length analysis (Claude-only)

### Question
Are users with more train-side activity also the users with longer validation impression lists,
and how much metric weight and headroom lies in their actual intersection (as opposed to each
dimension's separate marginal view in §4.2/§4.3)?

### Data / scope
Train-side impression count per validation user (Claude's own tier scheme, §4.3) crossed with
validation-side list length (§4.2's buckets). Validation only for scoring; no evaluation-period
rows or labels accessed.

### Method
Spearman/Pearson correlation between the two raw counts; a 5×6 user cross-tab; official evaluator
applied to all 30 disjoint activity-tier × list-length cells and to several named intersections.
GAUC contribution uses the official positive-count weighting from mixed-label users; nDCG
contribution uses equal user weighting.

### Result

- Spearman correlation (all validation users): 0.4620; (warm users only): 0.4677. Pearson: 0.4419.
  The two dimensions are moderately, not perfectly, associated.
- Share of users with lists of 6+ rises from 14.84% (T1) to 64.07% (T4).

| Intersection | Users | Rows | GAUC Weight | Total GAUC Gap | Total nDCG Gap | Total Primary Gap |
|---|---:|---:|---:|---:|---:|---:|
| T4 × list 6+ | 3,453 (15.43%) | 42,020 (33.64%) | 30.35% | 29.88% | 34.37% | 31.35% |
| T3/T4 × list 6+ | 5,680 (25.38%) | 64,133 (51.34%) | 50.79% | 50.65% | 53.94% | 51.72% |
| T2/T3/T4 × list 6+ | 7,165 (32.02%) | 78,253 (62.65%) | 64.36% | 63.65% | 65.83% | 64.36% |
| T3/T4 × list 11+ | 2,307 (10.31%) | 38,219 (30.60%) | 27.80% | 27.96% | 28.64% | 28.18% |

All 30 disjoint joint cells reconcile exactly to 22,377 users, 124,909 rows, 100% of official GAUC
weight, and the complete baseline-to-oracle GAUC/nDCG/primary gaps.

### Evidence classification
`HARD FACT`.

### Interpretation
The T3/T4 × list-6+ intersection contains only 25.38% of validation users but roughly half of the
official GAUC weight (50.79%) and roughly half of the current baseline-to-oracle primary gap
(51.72%). This establishes that §4.2 and §4.3's separate marginal findings were directionally
consistent because activity and list length are moderately associated — but the concentration is
substantial, not exclusive: about half of the primary gap remains *outside* T3/T4 × 6+.

### What it DOES NOT establish
That activity or list length causes poor rankings, that either quantity is a suitable model input,
or that a targeted loss/model will close the measured gap. These percentages describe the current
seed-0 baseline's headroom under the official evaluator; a different model could redistribute them.

### Source provenance
Claude B06 (post-review extension). Not attempted by Gemini or GPT.

---

## 6. Baseline Mechanism and Ablations

### 6.1 Exact baseline mechanism

Fields: `user_id`, `video_id`, `author_id`, `tab`, ten-bin train-quantile `dur_bucket`. FM: first-
order weights plus all pairwise embedding interactions, k=16. Objective: pointwise binary
cross-entropy/logistic loss on `long_view`. Optimizer: hand-written Adam (β1=0.9, β2=0.999,
ε=1e-8, lr=0.001, L2=1e-6 on W/V; bias updated by plain gradient descent). Batch 8,192, shuffled
each epoch, max 40 epochs, early-stopped on validation primary after 4 non-improving epochs
(improvement threshold 1e-5). Unknown validation categories map to one reserved UNK slot per
field.

**Evidence classification:** `HARD FACT` (line-by-line source inspection, GPT C01, cross-checked
against `source/starter-kit/baseline.py`; consistent with Claude's and Gemini's descriptions).

**Interpretation:** The training objective is pointwise while both scored metrics (GAUC, nDCG@5)
are within-user ranking metrics — a structural mismatch between what the baseline optimizes and
what is scored, worth keeping in mind when interpreting every ablation below.

**What it does NOT establish:** That a pairwise or listwise objective would improve validation —
untested by any of the three audits.

**Source provenance:** GPT C01 (most detailed); Claude/Gemini's baseline descriptions (§1) are
consistent with this and add no conflicting detail.

### 6.2 Field ablations — leave-one-out from the 5 official fields

Two independently-built codebases (Claude, GPT) converge closely on the same deltas:

| Field removed | Claude (3-seed) | Claude (5-seed reviewer rerun) | GPT (3-seed, or 5-seed where noted) | Agreement |
|---|---:|---:|---:|---|
| `user_id` | −0.00819 | not rerun | not tested | Claude only |
| `tab` | −0.01590 | not rerun | −0.015903 ± 0.000467 | near-exact match |
| `dur_bucket` | −0.00059 | not rerun | −0.000591 ± 0.000156 | near-exact match |
| `video_id` | +0.00136 | +0.00108 (positive 5/5) | +0.001082 ± 0.000585 (5-seed) | Claude's 5-seed rerun and GPT's independent 5-seed result match near-exactly |
| `author_id` | +0.00157 | +0.00132 (positive 5/5) | +0.001316 ± 0.000426 (5-seed) | same near-exact match |

Base full-5-field mean primary: Claude 0.60144 (3-seed) / 0.60157 (5-seed); GPT 0.601440 ± 0.000275
(3-seed). Local seed-to-seed noise floor (Claude C02, 5 seeds): std 0.00032 on primary — smaller
than, and the same order of magnitude as, the organizer's published 0.0008 (treat 0.0008 as the
more conservative reference; 5 local seeds estimate variance imprecisely).

**Evidence classification:** `STRONG NEGATIVE EVIDENCE` against removing `tab` (effect ~20-40x the
combined std). `WEAK NEGATIVE EVIDENCE` against removing `dur_bucket` (effect only ~1.7-2x
combined std). `WEAK NEGATIVE EVIDENCE` — narrowly scoped — for the claim that retaining the exact
dual `video_id` + `author_id` field pair helps *this exact pointwise FM*: removing either one
individually, in this formulation, did not hurt and produced a small but consistent, direction-
positive effect across 10 combined paired-seed runs (5 from Claude's rerun, 5 from GPT's
independent implementation, all 10 positive). `HARD FACT` that `user_id` is load-bearing (Claude
only; not cross-validated by GPT, which did not test this ablation).

**Interpretation:** `user_id` and `tab` are indispensable for this baseline; dropping `tab` alone
brings primary close to the item-popularity baseline (~0.58). `dur_bucket`'s individual
contribution is small and only weakly distinguishable from seed noise. The genuinely non-obvious
finding, independently reproduced by two separate codebases with 10 combined paired-seed runs, is
that removing `video_id` or `author_id` individually — the two most granular, highest-cardinality
fields — did not hurt and modestly *improved* validation primary in this exact pointwise FM. A
plausible mechanism (untested as a causal claim): §3.1 shows only ~1.6% of validation user-video
pairs repeat exactly from train, so the `video_id`/`author_id` pairwise interaction terms are
mostly fit to combinations that do not recur in validation, adding capacity/noise rather than
validation-relevant signal, given that `tab` and `dur_bucket` already carry usable item-side
information at this dataset's small catalog scale (~7.5K videos).

**What it does NOT establish:** That `video_id`/`author_id` are useless in general — only that, in
this exact pointwise-logloss FM at k=16/lr=0.001, removing either alone did not hurt across 10
combined paired-seed runs from two independent implementations. Does not test removing both
simultaneously, another loss/model family, or another dataset regime. Gemini did not run any field
ablation and provides no corroborating or conflicting evidence on this point.

**Source provenance:** Claude C01 (3-seed original + 5-seed reviewer rerun), GPT C02 (3-seed, 5-seed
for the two identity-field removals). Gemini: not attempted.

### 6.3 Local seed variance

Claude C02: 5 seeds, official 5-field config — mean primary 0.60157, population std 0.00032 (GAUC
std 0.00031, nDCG@5 std 0.00038). This is the same order of magnitude as, but tighter than, the
organizer's published 0.0008; the organizer figure remains the more conservative reference since 5
local seeds estimate variance imprecisely, and non-default configurations (different k, lr, field
sets — see below) show their own, sometimes larger, seed variance.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Claude C02 only (as a named investigation). GPT reports comparable
per-configuration standard deviations throughout its own ablation/sweep tables (e.g. 0.000275 for
its 3-seed base config), which corroborate a similar noise scale without constituting an
independent "seed variance" investigation. Gemini did not measure this.

### 6.4 Learning-rate sensitivity

Two audits swept different, only-partially-overlapping grids:

| lr | Claude mean primary ± std (3-seed) | GPT mean primary ± std (3-seed) |
|---:|---:|---:|
| 0.0003 | 0.60179 ± 0.00011 | not tested |
| 0.0005 | not tested | 0.601776 ± 0.000280 |
| 0.001 (official) | 0.60144 ± 0.00027 | 0.601440 ± 0.000275 |
| 0.002 | not tested | 0.601364 ± 0.000826 |
| 0.003 | 0.60009 ± 0.00084 | not tested |
| 0.01 | 0.59709 ± 0.00053 | not tested |

**Evidence classification:** `WEAK NEGATIVE EVIDENCE` for lr ≥ 0.003 (Claude: clear, if noisy,
degradation). `INCONCLUSIVE` for lr=0.0003 vs. 0.001 (Claude: delta only ~1.3x the larger std) and
for lr=0.0005/0.002 vs. 0.001 (GPT: both deltas straddle zero within their own seed spread).

**Interpretation:** The official lr=0.001 is a reasonable default and not an obvious quick win to
change within either tested grid; only clearly higher rates (≥0.003) show unambiguous degradation.

**Source provenance:** Claude C03, GPT D02. Gemini: not attempted.

### 6.5 Static-feature expansion (organizer dead-end reproduction on validation)

Two independently-built codebases converge closely:

| Config | Claude | GPT | Agreement |
|---|---:|---:|---|
| base (5 fields) | 0.60144 | 0.601440 ± 0.000275 | match |
| +3 item fields (`music_id`,`video_type`,`upload_type`) → 8 total | 0.60111 (Δ−0.00033) | 0.601108 ± 0.000461 (Δ−0.000332 ± 0.000205) | match to ~5 decimals |
| +5 user-bucket fields → 13 total (CWM-style) | 0.59993 (Δ−0.00151) | 0.599930 ± 0.000523 (Δ−0.001510 ± 0.000792, all 3 seeds lower) | match to ~5 decimals |

**Evidence classification:** `STRONG NEGATIVE EVIDENCE` for the full 13-field expansion (both
audits, independently). `INCONCLUSIVE` for the 8-field item-only expansion (GPT: ≈1.6σ,
directionally negative but not clearly distinguishable from seed noise with 3 seeds; Claude's
matching Δ is of the same small magnitude).

**Interpretation:** This independently and exactly cross-validates the organizer's existing
`constraints.md` C5 finding *on the validation split*, not merely by re-quoting the organizer's
original test-split numbers. Two separately-coded implementations reaching matching deltas to five
decimal places is strong corroboration.

**What it does NOT establish:** That no additional feature could help — only that this exact
field-expansion formulation, in this exact FM, does not.

**Source provenance:** Claude C04 (with its own review-corrected field-count labeling, 8/13 not
9/14), GPT D01. Gemini explicitly did not reproduce this locally, deferring to the organizer's
existing evidence (constraints.md C5) as reference context only.

### 6.6 FM embedding-dimension sweep

| k | Claude | GPT | Agreement |
|---:|---:|---:|---|
| 8 | 0.60111 ± 0.00080 | 0.601110 ± 0.000796 | match |
| 16 (official) | 0.60144 ± 0.00027 | 0.601440 ± 0.000275 | match |
| 32 | 0.60146 ± 0.00069 | 0.601460 ± 0.000688 | match |
| 64 | 0.60099 (PRE_AUDIT text) / 0.60098 (data_profile) ± 0.00044 | not tested | Claude only |

**Evidence classification:** `STRONG NEGATIVE EVIDENCE` against simple FM capacity scaling: all
tested k values are mutually within ~1 combined std of each other.

**Interpretation:** Independently and exactly cross-validates `constraints.md` C6 on the validation
split for k=8/16/32 (two matching implementations), and extends the null result one octave further
(k=64) than either the organizer or GPT tested.

**What it does NOT establish:** That a different, more expressive model family (DeepFM/DCN/xDeepFM)
would show the same flatness — only that this FM's raw embedding width is not the bottleneck.

**Source provenance:** Claude C05, GPT D02. Gemini: deferred to organizer's constraints.md C6,
not reproduced locally.

---

## 7. Post-Impression Feedback Structure

### 7.1 Prevalence / distribution

| Signal | Train | Validation | Notes |
|---|---:|---:|---|
| `is_click` | 46.34%–46.345% | 44.38%–44.383% | Dense |
| `is_like` | 1.868%–1.87% | 1.797%–1.80% | Sparse |
| `is_follow` | 0.101%–0.10% | **0.130%** (all three agree, incl. GPT's own reviewer correction from a 0.131% transcription slip) | Very sparse |
| `is_comment` | 0.257%–0.26% | 0.233% | Very sparse |
| `is_forward` | 0.100%–0.10% | 0.078% | Very sparse |
| `is_hate` | 0.042%–0.04% | 0.062% | Extremely sparse |
| `is_profile_enter` | 2.539%–2.54% | 1.945%–1.95% | Sparse |
| `play_time_ms` | mean 23,260–23,260.5 | mean 21,487–21,486.8, median 4,607–4,970 (split-dependent), p90 ≈62,800, p99 ≈206,000–213,000, ~12–14% exactly zero | Dense continuous |
| `profile_stay_time` | mean ≈3.3 | mean ≈1.9 | >99.98% exactly zero |
| `comment_stay_time` | mean ≈553 | mean ≈460 | ~95% exactly zero |

**Evidence classification:** `HARD FACT`.

**Interpretation:** `is_click` and `play_time_ms` are dense on both splits; `is_like`/
`is_profile_enter` are sparse-but-present (~2%); `is_follow`/`is_comment`/`is_forward`/`is_hate`
are all under 0.3% positive on validation — extremely sparse. `profile_stay_time` is almost
entirely zero and unlikely to carry row-level signal in raw form.

**What it does NOT establish:** Whether a sparse signal is still useful as an auxiliary task
(rare events can still transfer useful gradient) — only that a naive dense-supervision auxiliary
head would starve on the four rarest signals.

**Source provenance:** Claude D01, Gemini E01, GPT E01 — exact triplicate match, and this is one
of the clearest instances of the merge's 2-of-3-corroborate-a-review-correction pattern: Claude
and Gemini's `is_follow` figure (0.130%) was already correct in their original work, independently
matching what GPT's own review process had to fix.

### 7.2 Same-row `long_view` association (diagnostic only — never a model input)

| Signal | Pearson r (validation, Claude/Gemini) | Pearson r (train, GPT) |
|---|---:|---:|
| `is_click` | 0.751 (Claude) / 0.7515 (Gemini) | 0.7605 |
| `play_time_ms` (raw) | 0.632 (Claude) / 0.6319 (Gemini) | 0.6351 (raw) / 0.5960 (log1p) |
| `is_profile_enter` | 0.127 | 0.1461 |
| `comment_stay_time` | 0.169 | 0.2702 (log1p) |
| `is_like` | 0.095 | 0.0992 |
| `is_comment` | 0.059 | 0.0590 |
| `is_follow` | 0.025 | 0.0250 |
| `is_forward` | 0.025 | 0.0226 |
| `is_hate` | −0.004 | −0.0039 |
| `profile_stay_time` | −0.0005 | 0.0079 (log1p) |

Claude and Gemini computed this on validation; GPT computed it on train. Once the split is
accounted for, all values are consistent (no genuine conflict — e.g. click r ≈0.75-0.76 on both
splits).

**Evidence classification:** `HARD FACT`.

**Interpretation:** `is_click` and `play_time_ms` are almost mechanically tied to `long_view` at
the same row — expected, since `long_view` is a thresholded function of watch time, itself gated by
whether a click/play happened. This is dataset-specific, quantified evidence for *why* the
same-row-leakage rule (RULES.md §4, constraints.md C3) matters here: a model using either field as
a same-row input would trivially and unrealistically inflate apparent performance. All ten
feedback signals correlate with `long_view` far more weakly (|r| ≤ 0.17) at the same row.

**What it does NOT establish:** Anything about these signals' value as **historical** (lagged)
features or **auxiliary training targets** — both remain legitimate uses per RULES.md §4-5 and are
untested by any of the three audits.

**Source provenance:** Claude D02, Gemini E01, GPT E02.

### 7.3 Inter-feedback correlation structure

Claude's D03 identifies two structurally distinct signal families: a tight "watch-related" cluster
(`is_click`–`play_time_ms`, validation r=0.5167 — corrected on Claude's own review from an earlier,
incorrect "≈0.60" transcription) and a much sparser, weakly-correlated "active-engagement" cluster
(`is_like`/`is_follow`/`is_comment`/`is_forward`; notable pair `is_comment`–`comment_stay_time`
r=0.3029). GPT reports comparable individual correlation pairs (e.g. click–raw-play-time train
r not identically stated as a single pair value, but its full log-continuous correlation matrix is
consistent with Claude's two-cluster description) without adopting the explicit two-cluster
framing. Gemini did not compute an inter-feedback correlation matrix.

**Evidence classification:** `HARD FACT` for the individual correlation values; the two-cluster
interpretation is a structural observation, not itself a new measured quantity.

**What it does NOT establish:** Which, if any, signal would help as a multi-task auxiliary target
for `long_view` specifically — that requires held-out multi-task experiments, not correlation.

**Source provenance:** Claude D03 (primary), GPT E02 (corroborating detail). Gemini: not done.

---

## 8. Historical Information Availability

### 8.1 Overall prior-interaction coverage (population: validation users, counting their own TRAIN-side rows)

| Measurement | Claude | Gemini | GPT | Agreement |
|---|---:|---:|---:|---|
| ≥1 prior train interaction | 98.11% | 98% (98.11% in data profile) | 98.114% | exact match, 3/3 |
| ≥5 | 92.85% | 92.85% | 92.854% | exact match, 3/3 |
| ≥10 | 85.17% | 85.17% | 85.168% | exact match, 3/3 |
| Median prior interactions | 35 | 35 | 35 | exact match, 3/3 |
| Mean prior interactions | 47.4 | — | 47.42 | match |
| p90 prior interactions | 103 | — | 103 | match |
| p99 / max | — | — | 216 / 809 | GPT only |

**Evidence classification:** `HARD FACT`.

Because official train dates strictly precede validation dates, every train impression of a user
is by construction strictly prior to every validation impression of that same user — this specific
measurement needs no per-row timestamp comparison across the split boundary.

### 8.2 A distinct, easily-conflated population: train-side activity across ALL train users

Separately from §8.1's population (validation users' own train-history counts), Gemini's own
review process caught and corrected a real error: an earlier draft of Gemini's data profile stated
train-impressions-per-user median=35 / p99≈250 / max>300, copying §8.1's "median 35" statistic into
a table that was supposed to describe a *different* population — plain train-side activity across
all 26,210 train users (not restricted to users who also appear in validation). Direct
recomputation gives:

| Measurement | Value |
|---|---:|
| Train impressions per (any) train user — min | 1 |
| — median | 31 |
| — mean | 43.54 |
| — p90 | 97 |
| — p99 | 207 |
| — max | 809 |

Validation impressions per validation user (a third, again-different population — each user's
own validation-side row count): min 1, median 4, mean 5.58, p90 12, p99 26, max 74 (all three
audits agree on this population, since it is identical to §4.2's list-length distribution).

**This is deliberately preserved here as a worked example of exactly the kind of population
conflation the source-priority rules for this merge warn about** — three quantities ("median
prior interactions for validation users" = 35; "median train activity for all train users" = 31;
"median validation list length" = 4) are all legitimate but describe different populations and
must not be used interchangeably.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Gemini data_profile §7 (review-corrected) / Gemini_REVIEW_REPORT.md Issue
R-01. Not reported as a standalone figure by Claude or GPT.

### 8.3 Historical availability by activity tier

Reported under each audit's own tier scheme (§4.3) — not directly comparable across audits for the
same reason as §4.3, but all three show the same qualitative pattern: Cold users have zero history
by construction, and coverage rises to ~100% at ≥10 prior interactions well before the top tier.

| Tier scheme | ≥1 | ≥5 | ≥10 | Source |
|---|---|---|---|---|
| Claude (17/36/65 edges): Cold/T1/T2/T3/T4 | 0% / 100% / 100% / 100% / 100% | 0% / 79.4% / 100% / 100% / 100% | 0% / 49.3% / 100% / 100% / 100% | Claude D05 |
| GPT (13/31/59 edges): Cold/T1/T2/T3/T4 | 0% / 100% / 100% / 100% / 100% | 0% / 72.95% / 100% / 100% / 100% | 0% / 33.42% / 100% / 100% / 100% | GPT F01 |
| Gemini (<10/10-49/50-149/150+): Cold/T1/T2/T3/T4 | 0% / 100% / 100% / 100% / 100% | 0% / 59.37% / 100% / 100% / 100% | 0% / 0% / 100% / 100% / 100% | Gemini F01 |

**Source provenance:** Claude D05, Gemini F01, GPT F01.

### 8.4 Prior feedback-signal availability (GPT-exclusive detail)

GPT additionally measured prior availability separately for each feedback type (not just raw
interaction counts): users with ≥1/≥5/≥10 prior clicks (96.157%/82.531%/66.309%), likes
(23.229%/4.683%/2.239%), comments (7.785%/0.241%/0.018%), follows (3.423%/0.054%/0.022%), forwards
(3.365%/0.049%/0.013%), hates (1.028%/0.063%/0.031%), positive-play-time rows
(97.640%/91.053%/82.111%). Neither Claude nor Gemini broke history availability down by feedback
type at this granularity.

**Evidence classification:** `HARD FACT`.

**Interpretation:** General and click/play-time histories are broadly available; rare-action
histories (likes, and especially follow/comment/forward/hate) are not — fewer than 2.3% of
validation users have even 10 prior likes, and fewer than 0.04% have 10 prior instances of any of
the four rarest actions.

**Source provenance:** GPT F01 only.

### 8.5 Row-level history: same video / author / tag already seen

See §3.5 for the merged figures (Claude/GPT match closely on video/author; the tag figure is
scope-dependent per §3.2).

### 8.6 Combined interpretation

**Interpretation:** The large majority of validation traffic has substantial train-side interaction
history available (85% of users at the ≥10 threshold; §8.1), but this history is dominated by
distinct videos/authors rather than repeats (§3.5) — algorithms relying on exact item-ID matching
(e.g. DIN with strict item matching) may struggle without content/tag-level matching, whereas
tag-level or aggregate historical features (e.g. prior click rate) have much broader row coverage.
Rare-action histories (§8.4) are too sparse to support a dense per-user historical feature for
anything beyond clicks/play-time/interaction-count.

**What it does NOT establish:** That any specific historical-feature construction or sequence
model (DIN, SIM, etc.) would improve validation primary score — only that the raw data support for
such approaches exists at the coverage levels measured above. The official KuaiRand documentation
(cited by GPT F01) itself cautions that Pure has incomplete sequential logs relative to the larger
27K/1K variants.

**Source provenance:** Claude D04/D05, Gemini F01, GPT F01.

---

## 9. Video Basic / Statistic Features

### 9.1 Video basic feature inventory

`video_type` (3 values, ~99% one value "NORMAL"); `upload_type` (14 values; top: LongImport 38.6%,
Web 31.9%); `visible_status` (constant, see §2.3); `music_type` (5-6 nonmissing values, one
dominant); `tag` (110 nonmissing strings / 46 parsed tokens, see §2.1/§3.2); `music_id` (7,202
values, ≈1 per video); `video_duration` (median ≈81,171ms, p90 ≈237,830ms, 3.15% missing).

**Evidence classification:** `HARD FACT`.

**Source provenance:** Claude E01, Gemini (via C01 missingness + A01), GPT A01/G01.

### 9.2 `tab` and `dur_bucket` marginal long_view-rate spread (Claude-only direct measurement)

In TRAIN, `tab` (15 categories, highly imbalanced) ranges from 0.42% long_view rate (tab 3, n=3,574)
to 61.25% (tab 10, n=80), with the two dominant tabs at 4.22% (tab 0, n=150,013) and 38.61% (tab 1,
n=834,876) — a very large spread across a highly imbalanced categorical. `dur_bucket` (10 quantile
buckets) shows a much milder spread, 0.273 to 0.376.

**Evidence classification:** `HARD FACT`.

**Interpretation:** This is direct, independent corroboration — from a completely different
analytical angle (marginal label rate vs. controlled ablation) — of §6.2's field-ablation finding
that `tab` is load-bearing and `dur_bucket` contributes much less.

**Source provenance:** Claude E01 only. GPT's §6.2 field-ablation result is complementary
corroboration, not a duplicate measurement of this specific marginal-spread statistic. Gemini did
not measure this.

### 9.3 `video_features_statistic_pure.csv` — aggregation window and causal safety

**This is the strongest three-way convergent finding in this pre-audit set: three independent
methods, all reaching the same conclusion.**

- **Claude (arithmetic evidence):** `show_cnt × counts` is near-integer for 100% of videos
  (supporting the hypothesis that these fields are per-`counts`-unit averages). Comparing the
  reconstructed total (`show_cnt × counts`) against actual observed impressions for that video in
  train+valid standard logs: median ratio 11,465×, p10 5,248×, p90 38,199×, and 0% of videos have a
  reconstructed total smaller than observed. `counts` itself ranges 45-181 (median 147) — already
  longer than the dataset's 31-day span.
- **GPT (documentation evidence):** The official field documentation states these statistics are
  "average per day and scenario... over one month," but does not disclose the exact calendar
  window or cutoff relative to any given impression.
- **Gemini (independent qualitative flag):** Notes the statistics "represent global aggregates
  (likely over a long or future time window)" and that using them as raw counts requires caution
  regarding temporal leakage, without Claude's quantitative ratio evidence or GPT's documentation
  citation.

Read together, GPT's "per day, per scenario, over one month" documentation and Claude's finding
that `counts` (45-181) already exceeds a single dataset-day span are mutually consistent — both
point toward `counts` spanning multiple days and/or scenario-tabs, over a population and time
window that is not identified anywhere in available materials, and is far larger than this
dataset's own sampled train+validation traffic.

**Evidence classification:** `HARD FACT` for the numeric ratio and the documentation text.
`INCONCLUSIVE` for whether the aggregation window overlaps the evaluation period, i.e. for the
causal/leakage safety of any feature built from this file.

**What it does NOT establish:** That the file is definitely leaky, or definitely safe — only that
safety is undocumented and should not be assumed merely because the file ships with the official
dataset.

**Source provenance:** Claude E01 (quantitative ratio), GPT G02 (documentation citation), Gemini
G01 (independent qualitative caution) — three-way convergence via complementary evidence types.

### 9.4 Video-stat ratio features vs. `long_view`

Two different methods, same qualitative ranking of which ratio is strongest:

- **Claude (E02, correlation):** Pearson r with `long_view` (validation): long_time_play_cnt ratio
  0.302 (strongest), play_cnt ratio 0.185, complete_play_cnt ratio 0.181, like_cnt ratio 0.040
  (weakest, non-monotonic — rises through Q4 then dips slightly in Q5).
- **GPT (G02, standalone-ranker test):** The same ratios scored directly as rankers and compared to
  a train-derived smoothed item-popularity baseline (primary 0.580722): long_time_play/show ratio
  primary 0.580378 (essentially tied with item-popularity, Δ−0.000344, Spearman 0.7167 with train
  item rate — strongest of the tested ratios), valid_play/show 0.570874, complete_play/show
  0.550128, play/show 0.540600, like/comment/follow/share ratios all weaker (0.44-0.48).

**Evidence classification:** `HARD FACT` for the association/ranking-score numbers (both methods).
`WEAK NEGATIVE EVIDENCE` (GPT) for these exact standalone ratio formulations as rankers — none
clearly beat simple item-popularity. Both methods carry §9.3's causal-safety caveat.

**Interpretation:** The long-time-play-based ratio is consistently the strongest of the tested
ratios under both methods. Neither method establishes incremental value over the existing FM
baseline (which already includes `video_id` and can in principle memorize per-video effects
directly), only marginal association/standalone-ranking value.

**Source provenance:** Claude E02, GPT G02. Gemini did not perform either kind of ratio analysis.

### 9.5 Video-stat raw means, incl. a preserved reviewer correction (Gemini-only)

| Statistic | Missing % | Mean | Median |
|---|---:|---:|---:|
| `show_cnt` | 0% | 10,552 | 4,519 |
| `play_cnt` | 0% | 7,747 | 2,560 |
| `long_time_play_cnt` | 0% | 3,687 | 978 |
| `like_cnt` | 0% | **230.75** | 57.54 |
| `comment_cnt` | 0% | **12.93** | 2.46 |
| `follow_cnt` | 0% | 17.41 | 3.80 |

Gemini's own review process caught and fixed two errors here: an earlier draft claimed
`like_cnt` mean = 158 (actual, recomputed from the raw file: 230.75), and left `long_time_play_cnt`
and `comment_cnt` as unmeasured placeholders (now filled in above). This is preserved as another
worked example of a reviewer correction that must not regress.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Gemini G01 (review-corrected). Not reported by Claude (which focuses on the
integer-ratio/aggregation-window question, §9.3, rather than raw means) or GPT (which reports
pairwise redundancy, §9.6, rather than raw means).

### 9.6 Video-stat pairwise redundancy and duration cross-check (GPT-only)

Both video feature files have 7,583 rows with 100% coverage of train/validation videos.
`duration_ms` (from interaction logs) exactly equals basic `video_duration` on 100% of nonmissing
joined interaction rows (Spearman 1.0) — interaction-log duration is a fully-covered fallback for
the 3.15%-missing basic `video_duration`. Among the numeric video statistics, 54 field pairs have
|Spearman| ≥ 0.95, e.g. like_cnt vs. like_user_num (0.999865), follow_cnt vs. follow_user_num
(0.999754), long_time_play_cnt vs. long_time_play_user_num (0.999678), valid_play_cnt vs.
valid_play_user_num (0.999499), play_cnt vs. play_user_num (0.999000).

**Evidence classification:** `HARD FACT`.

**Interpretation:** The video-statistic source is complete but contains extensive exact and
near-exact redundancy (count vs. user-count pairs); correlation this strong does not by itself
prove a field is useless inside a nonlinear or regularized model, but provides a defensible basis
for avoiding naively duplicated raw fields.

**Source provenance:** GPT G01 only. Not measured by Claude or Gemini.

---

## 10. Temporal Structure

### 10.1 Daily volume: the 2022-04-08 anomaly and the peak/trough decay

`log_standard_4_08_to_4_21_pure.csv` (the official train file) has **zero rows dated 2022-04-08**,
despite that being the official train start date — the earliest date actually present is
2022-04-09 (13 distinct dates, not 14). Total train row count (1,141,112) is unaffected; this is a
source-file coverage quirk, not a missing-data problem to fix.

Daily row counts within train (all three audits independently reproduce the identical series):

```text
04-09: 52,736   04-13: 94,711   04-17: 44,023   04-21: 20,021
04-10: 227,808  04-14: 71,252   04-18: 24,560
04-11: 278,835  04-15: 58,892   04-19: 20,443
04-12: 166,076  04-16: 60,904   04-20: 20,851
```

Peak (04-11, 278,835 rows) is ~13.9x the last train day (04-21, 20,021 rows) — a steep,
roughly-monotonic decline after the peak, continuing smoothly into validation (mean ≈17,844
rows/day across 04-22..04-28).

**Evidence classification:** `HARD FACT`.

**Interpretation:** The dataset is not temporally stationary — row volume decays by more than an
order of magnitude from its early-train peak to the end of train, and continues declining into
validation at a broadly similar rate (i.e. the train→validation volume transition looks like a
continuation of the late-train trend, not a regime break). `long_view` rate itself stays roughly in
the 0.31-0.38 range across the same period with no comparable order-of-magnitude swing (see the
per-day table in §10.2/data_profile) — the volume drift is a traffic/exposure-count phenomenon, not
obviously a label-rate drift on its own.

**What it does NOT establish:** The *cause* of the early-train spike (logging artifact,
promotional event, cohort-onboarding wave — not recoverable from this data alone), nor that
recency weighting would help the scored metric.

**Source provenance:** Claude A06, Gemini H01 (review-added), GPT H01 — exact triplicate match on
the daily series and ratio, all three independently reproduced.

### 10.2 Does validation resemble early- or late-train more?

Two complementary methods, both concluding the answer is genuinely mixed and no single "resembles
X more" claim is supportable:

- **Claude (A06):** Jaccard overlap of video/user sets: early-late 0.869/0.885, early-valid
  0.787/0.808, late-valid 0.818/0.809 (validation's entity-set structure resembles the *tail* of
  train slightly more). `long_view`-rate gap to validation: early train 0.0190, late train 0.0388
  (validation's raw label rate is numerically *closer* to *early*-train's rate — the opposite
  direction).
- **GPT (H01):** Period means — early train (04-08..14) 148,570 rows/day, rate 0.33228; late train
  (04-15..21) 35,671 rows/day, rate 0.35211; validation 17,844 rows/day, rate 0.31328. Validation is
  closer to early train on long_view rate (gap 0.01900 vs. 0.03882) and mean duration (4,267ms vs.
  7,343ms gap), but closer to late train on `tab` distribution (Jensen-Shannon divergence 0.00252
  vs. 0.00392) and on volume/user/video counts.

**Evidence classification:** `INCONCLUSIVE` for any single combined "validation resembles period X
more" claim — both audits explicitly and independently decline to assert this. The individual
period-level measurements underlying it are `HARD FACT`.

**Interpretation:** Temporal drift is real (§10.1) but multidimensional: recency is not a
uniformly better distributional match across every measured axis. No recency-weighting conclusion
should be drawn from these facts alone.

**Source provenance:** Claude A06, GPT H01. Gemini did not perform a comparable early/late-train
vs. validation comparison beyond the daily-series correction described in §10.1.

---

## 11. Random-Exposure Audit

### 11.1 Date coverage and row counts (`log_random_4_22_to_5_08_pure.csv`)

| Measurement | Value | Agreement |
|---|---:|---|
| Total rows | 1,186,059 | exact match, 3/3 |
| Date range | 2022-04-22 .. 2022-05-08 (17 dates) | exact match, 3/3 |
| Rows in evaluation-date range (04-29..05-08) | 897,721 (75.69%) | exact match, 3/3 (date-only count in every audit — no evaluation-period label or feedback value was ever read from this file by any of the three audits) |
| Rows in validation-date range (04-22..04-28) | 288,338 | exact match, 3/3 |
| Rows in train-date range | 0 (the file's date range does not overlap train at all) | Gemini I01 explicit; consistent with all three |

**Evidence classification:** `HARD FACT`.

**Interpretation (Gemini):** Because the random log's date range does not overlap the train window
at all, using *any* row from this file as training data — not only the test-window-dated rows —
would mean training on information collected strictly after the train cutoff, breaking temporal
ordering. Per DATA_GUIDE.md §7, a defensible use of the validation-dated slice is as a
validation-time-window diagnostic/counterfactual set, not as additional training signal.

**Evidence classification for training use:** `STRONG NEGATIVE EVIDENCE` against using this file,
unfiltered, as training data.

**Source provenance:** Claude F01, Gemini I01, GPT I01.

### 11.2 Entity and pair overlap with standard logs — two genuinely different comparisons

**Full random log (minus test-window rows) vs. standard train+validation** — Gemini and GPT compute
the same comparison and reach the same corrected number:

| Measurement | Gemini (review-corrected) | GPT |
|---|---:|---:|
| Unique (user, video) pairs, full random log | — | 1,186,006 |
| Overlap with standard train+validation pairs | 702 / 1,186,006 (0.06%) | 702 / 1,186,006 (0.0592%) |
| Random users also in standard train+valid | — | 97.607% |
| Random videos also in standard train+valid | — | 99.499% |

Gemini's own review process caught and fixed this exact figure: the original computation (759
pairs) had been built from the *unfiltered* second standard log, which spans validation+test dates
— a rule-compliance bug (structural ID overlap only; no test label was ever read). The corrected
figure (702) is **independently cross-validated by GPT's separately-coded computation**, which
reaches exactly the same number.

**Random-log VALIDATION-PERIOD SLICE vs. standard-VALIDATION-ONLY** — a narrower, different
comparison, computed only by Claude:

| Measurement | Value |
|---|---:|
| Random log rows in validation-date range | 288,338 |
| Shared (user, video) pairs vs. standard validation | 17 / 288,328 (0.006%) |
| Validation-period random-log users also in train+standard-valid | 98.89% |
| Random-log videos also in standard logs | 99.50% |

**This 17-pair figure is not the same quantity as the 702-pair figure above** — different
comparison sets (validation-slice-only vs. full-log-minus-test) — and must not be conflated.

**Evidence classification:** `HARD FACT` for all figures under their own stated scope.

**Source provenance:** Gemini I01 (review-corrected 759→702), GPT I01 (independently matches at
702), Claude F01 (distinct 17-pair validation-slice-only figure).

### 11.3 Random-log validation-period `long_view` rate (Claude-only)

Claude additionally computed, from the validation-period slice only (a rule-compliant, reviewer-
verified use of validation-period labels), a `long_view` rate of 8.06% in the random log vs. 31.3%
in the standard validation log — roughly 4x lower, consistent with strong exposure/selection
differences between randomized and standard (algorithmically-ranked) exposure. GPT deliberately
chose not to load any feedback/label column from this file at all ("labels_loaded: false" in its
data profile), a more conservative methodological choice; Gemini's I01 does not report a random-log
`long_view` rate either.

**Evidence classification:** `HARD FACT`.

**What it does NOT establish:** The causal source or magnitude of the exposure bias, any specific
safe use (unbiased evaluation set, IPS-correction source, auxiliary training signal), or whether
this comparison generalizes beyond the seed-0 baseline's standard-validation rate.

**Source provenance:** Claude F01 only.

---

## 12. Engineering Feasibility

> All timing figures below are `ENGINEERING CONSTRAINT` facts specific to each audit's own
> machine/run, not a property of the dataset. They are reported separately per audit rather than
> merged into one number.

### 12.1 Cold pipeline runtime

| Stage | Claude (review-rerun) | GPT |
|---|---:|---:|
| CSV load (all files) | 2.99s | 2.88s |
| Encoding | 4.81s | 8.47s |
| FM training (+ epoch evals) | 49.7s (11 epochs) | 66.60s (11 epochs) |
| Final evaluation | 0.079s | 0.52s |
| **Cold total** | **~57.5s** | **78.52s** |
| Peak RSS | ~491MB | ~1.41GB |

Gemini reports only a qualitative "~40-50s" order-of-magnitude estimate, not an independently
timed run in its own audit.

**Interpretation:** At either measured rate, 50 iterations of a comparably-sized model costs on
the order of 40-65 minutes of pure training time — comfortably inside the 6-hour budget, leaving
room for larger models or repeated-seed evaluation per iteration.

**Source provenance:** Claude G01, Gemini (qualitative only), GPT J01.

### 12.2 Cache correctness and speedup

| Comparison | Claude | GPT |
|---|---:|---|
| Cache read time | 0.018s | 0.043s |
| Baseline being compared against | full **re-encoding** (4.81s) | raw **CSV load** (3.139s) |
| Resulting speedup | ~263x | ~72.8x |
| Fingerprint-inclusive "effective" speedup | not computed | ~2.27x (full content fingerprint costs 1.341s; fingerprint+read totals 1.384s) |
| Correctness check | bit-identical reload | bit-identical hashes; changed fingerprint correctly rejected |

These are not conflicting — they compare cache-read against two different baselines. GPT's
"effective" 2.27x figure, which includes the cost of computing a safe content fingerprint before
trusting the cache, is arguably the more realistic number for a harness that must guard against
silently serving a stale cache.

**Evidence classification:** `ENGINEERING CONSTRAINT` / `HARD FACT` for the specific run each
figure comes from.

**Source provenance:** Claude G02, GPT J01. Gemini: not measured.

### 12.3 Windows subprocess/process-tree robustness — a risk and its confirmed mitigation

Claude's G03 found a **failure mode**: bare `subprocess.run(timeout=3, capture_output=True)` did
NOT return within 3 seconds when the tracked child process itself spawned an unmanaged grandchild
that inherited stdio pipe handles — it blocked for the grandchild's full 30.13-second runtime, most
consistent with `communicate()` waiting on the inherited pipe to reach EOF. GPT's J02 separately
tested a simpler condition (no unmanaged grandchild) and found that a bare timeout (0.3s configured,
0.313s actual) and, more relevantly, `psutil`-based **recursive process-tree termination** both
worked correctly, leaving neither parent nor child alive.

**These are complementary, not contradictory findings.** GPT's passing psutil-based recursive-
termination test is a plausible working mitigation for exactly the failure mode Claude's more
adversarial test exposed; neither audit tested the other's exact scenario.

**Evidence classification:** `ENGINEERING CONSTRAINT` — ` HARD FACT` for each tested condition;
`INCONCLUSIVE` whether psutil-based termination specifically fixes Claude's exact grandchild-
inheritance scenario, since GPT did not test that scenario.

**Interpretation:** A harness enforcing per-iteration or 6-hour wall-clock budgets on this Windows
environment should not rely on a bare `subprocess.run(timeout=N)` call whenever the launched
process might itself spawn further children (e.g. a data-loader worker) — `psutil`-based recursive
process-tree enumeration and termination, or `CREATE_NEW_PROCESS_GROUP` + `taskkill /T /F`, is the
candidate fix, confirmed to work under GPT's (simpler) test condition but not yet confirmed under
Claude's specific adversarial condition.

**Source provenance:** Claude G03 (risk), GPT J02 (partial mitigation evidence). Gemini: not
tested.

### 12.4 NaN/Inf and syntax-error recovery

Both Claude and GPT independently verify that NaN/Inf in a submission is cleanly rejected by the
unmodified official `submit.py` (Claude) / `np.isfinite` (GPT), and that a syntax error in a
launched child script produces a clean nonzero return code with `"SyntaxError"` in stderr.

**Evidence classification:** `HARD FACT`.

**Source provenance:** Claude G04, GPT J02. Gemini: not tested.

### 12.5 Engineering-readiness: harness/pipeline/agent orchestration layer (GPT-only)

GPT's J02 (reviewer-corrected in scope from 7 to 15 files) found that, as of the pre-audit, **all
15 files** under `harness/` (`executor.py`, `guards.py`, `cache.py`, `diagnostics.py`, `logger.py`,
`score.py`, `submission.py`), `pipeline/` (`data_adapter.py`, `features.py`, `train.py`), and
`agent/` (`coder.py`, `controller.py`, `governor.py`, `proposer.py`, `reflector.py`) contain zero
executable non-comment lines; `reports/`, `submissions/`, `runlogs/`, and `tests/` contain no
files. The reviewer specifically caught that the original probe covered only 7 of these 15 files
and expanded the check to the full repository.

**Evidence classification:** `HARD FACT`.

**Interpretation:** The runtime primitives needed to build this layer (bounded subprocesses,
recursive Windows process-tree termination, syntax-error recovery, NaN/Inf detection,
content-fingerprinted caching — §12.2-12.4) were separately probed and function correctly, but the
entire agent-orchestration layer described in `PROBLEM.md` §1 (the propose/code/reflect/govern
research loop) is unimplemented, not only feature-engineering plumbing.

**What it does NOT establish:** The design of the eventual harness/agent implementation, or
whether this gap should be closed before `RUN_START` or incrementally during the run.

**Source provenance:** GPT J02 only (with its own reviewer-corrected file count). Not assessed by
Claude or Gemini.

---

## 13. Model / Objective Evidence

This section synthesizes what §6 and §9 already established about model formulation itself, at a
level above individual field-by-field ablations.

- **Objective/metric mismatch (HARD FACT, GPT C01, §6.1):** the baseline's training objective is
  pointwise binary cross-entropy on `long_view`, while both scored metrics are within-user ranking
  metrics. No audit tested whether a pairwise (BPR) or listwise loss closes any of the gaps
  identified in §4-§5; this remains untested by all three audits.
- **Simple capacity scaling is a dead end for this FM (STRONG NEGATIVE EVIDENCE, §6.6):**
  independently reproduced by two audits across k=8/16/32(/64); consistent with the organizer's own
  prior evidence (`constraints.md` C6).
- **The exact dual `video_id`+`author_id` field combination is empirically redundant in this exact
  pointwise FM (WEAK NEGATIVE EVIDENCE, §6.2):** independently reproduced by two audits across 10
  combined paired-seed runs; narrowly scoped to this exact model/field-set/objective and explicitly
  not shown to generalize to other model families.
- **`tab` is the single most load-bearing official field (STRONG NEGATIVE EVIDENCE against its
  removal, §6.2, corroborated by an independent marginal-rate analysis in §9.2).**
- **Static feature stuffing (the organizer's own tested CWM-style expansion) is a dead end for this
  FM (STRONG NEGATIVE EVIDENCE, §6.5):** independently reproduced by two audits on validation,
  strengthening rather than merely repeating `constraints.md` C5.
- Neither multi-task auxiliary-target architectures, historical-feature/sequence models, nor
  pairwise/listwise losses were tested by any of the three audits — see §14's "Dataset Opportunities
  Not Yet Tested" and each source audit's own "Questions the Autonomous Agent Should Resolve
  Itself" list.

`references.md` catalogs the available model/objective families (FM variants, BPR, listwise
losses, LambdaRank, DeepFM/DCN/xDeepFM, DIN, sequential models, multi-task architectures) without
ranking them; nothing in this merged pre-audit narrows that catalog beyond the specific negative
results listed above.

---

## 14. Evidence Summary

### Hard Facts

1. The validation-only official FM reproduces closely across three independent implementations:
   GAUC ≈0.6671-0.667133, nDCG@5 ≈0.5358-0.535806, primary ≈0.6015-0.601470 (seed 0), all within
   the published seed std of 0.0008 of the published 0.6674/0.5357/0.6016. (§1)
2. Entity overlap train→validation is high (>98% users/videos/authors); exact `(user,video)`/
   `(user,author)` pair overlap is low (1.6%/3.4%); `(user,tag)` pair overlap is much higher and
   scope-dependent (68.14% raw-string / 71.913% parsed-token). (§3.1-3.2)
3. 86.96%-87.07% of authors have exactly one video, depending on scope (full table vs.
   observed-only). (§3.3)
4. Repeat-pair affinity within train is sparse at video/author granularity (4.1%/5.9% of pairs) and
   dense at tag granularity (51.8%-55.3%, scope-dependent). (§3.4)
5. 42.2% of validation users (30.32% all-negative + 11.90% all-positive) have a fixed nDCG@5
   regardless of ranking and are excluded from GAUC; 57.78% are movable. (§4.1)
6. The 6-10 validation list-length bucket carries the single largest official GAUC weight share
   (36.39%). Activity-tier GAUC-weight concentration is confirmed by three audits using three
   different (non-comparable) tier schemes, all showing concentration in the highest tier. (§4.2-4.3)
7. Activity and validation list length are moderately positively associated (Spearman ρ≈0.46); their
   joint intersection (T3/T4 × 6+ lists, Claude's scheme) contains 25.38% of users but 50.79% of
   GAUC weight and 51.72% of the current baseline-to-oracle primary gap. (§5.1, Claude-only)
8. `user_id` and `tab` are load-bearing for the official FM (large, unambiguous drops on removal);
   `dur_bucket`'s contribution is small; removing `video_id` or `author_id` individually did not
   hurt and modestly improved primary across 10 combined paired-seed runs from two independent
   implementations. (§6.2)
9. Static feature stuffing (organizer's CWM-style expansion) and simple FM embedding-dimension
   scaling are both independently reproduced dead ends on validation by two audits, strengthening
   `constraints.md` C5/C6. (§6.5-6.6)
10. `is_click` (r≈0.75-0.76) and `play_time_ms` (r≈0.60-0.64) are strongly correlated with
    `long_view` at the same row — dataset-specific quantitative support for the existing
    same-row-leakage rule (constraints.md C3); all other feedback signals correlate at |r|≤0.17.
    (§7.2)
11. 85.17% of validation users have ≥10 prior train interactions (median 35 for that population);
    a *different* population (all 26,210 train users) has median train activity of 31 — these must
    not be conflated (§8.1-8.2, a preserved reviewer correction).
12. `video_features_statistic_pure.csv`'s aggregation window is undocumented and, by three
    independent lines of evidence (arithmetic ratio, official documentation text, independent
    qualitative flag), appears far larger than this dataset's own sampled traffic; causal/leakage
    safety for the scored period is unresolved. (§9.3)
13. `log_standard_4_08_to_4_21_pure.csv` has zero rows on the nominal train start date (2022-04-08);
    daily volume decays ~13.9x from its 04-11 peak to the end of train and continues declining
    smoothly into validation. (§10.1)
14. The random-exposure log's date range does not overlap train at all; 75.69% of its rows fall in
    the forbidden evaluation-date range (date-only counts only, in all three audits); its full-log
    (minus test) pair overlap with standard train+validation is 702/1,186,006 (0.06%),
    independently cross-validated by two audits. (§11.1-11.2)
15. All 15 files under `harness/`, `pipeline/`, and `agent/` are comment-only scaffolds with zero
    executable lines; the agent-orchestration layer described in PROBLEM.md is not yet implemented.
    (§12.5, GPT-only)

### Strong Negative Evidence

1. Retaining both `video_id` and `author_id` in the exact 5-field pointwise FM underperformed
   removing either one, across 10 combined paired-seed runs from two independent implementations.
   (§6.2) — narrowly scoped to this exact formulation.
2. Simple FM embedding-dimension scaling (k=8/16/32/64) gives no meaningful mean benefit on
   validation, independently confirmed by two audits and consistent with organizer evidence
   (constraints.md C6). (§6.6)
3. The organizer's exact static-feature expansion (8-field item-only: inconclusive; 13-field
   CWM-style: clearly negative, all seeds lower) does not help on validation, independently
   confirmed by two audits and consistent with organizer evidence (constraints.md C5). (§6.5)
4. Removing `tab` from the exact FM causes a large, unambiguous drop (~0.016 primary), confirmed by
   two audits. (§6.2)
5. Using the random-exposure log, unfiltered, as training data is unsafe: its date range does not
   overlap train at all, so any row from it postdates the train cutoff by construction. (§11.1)

### Weak Negative Evidence

1. Learning rates ≥0.003 show degradation relative to the official lr=0.001, though with
   substantial run-to-run noise at 0.003 specifically. (§6.4)
2. Standalone smoothed video-statistic ratios, scored directly as rankers, do not clearly beat
   simple item-popularity (GPT); the strongest of the tested ratios (long-time-play-based) is
   essentially tied with item-popularity. (§9.4)
3. Item-ID-based attention (e.g. strict-matching DIN) may struggle given the <2% exact item/author
   repeat rate, unless attention is computed over broader attributes (tags/categories). (Gemini,
   qualitative)

### Inconclusive

1. Whether validation "resembles late training more than early training" — volume/entity-overlap
   structure and `tab` distribution point one way; raw `long_view` rate points the other way. No
   combined claim is supportable from current evidence. (§10.2)
2. Whether video-statistic features can be used safely given the undocumented aggregation window.
   (§9.3)
3. `dur_bucket`'s individual field-ablation contribution (too small relative to seed noise at 3
   seeds). (§6.2)
4. Learning rate 0.0003-0.0005 vs. 0.001, and 0.002 vs. 0.001 (both audits' nearby-lr deltas
   straddle their own noise). (§6.4)
5. Whether `psutil`-based recursive process-tree termination specifically fixes the exact
   grandchild-inheritance timeout failure Claude found (GPT tested a different, non-adversarial
   condition). (§12.3)
6. Whether the random-exposure log's validation-period slice would make a useful unbiased
   diagnostic set — structurally plausible (near-zero pair overlap with standard traffic, much
   lower/different long_view rate) but not experimentally tested by any audit. (§11.3)

### Dataset Opportunities Not Yet Tested

(Availability statements only — not recommendations.)

- Historical/sequential user behavior as a feature or sequence-model input (substantial train-side
  history exists for most users; exact-item repeat is sparse, tag-level repeat is dense).
- Multi-task auxiliary targets using `is_click`/`play_time_ms` (dense, strongly same-row-correlated
  with `long_view`, but efficacy as an auxiliary head is untested) or the sparser engagement signals.
- Pairwise or listwise ranking losses, given the pointwise/ranking-metric mismatch noted in §13 and
  the metric-headroom concentration found in §4-§5.
- Video-statistic ratio features and aggregates, conditional on resolving the §9.3 causal-safety
  question.
- Coarser generalization keys (tag/author-level) as a way to work around the near-zero exact
  user-video pair repeat rate.
- Recency weighting or explicit temporal features, motivated by §10.1's volume decay but not yet
  motivated or ruled out on `long_view` rate itself.
- Random-exposure validation-period slice as a diagnostic/counterfactual evaluation set.

### Engineering Constraints

- Cold pipeline runtime is on the order of 1 minute per iteration on either measured environment
  (~57.5s / ~78.5s); comfortably inside the 6-hour / 50-iteration budget. (§12.1)
- Encoded-array caching gives a large, correctness-preserving speedup, with the exact factor
  depending on which baseline (re-encode vs. raw-load, with/without fingerprint overhead) it is
  measured against. (§12.2)
- Bare `subprocess.run(timeout=N)` can fail to bound wall-clock time on Windows if the tracked
  process spawns an unmanaged child; `psutil`-based recursive process-tree termination is a
  plausible, partially-confirmed mitigation. (§12.3)
- NaN/Inf and syntax-error failure modes are already cleanly detectable via existing official code
  and standard subprocess semantics. (§12.4)
- The entire harness/pipeline/agent orchestration layer (15 files) is unimplemented as of the
  pre-audit. (§12.5)

### Questions the Autonomous Agent Should Resolve Itself

- Which multi-task objective/architecture, if any, avoids negative transfer while using
  `is_click`/`play_time_ms` or other feedback as auxiliary targets.
- Which historical-feature representation (aggregate rate, sequence, tag-attention) provides the
  highest lift, and at what granularity (video/author/tag).
- Whether a pairwise or listwise loss outperforms pointwise logloss for this within-user ranking
  task, and whether it interacts with the metric-headroom concentration found in §4-5.
- Whether the §6.2 video/author-redundancy finding generalizes to any model family other than the
  exact tested pointwise FM.
- Whether and how to use `video_features_statistic_pure.csv` given its unresolved aggregation-window
  uncertainty — a risk/benefit judgment call this pre-audit deliberately leaves open.
- Whether temporal/recency weighting is worth pursuing given the volume decay in §10.1 and the
  mixed early/late-train resemblance in §10.2.
- Whether the random-exposure log's validation-period slice is worth incorporating as a diagnostic
  set, and how.
- What experiment has the highest expected information gain under the 50-iteration / 6-hour budget,
  given everything above.

---

## 15. Candidate Findings for Human Review

Do **not** edit `context/constraints.md` automatically. Presented for human review only; see
`research/consolidated/REVIEW_REPORT.md` §9 for the merge's own recommendation on each.

### Candidate 1 — Train-log date coverage gap and volume decay

**Finding:** `log_standard_4_08_to_4_21_pure.csv` has zero rows for 2022-04-08 (13 distinct dates,
not 14); daily row volume falls from its 04-11 peak (278,835 rows) to 20,021 rows on 04-21 (≈13.9x),
continuing to decline into validation.

**Evidence classification:** `HARD FACT`.

**Numerical evidence:** See §10.1. Independently reproduced by all three source audits, exact match.

**Confidence:** High.

**Recommended wording:** "`log_standard_4_08_to_4_21_pure.csv` contains no rows dated 2022-04-08
(13 represented train dates, not 14). Daily interaction volume falls from 278,835 rows at its
04-11 peak to 20,021 rows on 04-21 (≈13.9x), continuing to decline smoothly into validation. This
does not by itself establish that recency weighting would help the scored metric."

**Source provenance:** Claude A06, Gemini H01, GPT H01.

### Candidate 2 — Metric headroom concentration by activity tier and list length

**Finding:** Validation list-length bucket 6-10 carries the largest single official GAUC weight
share (36.39%, exact match between two independently-built implementations). Activity-tier
analysis, run under three different tier definitions, consistently shows GAUC weight and movable
nDCG headroom concentrated in the highest-activity tier. Claude's joint analysis further shows the
T3/T4 × 6+-list intersection (25.38% of users) contains roughly half (50.79%/51.72%) of total GAUC
weight and primary-score headroom.

**Evidence classification:** `HARD FACT`.

**Numerical evidence:** See §4.2, §4.3, §5.1.

**Confidence:** High for the list-length figure (exact cross-audit match) and the joint-analysis
figures (internally reconciled to 100% of users/rows/GAUC weight); moderate for any single
activity-tier percentage, since tier boundaries differ across audits and should not be quoted as
if interchangeable.

**Recommended wording:** "Under the reproduced official FM baseline, the validation list-length
bucket of 6-10 impressions carries the largest single share of official GAUC weight (36.39%).
Train-side activity concentrates GAUC weight and movable nDCG headroom in the highest-activity
tier under every tested tier definition, though the exact percentage depends on the tier
boundaries used. This is a diagnostic concentration under the current baseline, not evidence that
either dimension causes the gap or that a particular method will close it."

**Source provenance:** Claude B01/B02/B04/B05/B06, Gemini B01, GPT B02/B03.

### Candidate 3 — Video/author identity redundancy in the exact baseline FM

**Finding:** In the official 5-field pointwise FM (k=16, lr=0.001), individually removing
`video_id` or `author_id` did not hurt, and modestly improved, validation primary — reproduced
across 10 combined paired-seed runs from two independently-built codebases (all 10 positive).

**Evidence classification:** `WEAK NEGATIVE EVIDENCE`, narrowly scoped to this exact formulation.

**Numerical evidence:** Drop-video +0.00108 (Claude) / +0.001082±0.000585 (GPT); drop-author
+0.00132 (Claude) / +0.001316±0.000426 (GPT), all 5-seed, all positive.

**Confidence:** High for this exact tested formulation (two independent implementations converge);
explicitly not shown to generalize to other model families or losses.

**Recommended wording:** "In the official pointwise-logloss FM (k=16, lr=0.001), individually
removing `video_id` or `author_id` modestly increased validation primary across 10 combined
paired-seed runs from two independently-built codebases (all 10 positive; means +0.0011 to
+0.0013). This applies only to this exact field set/model/training protocol and does not establish
that video or author identity is generally uninformative."

**Source provenance:** Claude C01, GPT C02.

### Candidate 4 — `tab` field importance and static-feature/capacity dead ends (strengthens existing C5/C6)

**Finding:** `tab` removal materially damages the exact baseline FM. The organizer's static-feature
expansion (constraints.md C5) and FM-capacity-scaling (constraints.md C6) null results both
independently reproduce on the validation split by two separately-coded implementations, not
merely by re-quoting the organizer's original test-split numbers.

**Evidence classification:** `STRONG NEGATIVE EVIDENCE` (tab removal, static expansion, capacity
scaling — all against the tested alternative).

**Numerical evidence:** See §6.2, §6.5, §6.6.

**Confidence:** High — independent cross-validation between two codebases in all three cases.

**Recommended wording:** "`tab` removal costs ≈0.016 validation primary in the exact baseline FM
(two independent implementations agree to 3 decimal places). The organizer's existing static-
feature-expansion (C5) and FM-capacity-scaling (C6) findings are independently reproduced on the
validation split — not merely inherited from the organizer's test-split claim — by two separately
coded implementations, strengthening rather than merely repeating C5/C6."

**Source provenance:** Claude C01/C04/C05, GPT C02/D01/D02.

### Candidate 5 — Same-row feedback correlation with `long_view` (redundant with existing C3 — do not promote as new)

**Finding:** `is_click` (r≈0.75-0.76) and `play_time_ms` (r≈0.60-0.64) correlate very strongly with
`long_view` at the same row; all other feedback signals correlate at |r|≤0.17.

**Evidence classification:** `HARD FACT`.

**Numerical evidence:** See §7.2.

**Confidence:** High, triplicate cross-audit match.

**Assessment:** This is correct and independently triplicate-confirmed, but is dataset-specific
elaboration of a rule (`constraints.md` C3) that is already established. Claude's own review
report reached the same "do not promote, redundant with C3" conclusion for this exact finding; the
merge concurs (see REVIEW_REPORT §9).

**Source provenance:** Claude D02, Gemini E01, GPT E02.

### Candidate 6 — `video_features_statistic_pure.csv` aggregation-window uncertainty

**Finding:** The aggregation window/population for the video-statistics file is undocumented and,
by three independent lines of evidence, appears to be far larger than this dataset's own sampled
train+validation traffic; causal/leakage safety for the scored period is unresolved.

**Evidence classification:** `HARD FACT` (numeric ratio, documentation text) + `INCONCLUSIVE`
(causal safety).

**Numerical evidence:** See §9.3.

**Confidence:** High on the documentation gap and the numeric ratio itself; unresolved on
causal/leakage implications.

**Recommended wording:** "The official video-statistics file (`video_features_statistic_pure.csv`)
is documented as a per-day, per-scenario average computed over one month, but the exact calendar
window or cutoff relative to a given impression is not disclosed. `show_cnt × counts` is
near-integer for all videos, and the reconstructed total is on the order of 10,000x larger than
this dataset's own observed train+validation traffic for the same videos — consistent with, but not
proof of, a much larger external population/window. Causal/leakage safety for the April 22-28
validation period (or any scored period) is not established by local materials alone."

**Source provenance:** Claude E01, Gemini G01, GPT G02.

### Candidate 7 — Windows subprocess timeout robustness

**Finding:** Bare `subprocess.run(timeout=N)` failed to bound wall-clock time on this Windows
environment when the tracked process spawned an unmanaged grandchild inheriting stdio pipe
handles; `psutil`-based recursive process-tree termination is separately confirmed to work under a
simpler (non-adversarial) test condition.

**Evidence classification:** `ENGINEERING CONSTRAINT`.

**Numerical evidence:** See §12.3.

**Confidence:** High for the tested failure condition and the tested mitigation condition
separately; unresolved for whether the mitigation specifically fixes the failure condition, since
they were not tested together.

**Recommended wording:** "On this Windows environment, `subprocess.run(timeout=N)` does not
reliably bound wall-clock time if the tracked process spawns further unmanaged child processes
that inherit stdio handles (verified: a 3-second timeout took 30.13 seconds to return). Separately,
`psutil`-based recursive process-tree enumeration and termination was confirmed to work under a
simpler test condition. A harness enforcing per-iteration or total wall-clock budgets on Windows
should use tree-aware termination rather than a bare `subprocess.run(timeout=...)` call, and should
verify that mitigation against the specific inherited-pipe condition before relying on it."

**Source provenance:** Claude G03, GPT J02.

### Candidate 8 — Engineering readiness of the harness/pipeline/agent layer

**Finding:** All 15 files under `harness/`, `pipeline/`, and `agent/` are comment-only scaffolds
with zero executable lines; `reports/`, `submissions/`, `runlogs/`, `tests/` contain no files.

**Evidence classification:** `HARD FACT`.

**Numerical evidence:** See §12.5.

**Confidence:** High (reviewer-corrected and confirmed full-repository scope).

**Recommended wording:** "As of the pre-audit, all 15 files under `harness/`, `pipeline/`, and
`agent/` (execution, guards, caching, diagnostics, logging, scoring, submission, data adaptation,
feature construction, training, and the coder/controller/governor/proposer/reflector agent loop)
contain zero executable non-comment lines; `reports/`, `submissions/`, `runlogs/`, and `tests/`
contain no files. Runtime primitives needed to build this layer were separately probed and
function correctly (see Candidate 7 and §12.4)."

**Source provenance:** GPT J02 only.

### Candidate 9 — Population-conflation caution (historical availability vs. plain train activity)

**Finding:** "Median prior interactions = 35" (validation users' own train-side history count) and
"median train activity = 31" (all 26,210 train users) are two different, easily-conflated
populations; a draft version of one source audit's data profile originally conflated them and was
corrected on review.

**Evidence classification:** `HARD FACT`.

**Numerical evidence:** See §8.1-8.2.

**Confidence:** High.

**Recommended wording:** "'Median prior train interactions for validation users' (35) and 'median
train-side activity across all train users' (31) are different populations and should not be used
interchangeably; a downstream statistic or feature should specify which population it refers to."

**Source provenance:** Gemini data_profile §7 (review-corrected) / Gemini_REVIEW_REPORT.md Issue
R-01.
