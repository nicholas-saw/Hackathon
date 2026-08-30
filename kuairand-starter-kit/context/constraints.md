# KuaiRand-Pure — Established Evidence / Constraints

> Purpose: verified prior knowledge the autonomous research agent may rely on.
>
> This file states **what is true**, not what to try. Every entry says what its evidence
> establishes *and* what it does not establish, because the second half is what keeps a
> measurement from silently becoming a directive.
>
> **Scope of all local evidence: train + validation only.** No entry here rests on
> evaluation/test labels or on evaluation-period outcomes. Published organizer test
> numbers appear only where they are explicitly labelled as official reference material.
>
> Classification vocabulary is defined in `RULES.md` §8. Detailed derivations are in
> `research/PRE_AUDIT.md`; compact numbers are in `research/data_profile.md`.
>
> Constraint IDs C1–C6 retain the meanings they had before the audit update; C7 onward
> were added from the reviewed audit evidence. IDs are stable: a retired entry keeps its
> number and a marker saying where its evidence now lives, so that references from
> `PRE_AUDIT.md`, `REVIEW_REPORT.md`, and the other context files never silently
> re-point.
>
> Do not add to this file without human review. Do not put hypotheses or strategy here.

---

## 1. Official and Mathematical Facts

### C1 — Official target, task form, and metric semantics

**Classification:** HARD FACT

**Evidence:**
- Target `long_view` (binary); task is within-user ranking over logged impressions;
  `primary = (GAUC + nDCG@5) / 2`.
- `source/starter-kit/evaluate.py` is the scoring authority and defines: GAUC over
  users with `0 < positives < impressions`, weighted by each such user's positive
  count; nDCG@5 averaged over **all** users with equal weight; gain `2^rel − 1`; an
  empty GAUC denominator returns the 0.5 fallback.
- Official splits: train 2022-04-08..21, validation 2022-04-22..28, evaluation
  2022-04-29..05-08.

**Interpretation:**
Two metrics with two different user weightings are averaged into one number. A change
can move one and not the other, and the two populations they score are not the same set
of users.

**Does NOT establish:**
Nothing about which model, loss, or feature set scores well. It also does not license
any locally reimplemented or reweighted metric standing in for `evaluate.py`.

**Provenance:**
- Official Starter Kit (`evaluate.py`, `data.py`, `baseline_scores.json`, README)
- PRE_AUDIT §0; REVIEW_REPORT: official authority, no dispute

---

### C2 — Terms constant within a user cannot change that user's ranking

**Classification:** HARD FACT

**Evidence:**
- Mathematical consequence of ranking strictly within a user.
- Organizer-confirmed by measurement: the official README reports that
  `item_pop × user bias` and plain `item_pop` produce identical scores to the digit.

**Interpretation:**
A purely additive first-order user-side term contributes exactly zero to the metric.
User-side information can only reach the score through terms that vary within the
user — interactions with item/context features, shared representations, or
user-conditioned behaviour.

**Does NOT establish:**
That user features are useless, or that user-side modelling is a dead end. It
constrains the *form* in which user information can act, not its value.

**Provenance:**
- Official Starter Kit README; mathematical consequence of C1
- REVIEW_REPORT: no dispute

---

### C3 — Same-row post-impression feedback as an input

**Classification:** INVALID / FORBIDDEN

**Evidence:**
- Post-impression columns: `is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`, `is_hate`, `play_time_ms`, `profile_stay_time`, `comment_stay_time`,
  `is_profile_enter`.
- These are concurrent outcomes of the same impression as `long_view`. Measured
  same-row association is large for the dense ones — validation Pearson r with
  `long_view` is 0.7515 for `is_click` and 0.6319 (raw) for `play_time_ms` — which is
  exactly why using them as same-row inputs produces a meaningless score.

**Interpretation:**
Forbidden as an input to the same row's prediction, in any form, including hidden
inside a derived "engagement score". Permitted as (a) auxiliary targets, (b) features
aggregated from strictly earlier rows, (c) diagnostics.

**Does NOT establish:**
Anything about whether auxiliary targets or historical aggregates of these signals
*help*. That is untested and open.

**Provenance:**
- Official rules; PRE_AUDIT D01
- REVIEW_REPORT §7: classified INVALID / FORBIDDEN

---

### C4 — Multi-feedback auxiliary learning is permitted

**Classification:** HARD FACT

**Evidence:**
- The official material identifies KuaiRand's multi-feedback structure as a legitimate
  setting for auxiliary-task learning. Only `long_view` is scored.

**Interpretation:**
Predicting feedback signals as auxiliary targets is allowed under the rules.

**Does NOT establish:**
That any auxiliary task helps. Negative transfer is a real possibility here and is
unmeasured on this dataset; auxiliary densities range from ~46% (click) to ~0.1%
(follow, forward, hate), which is a wide spread for shared-representation methods.

**Provenance:**
- Official Starter Kit README / appendix
- REVIEW_REPORT: no dispute

---

### C23 — Per-user monotone transforms and global calibration are no-ops

> ID out of sequence: constraint IDs are append-only, so a late addition to an early
> section keeps the next free number rather than renumbering the file.

**Classification:** HARD FACT

**Evidence:**
- Both metrics depend only on the within-user *ordering* of scores: GAUC through
  Mann-Whitney U over ranks, nDCG@5 through the label sequence after sorting by score.
- Any strictly increasing function applied to one user's scores preserves that ordering,
  as does any global calibration (Platt scaling, temperature, an isotonic fit over the
  whole split).

**Interpretation:**
Probability calibration cannot move either metric. A well-calibrated score and a wildly
mis-calibrated one that ranks identically receive the same primary.

**Does NOT establish:**
That calibration is pointless during *training* — a calibrated objective can still change
what the model learns, and that is a separate question. Nor does it extend to transforms
that differ across a user's own rows: those change ordering and are not covered here.

**Provenance:**
- Mathematical consequence of C1 and the `evaluate.py` implementation
- Independently recorded in the `kuairand-starter-kit` context package (its C2)

---

## 2. Negative Evidence on the Official FM — Organizer-Confirmed and Locally Reproduced

> Both entries in this section are scoped to the **exact official five-field FM**. Neither
> generalises to a different loss, encoding, or model family.

### C5 — Static-feature expansion of the official FM

**Classification:** STRONG NEGATIVE EVIDENCE (for the exact 13-field formulation)

**Evidence:**
- Local, validation, 3 matched seeds. Base (5 fields) 0.601440 ± 0.000275.
- Full static bundle (**13** fields): 0.599930 ± 0.000523; paired delta
  **−0.001510 ± 0.000792**.
- Item-only expansion (**8** fields): 0.601108 ± 0.000461; paired delta
  −0.000332 ± 0.000205.
- Organizer reference (published, test split): 13 fields 0.5940 vs 5 fields 0.5950.
- The configurations contain 8 and 13 fields — the "9/14" labels in older notes are
  wrong; see `DATA_GUIDE.md` §8.

**Interpretation:**
The exact 13-field static bundle is reproducibly slightly worse than the five-field
baseline, and independently so in the organizer's own test-split run. Coarse static
buckets add little on top of identity fields in this FM.

**Does NOT establish:**
That static or derived features are useless in other models, encodings, or objectives.
The 8-field item-only expansion is **INCONCLUSIVE**, not negative — its delta is small
relative to run variability, so that configuration remains an open question rather than
a closed one.

**Provenance:**
- PRE_AUDIT C01; official README and `ablation_features.py`
- REVIEW_REPORT §3 (8/13 correction), §7 (13-field STRONG NEGATIVE, 8-field
  INCONCLUSIVE), §9: APPROVE

---

### C6 — Simple FM width scaling produced no meaningful gain

**Classification:** STRONG NEGATIVE EVIDENCE (for simple width scaling in this FM)

**Evidence:**
- Local, validation, 3 seeds per setting, mean primary ± population std:
  k=8 0.60111 ± 0.00080; k=16 0.60144 ± 0.00027; k=32 0.60146 ± 0.00069;
  k=64 0.60099 ± 0.00044.
- Spread across all four widths is smaller than the 0.002 practical epsilon and
  comparable to seed noise.
- Organizer reference (published, test split): k = 8/16/32 gave 0.5895 / 0.5902 /
  0.5887.
- Scope: the official five-field FM, embedding width varied alone, nothing else changed.

**Interpretation:**
Embedding width alone is not a lever in this exact model at this data size.

**Does NOT establish:**
That capacity in general is irrelevant, or that other model families, regularisation
schemes, schedules, or objectives cannot benefit from more parameters. It says nothing
about width in a *different* architecture.

**Provenance:**
- PRE_AUDIT J01; official README
- REVIEW_REPORT §7: STRONG NEGATIVE EVIDENCE, scoped to simple width scaling

---

## 3. Evidence Integrity

### C7 — Evaluation-period information is not development evidence

**Classification:** INVALID / FORBIDDEN

**Evidence:**
- Official rule: development uses train + validation only.
- Two source audits violated this before review — one locally scored standard-test
  labels, another summarised evaluation-period random-log outcomes and used
  evaluation-period identities in a comparison set. Every such result was removed from
  the evidence base rather than annotated.

**Interpretation:**
No locally computed test metric, evaluation-period outcome, evaluation-period feature
summary, or evaluation-period identity comparison may support a development decision.
Permitted during development: counting evaluation-window rows by `date`, and reading
the organizer's published test numbers as reference material.

**Does NOT establish:**
That evaluation rows may never be touched — the frozen final model is applied to
evaluation features to produce a submission. The prohibition is on evaluation
information flowing *back* into development.

**Provenance:**
- Official rules; PRE_AUDIT §0 and §14
- REVIEW_REPORT §3 (contamination removal), §7, §9: APPROVE

---

### C8 — The official baseline reproduces locally, and the noise scale is known

**Classification:** HARD FACT

**Evidence:**
- Reproduced seed 0, official configuration, validation: GAUC 0.667133,
  nDCG@5 0.535806, primary 0.601470 — against published 0.6674 / 0.5357 / 0.6016.
  Seed 0 selected epoch 7 and stopped after epoch 11.
- Five-seed validation-only rerun: mean primary 0.60157, population std 0.00032.
- Official configuration: fields `user_id`, `video_id`, `author_id`, `tab`,
  `dur_bucket`; k=16; lr=0.001; L2=1e-6; batch 8,192; max 40 epochs; patience 4;
  pointwise binary cross-entropy; Adam on W/V with a plain bias update; early stopping
  on validation primary.
- Published organizer seed std: 0.0008. The convergence epsilon 0.002 is roughly 2.5x
  that figure.

**Interpretation:**
The local environment reproduces the official baseline within seed and rounding
variation, so validation deltas measured in this environment are comparable to the
official ones. **Use 0.0008 as the generic noise reference, not the narrower local
0.00032** — a delta under ~0.0008 is indistinguishable from a seed, and a delta under
0.002 is below the competition's own practical threshold.

**Does NOT establish:**
Any local test result, and nothing about alternative models. The reproduction validates
the harness, not a research direction. The training objective being pointwise while the
metric is rank-based is a fact about the code, not evidence for any particular
alternative objective — none was tested.

**Provenance:**
- PRE_AUDIT R00 and C01; `research/experiment_results/baseline_validation.json`
- REVIEW_REPORT §3: published 0.0008 retained as the safer noise reference

---

## 4. Metric Structure

### C9 — Validation label composition, GAUC eligibility, and the oracle ceiling

**Classification:** HARD FACT

**Evidence:**
- Validation users 22,377 / rows 124,909.
- All-negative 6,785 users (30.321%), 21,807 rows (17.458%).
- All-positive 2,663 users (11.901%), 4,540 rows (3.635%).
- Mixed-label 12,929 users (57.778%), 98,562 rows (78.907%).
- Single-impression users: 3,917 (17.505%).
- Reproduced validation oracle (true labels as scores): GAUC 1.0000, nDCG@5 0.6968,
  primary 0.848393.
- Scope: official validation split, official evaluator.

**Interpretation:**
42.222% of validation users have uniform labels: they contribute a fixed nDCG (0 or 1)
regardless of ranking and are excluded from GAUC entirely. The mixed-label 57.778% of
users hold 78.907% of validation rows. Progress should be read against the 0.848393
ceiling, not against 1.0.

**Does NOT establish:**
That uniform-label users should be filtered out of training, that mixed users should be
upweighted, or that any reweighting improves the metric. Training composition and
metric composition are different questions, and the relationship between them is
untested.

**Provenance:**
- PRE_AUDIT B01; official `evaluate.py`
- REVIEW_REPORT §9: APPROVE

---

### C10 — Official GAUC weight denominator, and where baseline headroom sits

**Classification:** HARD FACT (for this reproduced baseline and these fixed bucket definitions)

**Evidence:**
- The official GAUC denominator is **34,592 positive rows belonging to mixed-label
  users only** — not all positive validation rows. All shares below use it.
- Activity tiers are quartiles of train interaction count among warm validation users:
  Cold 0, T1 1–17, T2 18–36, T3 37–65, T4 66+.

| Tier | Users | Rows | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 1.69% |
| T1 | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 14.67% |
| T2 | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 21.35% |
| T3 | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 27.50% |
| T4 | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 34.79% |

| List length | Users | Baseline GAUC | Baseline nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|
| 1 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.00% |
| 2–3 | 6,218 | 0.6472 | 0.5413 | 0.6086 | 10.27% |
| 4–5 | 4,119 | 0.6645 | 0.6185 | 0.7492 | 16.36% |
| 6–10 | 5,225 | 0.6756 | 0.5913 | 0.8536 | 36.39% |
| 11–20 | 2,346 | 0.6677 | 0.5037 | 0.9182 | 27.08% |
| 21+ | 552 | 0.6596 | 0.3934 | 0.9420 | 9.90% |

`*` empty GAUC denominator; the evaluator's 0.5 fallback, contributing zero weight.

- Train activity and validation list length are related but distinct: Spearman rho 0.4620.
- The joint cell T3/T4 x list-length 6+ holds 5,680 users (25.38%), 64,133 rows
  (51.34%), 50.79% of official GAUC weight, and 51.72% of the seed-0
  baseline-to-oracle primary gap.
- All 30 activity x list cells reconcile to 22,377 users, 124,909 rows, 100% of GAUC
  weight, and the full baseline-to-oracle gaps.
- Scope: reproduced seed-0 baseline predictions under the official configuration; these
  bucket edges only. nDCG contributions are equal-user weighted and are never
  multiplied by GAUC shares.

**Interpretation:**
Metric weight and current-baseline headroom are unevenly distributed across users, and
activity and list length are correlated without being interchangeable.

**Does NOT establish:**
Causality, attainable gain, or that any weighting, loss, filter, or model closes the
gap where it currently sits. A large share of headroom is not evidence that the share
is reachable, and headroom measured under *this* baseline may move under another.

**Provenance:**
- PRE_AUDIT B02
- REVIEW_REPORT §3 (corrected denominator), §9: APPROVE WITH REWORDING

---

## 5. Dataset Structure

### C11 — Development scale, coverage, and missingness

**Classification:** HARD FACT

**Evidence:**
- Train 1,141,112 rows / 26,210 users / 7,538 videos / 6,482 authors / 15 tabs / 13 dates.
- Validation 124,909 rows / 22,377 users / 5,951 videos / 5,315 authors / 15 tabs / 7 dates.
- Side tables: 27,285 users, 7,583 videos (basic), 7,583 videos (statistic); each covers
  100% of train and validation entities.
- Logs and the video-statistic table have no missing cells. Localised missingness: user
  `onehot_feat4` 3.2032%; `onehot_feat12`..`17` 2.6168% each; basic `video_duration`
  3.1518%; `music_type` 2.6770%; `tag` 1.2660%. Basic `visible_status` is constant.
- Train impressions per user: median 31, p90 97, p99 207, max 809 (all train users).
- Validation list length: median 4, p90 12, p99 26, max 74.

**Interpretation:**
Every side table joins completely; missingness is confined to a handful of columns and
is small; validation lists are short, with a median of four impressions.

**Does NOT establish:**
Usefulness, encoding, or causal validity of any field. Complete coverage is a join
property, not an information property.

**Provenance:**
- PRE_AUDIT A01
- REVIEW_REPORT §3 (activity median corrected to 31 for all train users), §8

---

### C12 — Warm entities, novel pairs, and author/video structural redundancy

**Classification:** HARD FACT

**Evidence:**

*Entity vs relationship overlap (validation against train):*
- Users seen in train: 21,955 / 22,377 = **98.114%**
- Videos seen: 5,944 / 5,951 = **99.882%**
- Authors seen: 5,310 / 5,315 = **99.906%**
- Unique user–video pairs seen: 1,974 / 121,337 = **1.627%**
- Unique user–author pairs seen: 4,081 / 120,885 = **3.376%**
- Raw user–tag-string pairs seen (missing tag = one explicit category):
  61,405 / 90,121 = **68.14%**. A parsed multi-token construction instead yields
  71.913% pair overlap and 78.413% validation-row coverage — a *different definition*,
  not a competing measurement of the same thing.

*Within-train repeat structure:*
- 4.130% of unique user–video pairs repeat, covering 8.194% of rows.
- 5.913% of unique user–author pairs repeat, covering 11.750% of rows.

*Author/video redundancy:*
- Every video maps to exactly one author (functional dependency).
- Full basic file: 5,661 / 6,510 authors (**86.96%**) have exactly one video; median 1,
  mean 1.165, max 26.
- Restricted to train/validation-observed videos: 5,647 / 6,487 (**87.051%**), max 24.

**Interpretation:**
Validation is almost entirely warm at the level of individual entities but almost
entirely novel at the level of the exact user–item and user–author relationship.
`author_id` carries substantial structural redundancy with `video_id` for most of the
catalogue. Coarser relationships (tags) have far broader support, with the number
depending on which tag representation is used.

**Does NOT establish:**
That `author_id` should be removed, that author-level interactions cannot add value,
that sparse exact-pair history is useless, or that tag features help. It also does not
merge the two tag definitions into one number — they must never be compared directly.

**Provenance:**
- PRE_AUDIT A02
- REVIEW_REPORT §4 (scope split 86.96% vs 87.051%; raw vs parsed tag), §9: APPROVE
  (author redundancy folded into this entry rather than promoted as its own constraint)

---

### C13 — Strictly prior history: broad at coarse granularity, sparse at fine

**Classification:** HARD FACT

**Evidence:**
- Every train timestamp precedes every validation timestamp, so all of train is
  legitimate history for validation.
- Validation users with ≥1 / ≥5 / ≥10 prior train interactions: 98.114% / 92.854% /
  **85.168%**. Median / mean / p90 prior interactions per validation user: 35 / 47.42 /
  103. (This median is over *validation users' train history*; the median over all
  train users is 31 — see C11. Different populations, both correct.)
- Validation users with ≥1 / ≥5 / ≥10 prior clicks: 96.157% / 82.531% / 66.309%.
  With prior likes: 23.229% / 4.683% / 2.239%. Follows, comments, forwards, and hates
  are sparser still.
- Validation rows with a prior interaction on the **same video**: **1.624%**.
  Same **author**: **3.381%**. A prior **parsed tag token**: **78.413%**.
- Availability diagnostic: 81.57% of validation rows have a strictly earlier same-user
  validation timestamp. Tied timestamps are **not** predecessors, and 5.60% of rows sit
  in non-unique user/timestamp groups.

**Interpretation:**
User-level and click-level history is broadly available; exact item or author repeats
are rare; coarse content granularity has wide coverage. Support therefore depends
heavily on the granularity chosen, and the two ends differ by more than an order of
magnitude.

**Does NOT establish:**
That aggregates, sequence models, or tag attention improve validation — none was
tested. The 81.57% figure is an availability count, **not** a validated online-history
protocol; whether within-validation outcomes can be made available before scoring
remains open. The official documentation also notes KuaiRand-Pure has incomplete
sequential logs.

**Provenance:**
- PRE_AUDIT E01
- REVIEW_REPORT §3 (82.09% corrected to 81.57%; tie semantics), §9: APPROVE

---

### C14 — Temporal volume and period-level distribution shift

**Classification:** HARD FACT (component measurements)

**Evidence:**
- The nominal train date 2022-04-08 has **zero rows**; train rows span 13 dates.
- Daily volume falls overall across the train window with small reversals: peak 278,835
  rows on 04-11, final train day 20,021 rows on 04-21 — a **13.9x** ratio.

| Period | Rows | Rows/day | `long_view` rate | Mean duration |
|---|---:|---:|---:|---:|
| Early train 04-09..14 | 891,418 | 148,570 | 0.33228 | 98,553 ms |
| Late train 04-15..21 | 249,694 | 35,671 | 0.35211 | 95,477 ms |
| Validation 04-22..28 | 124,909 | 17,844 | 0.31328 | 102,820 ms |

- Validation sits closer to **early** train in target rate (gap 0.01900 vs 0.03882) and
  mean duration (4,267 vs 7,343 ms), but closer to **late** train in tab distribution,
  volume, and some entity-set measures.

**Interpretation:**
Temporal change across the window is real, large in volume terms, and
multidimensional — different dimensions point in different directions.

**Does NOT establish:**
Whether validation "resembles" late train overall — that single verdict is
**INCONCLUSIVE**, and the component measurements must not be collapsed into it. It
establishes nothing about whether recency weighting, date features, or dropping early
rows helps.

**Provenance:**
- PRE_AUDIT G01
- REVIEW_REPORT §5 (combined verdict inconclusive), §9: APPROVE WITH REWORDING

---

## 6. Controlled Field Ablation on the Official FM

> Scoped to the **exact official FM formulation**, train/validation, official evaluator.
> It does not generalise to a different loss, encoding, or model family.

### C15 — Removing `tab` from the official FM

**Classification:** STRONG NEGATIVE EVIDENCE (against the removal)

**Evidence:**
- 3 matched seeds. Base 0.601440 ± 0.000275; without `tab` 0.585538 ± 0.000429.
- Paired delta **−0.015903 ± 0.000467** — roughly 8x the practical epsilon and 20x the
  published seed std.

**Interpretation:**
`tab` carries distinct, large value in this FM. Dropping it is a substantial regression.

**Does NOT establish:**
That `tab` must appear in every future model in this exact form, or that context
information is exhausted by a single 15-value categorical field.

**Provenance:**
- PRE_AUDIT C01
- REVIEW_REPORT §9: APPROVE WITH REWORDING (scoped to the tested FM)

---

### C16 — Retired (dual `video_id` + `author_id` ablation)

**Status:** withdrawn from `constraints.md` by human decision; the ID is retired rather
than reused.

The controlled five-seed ablation of the two identity fields is a **measurement**, not a
constraint. It is recorded in `research/data_profile.md` §12 with its paired deltas,
seed counts, and the reviewer's WEAK NEGATIVE EVIDENCE classification for the exact FM
formulation.

It was removed from this file because it was the only positive-direction result in the
package, and a positive result sitting among established constraints risks reading as a
direction to pursue. The related **structural** redundancy between `author_id` and
`video_id` remains available as evidence in C12, where it belongs.

---

## 7. Data-Source Scope Boundaries

### C17 — Retired (video-statistic provenance and causal safety)

**Status:** withdrawn from `constraints.md` by human decision; the ID is retired rather
than reused.

This was the package's one INCONCLUSIVE entry, retained against the general rule that
inconclusive findings stay out of this file. It has been removed to keep that rule
absolute: an open question does not belong among established constraints, however
consequential it is.

The substance is preserved in full, and in the two places an implementer actually reads:

- `DATA_GUIDE.md` §6 — the semantic caveat, stated before any use of these fields:
  values are averages not totals, the window's endpoints are undocumented, the source
  population is undocumented, 54 field pairs are near-redundant, and causal
  admissibility for an April 22–28 impression is therefore **not established** —
  neither a settled ban nor a clearance.
- `research/data_profile.md` §9 — the measurements: reconstruction ratios, redundancy
  count, marginal association, and the explicit INCONCLUSIVE verdict on aggregation
  population, calendar window, and causal safety.

Whether these fields can be given acceptable provenance, and whether they add anything
beyond identity features, is open work for the agent.

---

### C18 — Random-exposure log: permitted scope and validation-slice structure

**Classification:** HARD FACT (retained structure). The evaluation-period portion is
INVALID / FORBIDDEN as development evidence under C7.

**Evidence:**
- File total 1,186,059 rows spanning 2022-04-22..05-08.
- Validation dates 04-22..28: **288,338** rows. Evaluation dates 04-29..05-08:
  **897,721** rows, counted by `date` only.
- Validation slice: 19,091 users, 7,546 videos, `long_view` rate **0.08056** — against
  standard validation's 0.31328.
- The validation slice shares **17 of its 288,328** unique user–video pairs (0.006%)
  with standard validation.
- Older figures derived from full-file entity/pair inspection or from locally inspected
  evaluation-period outcomes were removed, not corrected.

**Interpretation:**
The eligible validation-period random stream is distributionally very different from
standard validation traffic — roughly a quarter of the positive rate — and is almost
entirely pair-disjoint from it.

**Does NOT establish:**
A propensity estimator, an unbiased replacement metric, a training use, or predictive
validity for standard traffic. Whether this slice is a useful secondary diagnostic is
**INCONCLUSIVE** and untested.

**Provenance:**
- PRE_AUDIT H01
- REVIEW_REPORT §3 (contamination correction), §5, §9: APPROVE

---

## 8. Engineering Constraints

### C19 — A bare Windows subprocess timeout did not bound the tested process tree

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- On the tested Windows inherited-pipe process tree,
  `subprocess.run(timeout=3, capture_output=True)` returned only after the grandchild's
  full **30.13 s** lifetime.
- A separate recursive-termination probe did successfully remove both parent and child.

**Interpretation:**
A timeout argument alone is not a guarantee on this platform when a child spawns its own
child and pipes are inherited. Process-tree control has to be explicit.

**Does NOT establish:**
That any particular replacement mechanism works — the recursive probe was a different
test under different conditions, and the harness that will run the measured iterations
has not been validated against this failure mode.

**Provenance:**
- PRE_AUDIT I01
- REVIEW_REPORT §7 (ENGINEERING CONSTRAINT), §9: APPROVE WITH REWORDING

---

### C20 — Baseline runtime and caching, as observed

**Classification:** ENGINEERING CONSTRAINT (run- and implementation-specific)

**Evidence:**
- One reviewer rerun: ~57.5 s cold baseline (2.99 s load, 4.81 s encode, 49.7 s train);
  0.018 s cache read; bit-identical arrays on reload.
- A separate fingerprinted implementation: 78.52 s cold; 1.384 s for full-content
  fingerprint plus cache read; it correctly rejected a changed source fingerprint.
- Environment: Windows 11, Python 3.13.7, CPU only.

**Interpretation:**
A baseline-class iteration costs on the order of a minute on CPU, and deterministic
caching of the load/encode stages is achievable. The two timings come from different
implementations under different instrumentation — they are not a contradiction and not a
stable benchmark.

**Does NOT establish:**
Runtime for any more complex model, a validated six-hour autonomous run, resume or
checkpoint behaviour, or a production cache policy. Budgeting 50 iterations against
6 hours on these numbers assumes iterations stay baseline-class, which is not given.

**Provenance:**
- PRE_AUDIT I01
- REVIEW_REPORT §4 (retained as run-specific observations)

---

### C21 — Submission and scoring validity contract

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- The official submission reader rejects `NaN` and `Inf` scores.
- `row_id` is the required key: `(user_id, video_id)` is not unique in the evaluation
  split (the organizer reports 3.06% duplicated pairs, up to 12 repeats).
- A syntax error in a child process returns a detectable nonzero exit status.

**Interpretation:**
Score-validity and row-alignment failures are detectable before submission and must be
checked rather than assumed.

**Does NOT establish:**
That the current repository implements any of these checks — see C22.

**Provenance:**
- Official `submit.py`; PRE_AUDIT I01
- REVIEW_REPORT §8

---

### C22 — The implementation layer is unimplemented scaffolding

**Classification:** ENGINEERING CONSTRAINT

**Evidence:**
- All **15** files under `harness/`, `pipeline/`, and `agent/` contain **zero**
  executable non-comment lines. `pipeline/models/`, `pipeline/objectives/`, and
  `agent/prompts/` are empty directories.
- An earlier probe counted only seven such files; review corrected this to 15.

**Interpretation:**
Nothing in the autonomous system is built yet. Every guarantee the run depends on —
scoring, execution, timeout handling, caching, logging, submission validation — is
currently a comment.

**Does NOT establish:**
Any estimate of the effort required, or that the intended design is sound. It states
repository state at audit time; re-verify before relying on it.

**Provenance:**
- PRE_AUDIT I01
- REVIEW_REPORT §3, §9: APPROVE

---

## 9. What Must NOT Appear in This File

Not permitted here:

- Strategy or direction — "try recency weighting", "use DIN", "multi-task is the best
  next step", "prioritise video statistics", "expand on whichever ablation moved in the right direction".
- A negative result generalised past its tested formulation — "BPR is useless",
  "static features don't work".
- An INCONCLUSIVE finding rewritten as a prohibition or an endorsement.
- Any number derived from evaluation/test labels or evaluation-period outcomes.
- A GAUC weight share computed over all positive rows rather than over positives
  belonging to mixed-label users.

Write the verified evidence instead, with its scope, and let the agent draw the
conclusion. If a proposed addition cannot be phrased as "X was measured to be Y under
scope Z", it does not belong here.

Every addition requires: finding, classification, numerical evidence with scope,
investigation ID, what it establishes, what it does not establish, and human review.
