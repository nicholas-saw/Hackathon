# KuaiRand-Pure Pre-Audit Review Report

## 1. Executive Summary

The pre-audit contains substantial good work: the core dataset profile, official evaluator interpretation, baseline reproduction on validation, history coverage, and most controlled FM experiments are reproducible and appropriately scoped. I reviewed all **28** listed investigations: **13 VERIFIED**, **9 VERIFIED WITH MINOR EDIT**, **5 REQUIRES CORRECTION**, and **1 INCONCLUSIVE**.

Two major issues were found. First, the original pre-audit locally scored standard-test labels once and summarized outcomes/features from the evaluation-period random log. Those actions violated the train+validation-only rule even though they were not used for model selection. Second, all reported “GAUC weight shares” used every positive validation row rather than the official denominator of positives from mixed-label users only. Both issues are now corrected. The test-derived values were removed, the loaders were hardened, the affected artifacts were regenerated, and the metric tables/plot were updated.

The evidence base is **ready for human constraint review**, provided the candidate constraints use the corrected wording in this report. Uncertainty remains around video-statistic causal validity, several small/noisy model effects, recency, and any untested historical, multi-task, random-log, or alternative-loss method.

Post-review extension: at the user's request, B06 was added after the 28-investigation audit count above. It directly measures the relationship between train-activity tier and validation list length and was validated as a 30-cell disjoint decomposition; it does not change the original review-status counts.

## 2. Critical Issues Found

### Issue R-01 — Evaluation-period outcomes were accessed during pre-audit

Severity: **MAJOR**

Affected investigation(s): baseline reproduction; F01

Problem: The unmodified official baseline was run in a mode that scored local test labels. Separately, the original random-log audit calculated evaluation-period `long_view` rate and entity overlaps using evaluation-period rows.

Why it matters: The audit mandate permits train and validation only. Calling the use “structural” does not make an outcome statistic compliant, and an official script is not exempt from the split rule.

Correction made: Removed the locally produced test metrics and evaluation-period random-log outcome/feature summaries. `common.load_valid_log()` and `common.load_random_log()` now check the date before materializing complete rows; random-log evaluation access is limited to date-only counts. `phase_i_random_log.py` and its JSON artifact were regenerated.

Corrected result: Published test numbers remain reference material only. All retained local metrics and outcome summaries use train/validation. The validation-period random log has 288,338 rows, 19,091 users, 7,546 videos, and long-view rate 0.08056; it shares 17 unique user-video pairs with standard validation.

Remaining uncertainty: There is no repository history with which to prove what code ran before the review. The contamination is nevertheless bounded by the written report and code paths: Phase C model experiments use the research loader/encoder and were not selected using test results.

### Issue R-02 — GAUC weight shares used the wrong denominator

Severity: **MAJOR**

Affected investigation(s): B01, B02, B04, B05; `phase_b_headroom.png`

Problem: The prior code divided bucket positives by all validation positives. Official `evaluate.py` excludes uniform-label users from GAUC and weights contributing mixed-label users by their positive count.

Why it matters: The reported values were explicitly called official GAUC weight shares and were used for headroom prioritization. The clearest sanity failure was assigning 4.06% GAUC weight to length-1 users, none of whom can enter GAUC.

Correction made: The denominator is now the 34,592 positive rows belonging to mixed-label users. The JSON, narrative tables, profile, and plot were regenerated. GAUC and nDCG contributions are kept separate; nDCG gaps are weighted by user count when discussing their contribution to the overall metric.

Corrected result: T3/T4 shares are 27.50%/34.79% (62.29% combined), not 26.61%/31.79%. The length 6–10 share is 36.39%, not 32.83%; length 1 is 0%, not 4.06%.

Remaining uncertainty: Bucket analysis describes where the current baseline-to-oracle gap lies; it does not show which method can close it.

### Issue R-03 — Same-timestamp rows were treated as historical predecessors

Severity: **MINOR**

Affected investigation(s): D04/D05 bonus within-validation history statistic

Problem: `cumcount()` arbitrarily ordered rows tied on `(user_id, date, hourmin, time_ms)`, contrary to the strict-earlier rule.

Why it matters: A later feature implementation copied from this diagnostic could update history before scoring a simultaneous row.

Correction made: History is now accumulated by timestamp group and broadcast before updating with that group.

Corrected result: 81.57% of validation rows have a strictly earlier same-user validation row, not 82.09%. Some 5.60% of rows are in non-unique user/timestamp groups.

Remaining uncertainty: No deployable within-validation/test online-feedback protocol was established; this remains an availability diagnostic only.

### Issue R-04 — Static-expansion configurations were mislabeled

Severity: **MINOR**

Affected investigation(s): C04

Problem: Configurations were named `item_9field` and `cwm_14field`, while their saved field lists contain 8 and 13 fields.

Why it matters: The mismatch made the code, result artifact, and prose internally inconsistent.

Correction made: Renamed them `item_8field` and `cwm_13field` in script, JSON, log, profile, and pre-audit.

Corrected result: Scores are unchanged: 0.60111 and 0.59993 versus base 0.60144.

Remaining uncertainty: Only the exact tested expansion is covered.

### Issue R-05 — Tag-repeat computations used inconsistent missing-value semantics

Severity: **MINOR**

Affected investigation(s): A03, A05

Problem: Pair overlap converted missing tags to a string, while repeat `groupby` silently dropped missing tags.

Why it matters: Identically named user-tag statistics used different denominators.

Correction made: Missing tag is now one explicit categorical value in both calculations.

Corrected result: Train/validation repeated-pair rates are 51.77%/24.45%, and row shares are 84.98%/45.49%.

Remaining uncertainty: Treating missing as one category is an explicit encoding choice; any future tag model should record it.

## 3. Verified Findings

The following table records every investigation. “Rerun” means the reviewer executed the relevant script or an independent reproduction; “artifact/code” means code and saved outputs were cross-checked without repeating an expensive full sweep.

| ID | Original claim | Review status | Checks / rerun / reproduction | Corrected or retained result and interpretation | Evidence class |
|---|---|---|---|---|---|
| A01 | Dataset/entity cardinalities | VERIFIED | Independent raw-data reproduction | 1,141,112 train rows; 124,909 valid; 26,210/22,377 users; 7,538/5,951 videos; 6,482/5,315 authors | HARD FACT |
| A02 | Missingness profile | VERIFIED | Reran Phase A; checked raw side files | Core logs have no missing values; reported side-field missingness reproduced | HARD FACT |
| A03 | High entity but low pair overlap | VERIFIED | Independent set calculations | 98.11% users, 99.88% videos, 99.91% authors; 1.63% user-video pairs | HARD FACT |
| A04 | 87% one-video authors | VERIFIED | Independent groupby | 5,661/6,510 = 86.96%; max 26 videos/author | HARD FACT |
| A05 | Sparse video/author repeats; dense tags | VERIFIED WITH MINOR EDIT | Independent reproduction and rerun | Core conclusion holds; tag values corrected for explicit missing category | HARD FACT |
| A06 | Missing 04-08 and sharp volume decline | VERIFIED | Independent daily counts | 13 dates; 278,835 rows on 04-11 versus 20,021 on 04-21 | HARD FACT |
| B01 | Activity-tier weight/headroom | REQUIRES CORRECTION | Audited official GAUC denominator; reran Phase B | T4 = 34.79%; T3+T4 = 62.29%; nDCG weighting kept separate | HARD FACT |
| B02 | List-length weight/headroom | REQUIRES CORRECTION | Same; checked zero-weight length-1 sanity | Length 6–10 = 36.39%; length 1 = 0%; 6–10 is largest nDCG-gap contributor | HARD FACT |
| B03 | Uniform-label proportions/oracle | VERIFIED | Independent per-user counts; official evaluator rerun | 30.32% all-negative, 11.90% all-positive; oracle primary 0.848393 | HARD FACT |
| B04 | GAUC weight concentration | REQUIRES CORRECTION | Same official-denominator audit | Corrected shares in Phase B artifact/profile | HARD FACT |
| B05 | Combined movable headroom | REQUIRES CORRECTION | Decomposed GAUC by official weight and nDCG by user share | Concentration remains, but multiplying GAUC share by an nDCG gap is not an official metric contribution | HARD FACT after correction |
| B06 | Joint activity-tier × list-length relationship | POST-REVIEW EXTENSION | Spearman/Pearson association, 5×6 user cross-tab, and exact disjoint metric reconciliation | Spearman ρ=0.462; T3/T4 × 6+ contains 50.79% of GAUC weight and 51.72% of primary headroom | HARD FACT |
| C01 | Dropping video/author modestly helps this FM | VERIFIED WITH MINOR EDIT | Five matched seeds rerun | Drop-video +0.00108, drop-author +0.00132; positive in 5/5 each; narrow WEAK NEGATIVE only | WEAK NEGATIVE |
| C02 | Local seed variance | VERIFIED WITH MINOR EDIT | Five saved seeds reproduced during C01 rerun | Mean 0.60157; population std 0.00032, sample std 0.00035; organizer 0.0008 is the more conservative reference | HARD FACT measurement |
| C03 | LR ≥0.003 clearly degrades | VERIFIED WITH MINOR EDIT | Checked split/procedure/seeds and aggregate artifact | 0.01 is clearly poor; 0.003 is lower but noisy; 0.0003 vs 0.001 remains inconclusive | WEAK NEGATIVE for tested high LR; otherwise INCONCLUSIVE |
| C04 | Static expansion does not help | VERIFIED WITH MINOR EDIT | Checked actual field lists and 3-seed results | Correct names are 8/13 fields; base 0.60144, item 0.60111, full CWM 0.59993 | STRONG NEGATIVE for exact formulation |
| C05 | FM width 8–64 is flat | VERIFIED | Checked 3-seed sweep and procedure | Means span 0.60098–0.60146, within run variability | STRONG NEGATIVE for simple width scaling |
| D01 | Feedback prevalence | VERIFIED | Independent validation prevalence | Values reproduced, including click 44.38% and follow 0.130% | HARD FACT |
| D02 | Same-row feedback association | VERIFIED | Checked JSON and code; diagnostic-only boundary | Click r=0.751 and play time r=0.632; never model inputs | HARD FACT diagnostic |
| D03 | Two feedback clusters | VERIFIED WITH MINOR EDIT | Checked correlation matrix | Click/play-time r=0.5167, not “0.60s”; comment/comment-stay r=0.3029; clustering language remains descriptive | HARD FACT correlations |
| D04 | Prior train-history availability | VERIFIED | Independent user-count reproduction | 98.11% have ≥1, 85.17% ≥10; median 35, p90 103 | HARD FACT |
| D05 | Availability varies by activity | VERIFIED | Reran Phase F | T2+ all have ≥10 train rows; strict timestamp bonus corrected separately | HARD FACT |
| E01 | Video-stat inventory/window inference | INCONCLUSIVE | Audited calculations and semantic leap | Integer-product and scale ratios reproduce; actual aggregation population/window and causal safety remain unknown | HARD FACT numerics + INCONCLUSIVE semantics |
| E02 | Ratios associate with validation target | VERIFIED WITH MINOR EDIT | Checked quintiles/correlations | r values reproduce; like-ratio Q5 dips slightly; association is not incremental value or safety | HARD FACT association + INCONCLUSIVE safety |
| F01 | Random log is distinct and temporally risky | REQUIRES CORRECTION | Reran using validation slice only; eval date counts only | 288,338 valid-period rows; 17 shared pairs; 8.06% vs 31.3% long-view rate; no eval outcomes retained | HARD FACT |
| G01 | Baseline runtime about one minute | VERIFIED WITH MINOR EDIT | Full engineering rerun | Cold run 57.5s; run-specific, not a stable constraint | HARD FACT for recorded run |
| G02 | Cache is identical and much faster | VERIFIED | Reran | Bit-identical; 0.018s reload vs 4.81s encode (~263× on this run) | HARD FACT for recorded run |
| G03 | Bare Windows timeout can overrun | VERIFIED WITH MINOR EDIT | Reran synthetic process-tree test | 3s timeout returned after 30.11s in the tested inherited-pipe condition; proposed fixes remain untested here | HARD FACT for tested condition |
| G04 | NaN and syntax errors are detectable | VERIFIED | Reran official validator/subprocess checks | Both cleanly rejected/detected | HARD FACT |

## 4. Corrected Findings

| Original claim | Corrected claim | Reason | New evidence classification |
|---|---|---|---|
| T3+T4 had 58.4% of GAUC weight; length 6–10 had 32.8%; length 1 had 4.06% | 62.29%, 36.39%, and 0% respectively | Uniform-label positives were incorrectly included in the GAUC denominator | HARD FACT |
| “Weighted nDCG gap” used GAUC shares × nDCG gap | GAUC gap uses official positive weights; nDCG gap contribution uses equal user weights | The two metrics have different aggregation semantics | HARD FACT decomposition |
| Random-log evaluation outcomes were acceptable as structural facts | Evaluation-period outcomes/features are out of scope; only date-only counts remain | Train+validation-only rule | Integrity correction |
| 82.09% had earlier within-validation history | 81.57% have strictly earlier history | Same-timestamp rows were arbitrarily ordered | HARD FACT |
| `item_9field` / `cwm_14field` | `item_8field` / `cwm_13field` | Off-by-one naming; saved field lists were correct | HARD FACT |
| All ratio quintile trends were monotonic | Three are monotonic; like ratio dips Q4→Q5 | Direct artifact inspection | HARD FACT |
| Click/play-time inter-correlation was in the “0.60s” | It is 0.5167 | Direct artifact inspection | HARD FACT |
| Local std 0.00032 was a “more conservative” noise floor | It is lower, not more conservative; use 0.0008 as the conservative reference | Statistical wording error | HARD FACT measurement |

## 5. Inconclusive Findings

- The population, time window, and causal safety of `video_features_statistic_pure.csv` cannot be inferred from the files. `show_cnt × counts` being nearly integral and far larger than sampled traffic supports an external/larger aggregation hypothesis, not a proof of the window.
- Whether recency weighting helps is untested. Volume/entity overlap and raw target-rate comparisons point in different directions.
- `dur_bucket` removal (−0.00059 in the original three-seed run) is too small for a confident conclusion.
- Learning rate 0.0003 versus 0.001 is unresolved. The 0.003 setting is lower but noisy; only 0.01 is a clear failure among the tested higher rates.
- The random validation log may be a useful diagnostic, but “random exposure” does not by itself make it interchangeable with the official standard-validation objective.
- Historical features, sequence models, multi-task learning, pairwise/listwise losses, and ratio features were not tested for incremental validation value.

## 6. Leakage / Integrity Audit

- **Test access:** Violations were found in the original work: one local standard-test score and evaluation-period random-log outcome/feature summaries. They were removed. Current research scripts expose only train/validation rows; evaluation counts inspect `date` only.
- **Same-row outcome leakage:** No model input path uses current-row feedback. D01–D03 use it diagnostically only. Phase C features contain identities/context/static features only.
- **Future-history leakage:** No historical model was trained. Phase F's within-validation availability calculation was corrected so tied timestamps do not precede one another. Any future feature must compute from prior history, score the row/timestamp group, then update.
- **Validation leakage:** Validation labels are used for official early stopping and comparison, as disclosed. Vocabularies, duration quantiles, and model fitting use train only. Cached validation users/labels/scores exactly align with raw validation order. No train feature is constructed from validation outcomes.
- **Random-log temporal leakage:** The regenerated artifact uses only 2022-04-22..28 outcomes/features. 2022-04-29..05-08 contributes date-only row counts and nothing else.
- **Target-derived filtering/weighting:** Model training uses all train rows. Metric subgrouping uses labels descriptively, not to filter model training. Corrected GAUC analysis now matches official inclusion/weighting.
- **Source-file integrity:** No file under `source/` was edited during review. SHA-256 hashes of the six official starter-kit references are recorded in `review_artifacts/core_reproduction.json`. Because this workspace has no usable Git history, provenance against an external pristine copy cannot be independently proven; the inspected logic matches the documented official behavior.

## 7. Statistical Reliability Review

The official baseline comparison is validation-only and uses the same encoder, optimizer, early stopping, and evaluator across configurations. The five baseline seeds have mean primary 0.60157 and sample std 0.00035. The published 0.0008 seed std remains the safer generic threshold because five local seeds estimate variance imprecisely and variance changes by configuration.

C01 received the strongest additional check: five matched seeds. Drop-video and drop-author were positive in 5/5 paired comparisons, with mean deltas +0.00108 and +0.00132. This supports a reproducible weak negative against those fields in the exact pointwise FM, not a general field prohibition.

C04 and C05 use only three seeds per configuration. Their conclusions are appropriately narrow: the tested static bundle and simple FM width scaling did not help. C03 needed weaker wording: the 0.003 result has substantial variance, while 0.01 is clearly poor. All results are also conditioned on repeated validation use for early stopping and hypothesis comparison; they should not be portrayed as untouched-holdout estimates.

## 8. Data Profile Consistency

After correction, `data_profile.md` agrees with the scripts and JSON artifacts on row/entity counts, overlap, author redundancy, uniform-label proportions, oracle metrics, feedback prevalence, history availability, temporal volume, and validation-period random-log statistics.

The post-review B06 extension also agrees with the regenerated Phase B artifact: the 30 activity-tier × list-length cells sum exactly to 22,377 users, 124,909 rows, 100% of official GAUC weight, and the full baseline-to-oracle metric gaps.

Corrections made:

- Official GAUC shares and associated plot.
- User-tag repeat denominators and counts.
- Strictly earlier within-validation history percentage.
- Random-log entity coverage and removal of evaluation-period outcome rate.
- C04 configuration names and field counts.
- Ratio-trend and inter-feedback wording.
- Engineering timings/cache factor after the hardened loader rerun.
- Stale investigation links in `data_profile.md`.
- Joint B06 activity-tier × list-length tables and exact gap reconciliation.

## 9. Candidate constraints.md Review

| Candidate | Decision | Reason |
|---|---|---|
| 1 — Missing 04-08 / volume decline | APPROVE WITH REWORDING | Deterministic, but say volume “falls from peak to final day,” not monotonically decays every day |
| 2 — Activity/list metric concentration | APPROVE WITH REWORDING | B06 now verifies a moderate association and joint concentration; retain separate GAUC/nDCG weighting and avoid causal claims |
| 3 — Same-row click/play-time correlations | DO NOT PROMOTE | Correct but redundant with existing C3; same-row diagnostic correlations add little durable constraint value |
| 4 — Video-stat aggregation uncertainty | APPROVE WITH REWORDING | Numeric scale evidence is valid; semantic window/population inference must remain explicitly unresolved |
| 5 — Windows subprocess timeout | APPROVE WITH REWORDING | Verified for the specific inherited-pipe/grandchild condition; recommended fixes were not tested |
| 6 — Video/author FM ablation | APPROVE WITH REWORDING | Five-seed paired rerun reproduced the narrow result; must not generalize beyond this FM formulation |
| 7 — Static expansion and FM width | APPROVE WITH REWORDING | Validation evidence supports existing C5/C6; correct field counts to 8/13 and avoid relying on test results as review evidence |

## 10. Recommended Wording for Approved Constraints

**Candidate 1:** `log_standard_4_08_to_4_21_pure.csv` contains no rows dated 2022-04-08 and therefore has 13 represented train dates. Daily volume falls from 278,835 rows at its 04-11 peak to 20,021 on 04-21 (13.9×); this does not establish that recency weighting helps.

**Candidate 2:** Train-side user activity and validation list length are moderately positively associated (Spearman rho = 0.462). The T3/T4 activity-tier × 6+ list-length intersection contains 25.38% of validation users, 50.79% of official GAUC weight, and 51.72% of the current seed-0 baseline-to-oracle primary gap. This is a diagnostic concentration, not evidence that either dimension causes the error or that a particular method will close it.

**Candidate 4:** The aggregation window and source population for `video_features_statistic_pure.csv` are undocumented. `show_cnt × counts` is nearly integral for all videos and reconstructed totals are much larger than sampled train+validation traffic (median ratio about 11,465×), suggesting a larger external population/window. Feature safety and evaluation-period overlap remain unresolved.

**Candidate 5:** On this Windows environment, `subprocess.run(timeout=3, capture_output=True)` returned after 30.11 seconds when the child spawned a 30-second grandchild inheriting the output pipe. Bare subprocess timeout is therefore insufficient for this tested process-tree condition; any replacement tree-termination mechanism must be tested separately.

**Candidate 6:** In the official pointwise-logloss FM with k=16 and lr=0.001, individually removing `video_id` or `author_id` improved validation primary by +0.00108/+0.00132 over five matched seeds, with positive deltas in 5/5 seeds for each. This is a weak negative for this exact formulation only and does not establish that item or author identity is generally uninformative.

**Candidate 7:** Existing C5/C6 are independently consistent with validation experiments: the tested 8/13-field static expansions scored 0.60111/0.59993 versus 0.60144 for the five-field baseline, and FM dimensions 8/16/32/64 scored 0.60111/0.60144/0.60146/0.60098. These are negatives only for the tested static bundle and simple FM width scaling.

## 11. Remaining Questions for the Autonomous Agent

- Whether any historical feature or sequence model improves validation while preserving strict prior-only construction.
- Whether pairwise/listwise training improves the official primary metric.
- Whether the C01 field-ablation result survives other losses and model families.
- Whether video statistics can be used under an acceptable causal/provenance policy and add value beyond identity.
- Whether multi-task auxiliary outcomes help and how sparse tasks should be weighted.
- Whether the validation-period random log provides a useful secondary diagnostic without displacing the official validation objective.
- Whether any temporal weighting helps despite conflicting descriptive drift indicators.
- Whether tag affinity adds incremental value rather than merely broad coverage.

## 12. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

The major integrity and metric errors were contained and corrected, high-impact structural results independently reproduced, the surprising C01 result rerun with matched seeds, and stale artifacts regenerated. The remaining uncertainty is genuine research uncertainty rather than an unresolved audit failure. Human review should use the wording in Section 10, not the original candidate text.
