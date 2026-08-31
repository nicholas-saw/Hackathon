# Consolidated Pre-Audit Review Report

## 1. Executive Summary

Three independent KuaiRand-Pure audit sets were merged by topic, with official Starter Kit/rules first, reviewer corrections second, reproduced artifacts third, and original audit prose last.

- Audit sets merged: **3**.
- Duplicate topic groups consolidated: **23**.
- Material conflicts or stale values resolved: **20**.
- Conflicts/semantic disputes still inconclusive: **4**.
- Candidate findings recommended for human `constraints.md` approval: **12**.
- Critical integrity issue retained in the final package: **none**. The source audits did contain test/evaluation contamination and an incorrect GAUC denominator; both are excluded/corrected here and documented below.
- Readiness: **READY FOR HUMAN CONSTRAINT REVIEW**.

The final package is trustworthy for its stated purpose: a conservative evidence base for an autonomous research agent. It is not a recommendation list and does not begin model development.

## 2. Source Sets Reviewed

| Set | PRE_AUDIT | REVIEW_REPORT | data_profile | Notes |
|---|---|---|---|---|
| Audit 1 — GPT | `Sets of audit file/gpt_PRE_AUDIT.md` | `gpt_REVIEW_REPORT.md` | `gpt_data_profile.md` | Detailed local validation reproduction and current saved artifacts |
| Audit 2 — Gemini | `Sets of audit file/Gemini_PRE_AUDIT.md` | `Gemini_REVIEW_REPORT.md` | `Gemini_data_profile.md` | Independent structural audit; several reviewer corrections |
| Audit 3 — Claude | `Sets of audit file/claude_PRE_AUDIT.md` | `claude_REVIEW_REPORT.md` | `claude_data_profile.md` | Broadest corrected metric/leakage audit; controlling source for major corrections |

Also reviewed as higher authority: `context/PROBLEM.md`, `context/RULES.md`, `context/DATA_GUIDE.md`, `context/constraints.md`, and the official Starter Kit. The pre-existing main `research/*.md` files matched Audit 1 and were not counted as another source.

## 3. Major Corrections Preserved

### Test/evaluation contamination removal

One source audit locally scored standard-test labels. Another summarized evaluation-period random-log outcomes/features, and one diagnostic used evaluation-period identities in a comparison set. None of those results is retained as development evidence.

Final rule:

- Standard evaluation/test labels are absent from local conclusions and the compact profile.
- Evaluation-period random-log rows contribute only a date-only row count.
- Random-log outcomes/features are restricted to 2022-04-22..28.
- Published official test numbers remain reference material only and are not used to support a development conclusion.

### Corrected GAUC weight-share definition

The official denominator is **34,592 positives from mixed-label validation users only**, not all positive validation rows. The final activity/list tables use that denominator. Consequences include:

- T3/T4 activity weight = 27.50%/34.79% (62.29% combined) under the consolidated 17/36/65 tier edges.
- List length 6–10 weight = 36.39%.
- List length 1 weight = 0%, as required because no single-label user contributes to GAUC.
- nDCG contributions remain equal-user weighted and are never multiplied by GAUC shares.

### Other retained reviewer corrections

- Validation `is_follow`: 0.130%, not 0.131%.
- Train activity across all train users: median 31, p99 207, max 809; median 35 belongs to validation users’ train history.
- `like_cnt` mean: 230.75, not 158; missing statistic means were filled where reproduced.
- Strictly earlier same-user validation history: 81.57%, not 82.09%; tied timestamps are not predecessors.
- Static configurations: 8 and 13 fields, not 9 and 14.
- Raw tag repeats use one explicit missing category.
- Click/play-time inter-correlation: 0.5167; the like-ratio quintile trend is not fully monotonic.
- Published std 0.0008, not local std 0.00032, is the safer generic noise reference.
- Repository readiness covers 15 scaffold-only files, not the original seven-file sample.

## 4. Conflicts Resolved

| Conflict | Reason | Final decision |
|---|---|---|
| 86.96% vs 87.051% one-video authors; max 26 vs 24 | Full basic file vs train/validation-observed videos | Keep both with scope; compact headline is full-file 86.96% |
| 68.14% vs 71.913% user-tag overlap | Raw tag strings vs parsed multi-token representation | Keep both definitions; never merge numerically |
| Different activity tiers | Three audits used different boundaries/populations | Standardize on warm-validation-user quartiles: 1–17/18–36/37–65/66+ |
| Different list-bucket baseline scores | Gemini used one seed and 12-epoch cap | Use full official-configuration seed-0 values verified by GPT/Claude |
| Nonzero length-1 GAUC share | Stale denominator included uniform-label positives | Correct to 0% |
| 31 vs 35 activity median | All train users vs validation users’ prior history | Retain both under explicit populations |
| Random pair overlaps 702 vs 17 | Full random vs standard train+valid, versus validation-only random vs standard validation | Use 17 in final conclusions; omit evaluation-period identity comparison |
| Runtime 57.5s vs 78.52s | Different loaders/instrumentation and run conditions | Retain as run-specific observations, not one stable benchmark |
| STRONG vs WEAK negative for video/author IDs | Five consistent but sub-epsilon paired deltas | WEAK NEGATIVE EVIDENCE for the exact dual-ID FM only |
| WEAK NEGATIVE vs unresolved `dur_bucket` ablation | Small practical effect and limited seeds | INCONCLUSIVE |
| “Safe” video statistics vs undocumented timing | Completeness does not establish causal timing | Safety INCONCLUSIVE |
| 9/14 vs 8/13 static fields | Stale labels disagreed with actual field lists | Use 8/13 |

The complete 20-item ledger is in `MERGE_WORKLOG.md`.

## 5. Conflicts Still Inconclusive

| Question | Why unresolved | Final handling |
|---|---|---|
| Video-statistic aggregation population/window and causal safety | No local/official file identifies endpoints or source population | INCONCLUSIVE; no safety claim |
| Whether validation is globally closer to early or late train | Target rate/duration favor early; volume/tab/entity structure favor late | INCONCLUSIVE; preserve component measurements |
| Whether validation-period random exposure is useful for model selection | It is distributionally distinct and untested as a predictive secondary diagnostic | INCONCLUSIVE |
| Whether within-validation outcomes can form deployable online history | Coverage exists, but no online availability/serving protocol was established | INCONCLUSIVE |

Untested methods—historical features, sequence models, multi-task learning, pairwise/listwise losses, ratio features in a combined model—are research questions, not source conflicts, and remain INCONCLUSIVE for efficacy.

## 6. Duplicate Findings Consolidated

Twenty-three multi-source topic groups were consolidated. The largest merged groups were:

- Core scale, missingness, side-table coverage, and 04-08 anomaly.
- Entity and pair overlap, author/video redundancy, and repeat structure.
- Uniform-label metric invariance and oracle ceiling.
- Activity/list-length headroom, with standardized definitions and corrected weights.
- Feedback density/association and strict same-row leakage boundary.
- Prior train-history availability by granularity.
- Video-basic/statistic inventory, redundancy, ratios, and timing uncertainty.
- Temporal structure and random-exposure scope.
- Baseline mechanism, field/static/capacity ablations, runtime, caching, and process control.

Equivalent rounded values were collapsed to the most precise reviewed value. Measurements with different scopes were retained separately or one was omitted from the compact profile.

## 7. Evidence Classification Changes

Only the requested controlled vocabulary is used in the final package.

| Finding | Source wording | Final classification | Reason |
|---|---|---|---|
| Dual video/author identity in exact FM | STRONG evidence of redundancy / narrow weak negative | WEAK NEGATIVE EVIDENCE | Five matched seeds agree, but mean gains are small and below the 0.002 practical epsilon |
| `dur_bucket` removal | WEAK NEGATIVE in one audit | INCONCLUSIVE | Small effect and limited practical magnitude |
| Item-only 8-field static expansion | Directionally negative | INCONCLUSIVE | Delta is small relative to variability |
| Full 13-field static expansion | Strong negative | STRONG NEGATIVE EVIDENCE | Three seeds lower; reviewed and narrowly scoped |
| FM width scaling | Null result | STRONG NEGATIVE EVIDENCE | Repeated k=8/16/32/64 comparison, scoped to simple width scaling |
| High learning rates | “Clearly degrades” for ≥0.003 | WEAK NEGATIVE EVIDENCE | 0.01 is clear; 0.003 is lower but noisy, so broader claim is softened |
| Current-row feedback as features | Leakage caution | INVALID / FORBIDDEN | Official rule |
| Evaluation-derived development evidence | Structural/reference in older sources | INVALID / FORBIDDEN | Official train+validation-only rule |
| Video-stat timing safety | Likely/safe/unclear variants | INCONCLUSIVE | Exact source population and endpoint are undocumented |
| Windows bare timeout | Runtime fact | ENGINEERING CONSTRAINT | Verified failure to bound a realistic process tree |

No “worked once” result was promoted to STRONG POSITIVE EVIDENCE; no final positive model-efficacy claim exists because the audits did not test such methods.

## 8. Data Profile Consistency Check

The consolidated `PRE_AUDIT.md`, `data_profile.md`, and `MERGE_WORKLOG.md` were checked for:

- Official row counts, split boundaries, baseline scores, and evaluator semantics.
- One-video-author scope labels.
- Raw-tag versus parsed-token definitions.
- Activity-tier boundaries and GAUC denominator.
- Length-1 GAUC share and nDCG weighting.
- Corrected `is_follow`, activity, statistic, timestamp, and static-field values.
- Absence of locally derived standard-test metrics and evaluation-period random-log outcomes.
- Evidence labels for dual identity, duration, static expansion, and width.

Current `research/experiment_results/` directly supports Audit 1’s reproduced baseline/profile values. Audit 2/3-specific reviewer artifacts were not copied into the shared `Sets of audit file/` folder; those corrections are therefore attributed to their reviewed reports, with cross-audit agreement or official semantics used where available. No number is represented as newly reproduced during this consolidation.

Result: the three consolidated documents agree under their stated scopes.

## 9. Candidate Findings for constraints.md

These are candidates only; `context/constraints.md` was not edited.

| Finding | Final Classification | Recommendation | Reason |
|---|---|---|---|
| 42.222% of validation users have uniform labels; only mixed users enter GAUC | HARD FACT | APPROVE | Official metric consequence and exact reproduced counts |
| Entities are >98% warm but user–video/user–author pair overlap is 1.627%/3.376% | HARD FACT | APPROVE | Independently reproduced structural fact |
| 86.96% of full-file authors have exactly one video | HARD FACT | DO NOT PROMOTE SEPARATELY | Exact fact, but already captured by the broader entity/redundancy candidate |
| 85.168% of validation users have ≥10 train interactions, while exact-item row coverage is 1.624% | HARD FACT | APPROVE | Strict train-derived history; no strategy directive |
| Train has no 04-08 rows and peak-to-final-day volume ratio is 13.9× | HARD FACT | APPROVE WITH REWORDING | Say “falls overall with reversals,” not monotonically every day |
| T3/T4 × list 6+ holds 50.79% of GAUC weight and 51.72% of seed-0 primary gap | HARD FACT | APPROVE WITH REWORDING | Keep bucket definitions, current-baseline scope, and non-causal wording |
| Click/play time are dense and same-row associated with `long_view` | HARD FACT | DO NOT PROMOTE | Correct but largely overlaps existing leakage constraint C3; efficacy untested |
| Video-statistic population/window and causal safety are undocumented | INCONCLUSIVE | APPROVE WITH REWORDING | Durable uncertainty; do not convert it into a ban or safety claim |
| Evaluation-period random outcomes must not support development; validation slice is distinct | INVALID / FORBIDDEN + HARD FACT | APPROVE | Preserves a major integrity correction and scoped structure |
| Bare Windows timeout overran to 30.13s in the tested child/grandchild condition | ENGINEERING CONSTRAINT | APPROVE WITH REWORDING | Verified condition; replacement mechanism not established by that test |
| Exact dual video+author-ID FM is weakly disfavored over five matched seeds | WEAK NEGATIVE EVIDENCE | APPROVE WITH REWORDING | Narrow reproducible result; must not generalize beyond this FM |
| Exact 13-field static stuffing and simple FM width scaling show no meaningful gain | STRONG NEGATIVE EVIDENCE | APPROVE | Repeated validation evidence, already aligned with C5/C6 |
| `tab` removal costs −0.015903 primary in the exact FM | STRONG NEGATIVE EVIDENCE | APPROVE WITH REWORDING | Large repeated effect; scope to the tested FM |
| Fifteen harness/pipeline/agent files are comment-only scaffolds | ENGINEERING CONSTRAINT | APPROVE | Exact repository-state fact; material to run readiness |

Approved/approve-with-rewording candidates: **12**. Do-not-promote-separately candidates: **2**. (The random-log row combines an official invalid-use boundary with a retained structural fact and counts as one candidate.)

## 10. Remaining Questions for Autonomous Agent

- Whether any strictly prior historical aggregate or sequence model improves validation.
- Whether pairwise/listwise training improves the official primary metric.
- Whether the video/author ablation survives other losses and model families.
- Whether video statistics can be given acceptable provenance and add incremental value beyond identity.
- Whether multi-task auxiliary outcomes help, and how sparse tasks affect transfer.
- Whether the validation-period random log is a useful secondary diagnostic without replacing standard validation.
- Whether any temporal feature or weighting helps despite conflicting drift indicators.
- Whether raw tags, parsed tokens, or another content granularity add incremental value.
- What safe online protocol, if any, would make within-period outcomes available before scoring.

## 11. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

All known-invalid development evidence is excluded, the official GAUC denominator is enforced, scope-dependent values are labeled, reviewer corrections are preserved, and the remaining disputes are explicitly INCONCLUSIVE. No autonomous model-development phase was started.
