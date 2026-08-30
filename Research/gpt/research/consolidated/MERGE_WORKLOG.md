# KuaiRand-Pure Audit Merge Worklog

> Temporary reasoning record for the consolidation completed on 2026-08-30. The three source sets are preserved under `Sets of audit file/`. “GPT,” “Gemini,” and “Claude” below identify source sets, not additional datasets. Reviewer-corrected statements take priority over their original audit text.

## Source inventory and authority

| Audit set | PRE_AUDIT | REVIEW_REPORT | data_profile | Merge role |
|---|---|---|---|---|
| Audit 1 — GPT | `gpt_PRE_AUDIT.md` | `gpt_REVIEW_REPORT.md` | `gpt_data_profile.md` | Detailed validation-only reproduction; current `research/experiment_results/` supports this set |
| Audit 2 — Gemini | `Gemini_PRE_AUDIT.md` | `Gemini_REVIEW_REPORT.md` | `Gemini_data_profile.md` | Independent structural audit; reviewer corrected activity/statistic values and random-log scope |
| Audit 3 — Claude | `claude_PRE_AUDIT.md` | `claude_REVIEW_REPORT.md` | `claude_data_profile.md` | Broadest reviewed audit; controlling source for GAUC denominator, contamination removals, tied timestamps, and joint metric decomposition |

Official authority read first: `context/PROBLEM.md`, `context/RULES.md`, `context/DATA_GUIDE.md`, `context/constraints.md`, and the unmodified Starter Kit (`README.md`, `data.py`, `evaluate.py`, `baseline.py`, `baseline_scores.json`). The existing main `research/*.md` files matched the GPT set and were treated as a copy, not a fourth audit.

## Topic crosswalk

| Topic | Audit 1 — GPT | Audit 2 — Gemini | Audit 3 — Claude | Reviewer status | Final decision |
|---|---|---|---|---|---|
| Official task, split, and metric | Correct; evaluator inspected | Correct | Correct | Official code controls | Merge as HARD FACT; evaluation row count may be stated only as an official/date-only fact |
| Validation baseline reproduction | 0.667133 / 0.535806 / 0.601470, official configuration | Only published score; one descriptive run was capped at 12 epochs | 0.6671 / 0.5358 / 0.6015; five-seed mean 0.60157 | Reproduced | Use precise GPT seed-0 values and Claude five-seed summary; exclude every locally computed test score |
| Split/entity cardinalities | Exact counts | Same counts | Same counts | Verified independently | Merge |
| Missingness and side-table coverage | Detailed inventory and 100% entity coverage | Same selected missingness | Same | Verified | Merge |
| Train→validation entity overlap | 98.114/99.882/99.906% | Same rounded | Same rounded | Verified | Use precise GPT counts/percentages |
| User-video/user-author overlap | 1.627% / 3.376% | 1.63% / 3.38% | Same | Verified | Merge |
| User-tag overlap | Parsed tag-token pairs: 71.913%; row-any-token 78.413% | Raw tag-string pairs: 68.14% | Raw tag string with missing as explicit category: 68.14%; row same-tag 73.19% | Definitions differ | Preserve both only with definitions; do not present as one number |
| Author-video redundancy | Observed train+valid videos: 5,647/6,487 = 87.051%, max 24 | Full basic file: 5,661/6,510 = 86.96% | Full basic file: 86.96%, max 26 | Scope explains difference | Use full-file 86.96% in profile; retain observed-scope result in PRE_AUDIT |
| Within-split repeat pairs | Detailed user-video/author/token repeats | Selected rates | Corrected raw-tag missing semantics: 51.77% train pairs, 24.45% valid pairs | Tag correction confirmed | Use exact video/author results; use corrected raw-tag definition and label parsed-token alternative separately |
| Train user activity | Median 31, p99 207, max 809 | Corrected from stale 35/~250 | Consistent | Reviewer correction confirmed | Use median 31; reserve 35 for validation users’ train history |
| Validation list/label composition | Exact counts | Same | Same | Verified | Merge |
| Oracle validation ceiling | 0.8484 official/reproduced | 0.8484 | 0.848393 local validation | Verified | Use 0.848393 locally reproduced; 0.8484 rounded |
| Activity tiers | Quartiles over all train users: 1–13/14–31/32–59/60+ | Ad hoc <10/10–49/50–149/150+, single short run | Quartiles among warm validation users: 1–17/18–36/37–65/66+ | Different definitions; Claude denominator corrected | Standardize consolidated package on Claude’s 17/36/65 boundaries; document alternatives as non-conflicting omitted views |
| GAUC weight shares | Official denominator in GPT artifacts, but different activity buckets | Not reliable for precise bucket metrics | Original denominator wrong, reviewer corrected to 34,592 positives from mixed users | Major correction confirmed | Use only official-denominator shares; never use all validation positives |
| List-length buckets | Full seed-0 baseline table, official denominator | Older 12-epoch scores differ | Corrected table matches GPT on final values | Reviewer correction confirmed | Use GPT/Claude final table; length-1 GAUC weight is 0% |
| Joint activity × list length | Not measured | Confound identified only | 30-cell decomposition, rho=0.4620, reconciles all totals | Post-review extension verified | Retain as HARD FACT with current-baseline/non-causal scope |
| Baseline mechanism | Exact code walk | Broad description | Same | Official code controls | Merge |
| Video/author field ablation | Five matched seeds; originally called STRONG NEGATIVE against dual-ID formulation | Not run | Five matched seeds; reviewer calls narrow WEAK NEGATIVE | Reviewer wording is more conservative | Use WEAK NEGATIVE EVIDENCE for the exact FM formulation |
| `tab` ablation | −0.015903 ± 0.000467 paired | Not run | Same | Repeated controlled result | STRONG NEGATIVE EVIDENCE against removing `tab` in this FM |
| `dur_bucket` ablation | −0.000591 ± 0.000156; called WEAK NEGATIVE | Not run | Reviewer lists conclusion as unresolved/practically small | Conservative merge | INCONCLUSIVE |
| Static expansions | Correct 8/13 fields; 13-field negative | Organizer result only | Reviewer corrected stale 9/14 labels to 8/13 | Correction confirmed | 13-field exact formulation STRONG NEGATIVE EVIDENCE; 8-field item-only INCONCLUSIVE |
| FM embedding dimension | k=8/16/32 flat | Organizer evidence only | k=8/16/32/64 flat | Repeated | STRONG NEGATIVE EVIDENCE against simple width scaling in tested FM |
| Learning rate | 0.0005/0.001/0.002 differences inconclusive | Not run | 0.0003/0.001 inconclusive; 0.01 clearly poor; 0.003 noisy/lower | Reviewed | WEAK NEGATIVE EVIDENCE only for tested high-LR settings, principally 0.01; nearby rates INCONCLUSIVE |
| Feedback prevalence | Detailed; corrected follow to 0.130% | Same rounded | Same | GPT reviewer correction confirmed | Merge corrected value |
| Same-row feedback association | Raw/log1p definitions explicit | Some correlations used different transforms | Corrected click-play correlation 0.5167 | Definition/transcription corrections | Retain target correlations only with transform noted; same-row inputs marked INVALID / FORBIDDEN |
| Prior train history | 98.114/92.854/85.168%; median 35 | Same rounded; some row coverage less precise | Same core values | Verified | Merge; explicitly distinguish user coverage, row coverage, and activity-bucket definition |
| Within-validation earlier history | Not central | Not established | 82.09% corrected to 81.57% because tied timestamps are not earlier | Correction confirmed | Retain 81.57% as availability diagnostic only; protocol remains INCONCLUSIVE |
| Video basic features | Detailed redundancy inventory | Basic selected stats | Same | Verified | Merge compact facts |
| Video-stat means | Detailed correlations/ratios | Corrected like mean 230.75 and filled missing means | Same aggregation caveat, different selected statistics | Corrections confirmed | Keep verified means; do not label the source “safe” |
| Video-stat aggregation window | Official “one month,” endpoint undocumented; causal safety unresolved | “Likely long/future” caution, but profile incorrectly called fields safe | Reviewer: population/window/safety cannot be inferred | Genuine unresolved semantic conflict | INCONCLUSIVE; no causal feature claim |
| Video-stat ratio evidence | Standalone scores, all ≤ item popularity | Marginal inventory only | Marginal validation correlations/quintiles | Different methods | Retain both scopes: HARD FACT associations; WEAK NEGATIVE EVIDENCE for exact standalone scorers; safety INCONCLUSIVE |
| Temporal volume | Exact periods; multidimensional similarity | Reviewer surfaced 04-08 zero day and 13.9× swing | Same, plus set overlap | Verified | Merge; “validation resembles late train” remains INCONCLUSIVE |
| Random log | Full-file IDs/date only; 702 overlap with train+valid | Reviewer removed test-period ID comparison; 702 was a different full-file scope | Reviewer restricts outcomes/features to validation slice; 288,338 rows and 17 shared pairs with standard validation | Major integrity correction | Final profile uses validation-slice outcomes/features only; evaluation period contributes date-only count; omit full-file entity/pair analysis from conclusions |
| Engineering runtime/cache | 78.52s pipeline; fingerprinted cache; 1.41GB profile peak | Rough 40–50s | Reviewer rerun 57.5s; 0.018s cache; ~491MB run peak | Run/harness scopes differ | Report as run-specific observations, not stable constraints; prefer reviewer rerun for compact profile and describe GPT run separately in PRE_AUDIT |
| Windows timeout/process tree | Hardened recursive termination probe passed | Not tested | Bare inherited-pipe case overran 3s to 30.13s; fix untested there | Both compatible | HARD FACT for each tested condition; bare timeout is an ENGINEERING CONSTRAINT |
| Repository readiness | Reviewer expanded 7 files to 15 comment-only scaffolds | Not inventoried | Not inventoried | GPT correction confirmed | Retain 15-file HARD FACT; no implementation performed |

## Resolved conflict ledger

The following 20 material conflicts or stale values were resolved:

1. Removed locally generated standard-test metrics.
2. Removed evaluation-period random-log outcomes/features; retained date-only counts.
3. Corrected GAUC denominator to 34,592 positives from mixed-label users.
4. Separated GAUC weighting from equal-user nDCG weighting.
5. Corrected length-1 GAUC weight from a stale nonzero value to 0%.
6. Standardized activity-tier boundaries to 0 / 1–17 / 18–36 / 37–65 / 66+.
7. Replaced Gemini’s short-run list metrics with the reviewed full baseline values.
8. Corrected validation `is_follow` prevalence to 0.130% (163/124,909).
9. Corrected all-user train activity median/p99/max to 31/207/809; kept history median 35 under its different population.
10. Corrected `like_cnt` mean to 230.75 and filled long-play/comment means.
11. Separated 86.96% full-file author redundancy from 87.051% observed-scope redundancy.
12. Separated raw tag-string overlap from parsed-token overlap.
13. Applied explicit-missing-category semantics to raw tag repeats.
14. Corrected strictly earlier within-validation coverage from 82.09% to 81.57%.
15. Corrected static schema names from 9/14 to 8/13 fields.
16. Corrected the click/play-time inter-correlation wording/value to 0.5167.
17. Removed the false claim that every video-ratio quintile trend is monotonic.
18. Downgraded video/author ablation evidence to WEAK NEGATIVE EVIDENCE for the exact FM.
19. Made `dur_bucket` removal INCONCLUSIVE.
20. Corrected the claim that local std 0.00032 is “more conservative”; the published 0.0008 is used as the safer generic reference.

## Conflicts and semantics still inconclusive

| Question | Why unresolved | Final treatment |
|---|---|---|
| Exact population/window and causal safety of `video_features_statistic_pure.csv` | Local files do not document endpoints or source population | INCONCLUSIVE; quarantine any causal claim |
| Whether validation is globally “closer” to early or late train | Rate/duration favor early; volume/tab/entity structure favor late | INCONCLUSIVE; report component facts only |
| Whether validation-period random exposure is a useful model-selection diagnostic | Distribution is distinct and no controlled predictive-validity experiment exists | INCONCLUSIVE |
| Whether within-validation outcomes can be used as deployable online history | Availability was measured, but no online serving protocol was established | INCONCLUSIVE |

## Merge accounting

- Duplicate topic groups consolidated (supported by at least two audit sets): **23**.
- Material conflicts/stale values resolved: **20**.
- Conflicts/semantic disputes still inconclusive: **4**.
- Raw source sets modified: **0**.
