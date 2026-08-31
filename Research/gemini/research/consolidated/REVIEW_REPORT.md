# Consolidated Pre-Audit Review Report

## 1. Executive Summary

This report merges three independent pre-audit sets (Gemini, Claude, GPT) into one authoritative research package. The consolidation resolved several numeric discrepancies and clarified terminology across the audits.

- **3 audit sets** were merged.
- **1 major conflict/correction** was found regarding the inclusion of evaluation-period test labels during validation analysis (Claude test leakage), which has been strictly removed.
- **2 major numeric corrections** were retained (GAUC weight denominator corrected to mixed-label users only; User Activity median corrected to 31).
- The merged package is **highly trustworthy** and ready for human constraint review.

## 2. Source Sets Reviewed

- Audit 1: Gemini (PRE_AUDIT, REVIEW_REPORT, data_profile)
- Audit 2: Claude (PRE_AUDIT, REVIEW_REPORT, data_profile)
- Audit 3: GPT (PRE_AUDIT, REVIEW_REPORT, data_profile)

## 3. Major Corrections Preserved

- **Test/evaluation contamination**: Any results derived from standard-test labels or evaluation-period random-log outcomes present in early audits (Claude) have been completely removed. The random log is only verified for its date range and is forbidden from use in training.
- **GAUC weight-share denominator**: GAUC weight shares now exclusively use the official denominator (positive counts from mixed-label users only = 34,592), fixing early audits that included all-positive users.
- **User activity percentiles**: The median train activity for all users was corrected to 31 (p99: 207), replacing a conflated value of 35 that referred to prior interactions for returning users only.
- **Video statistic means**: The `like_cnt` mean was corrected to 230.75.

## 4. Conflicts Resolved

| Conflict                         | Reason                                    | Final Decision                                                                      |
| -------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `is_follow` prevalence           | Rounding discrepancy (0.131% vs 0.13049%) | Standardized to exact 0.130%.                                                       |
| Baseline ablation interpretation | Weak vs Strong negative on dual IDs       | Standardized to WEAK NEGATIVE for the exact baseline FM, not a general prohibition. |
| Temporal trend interpretation    | "Smooth decay" vs "Not smooth"            | Described numerically: 13.9x peak-to-trough drop. Cause left inconclusive.          |
| Engineering scope files          | 7 files vs 15 files                       | Corrected to 15 empty scaffold files across the repository.                         |

## 5. Conflicts Still Inconclusive

- **Causal validity of `video_features_statistic_pure.csv`**: The exact time aggregation window remains undocumented. Whether it safely excludes future evaluation information is unproven and must be treated as INCONCLUSIVE.
- **Recency weighting / Temporal drift**: While the volume decays sharply, it is inconclusive if the target distribution shifts enough to warrant recency weighting.
- **Multi-task transfer**: While `is_click` and `play_time_ms` are strongly correlated with `long_view`, their efficacy as multi-task targets remains inconclusive until empirically tested.

## 6. Duplicate Findings Consolidated

- **Entity redundancy**: 87% of authors having exactly one video was independently found by all three audits and merged as a HARD FACT.
- **Entity overlap**: 98% user and 99% video overlap, but <2% exact user-video pair overlap. Merged using the precise >3 decimal place counts.
- **History coverage**: 85% of users having >=10 interactions was merged as a HARD FACT.
- **Metric Headroom**: 42.22% invariant validation users was confirmed across audits.

## 7. Evidence Classification Changes

- **Baseline author/video ablation**: Downgraded to WEAK NEGATIVE (from some strong assumptions) because the slight improvement (+0.001) only applies to this exact Factorization Machine structure and does not imply the features are universally useless.
- **Auxiliary task effectiveness**: The claim that click/play_time are "effective" auxiliary tasks was downgraded from a fact to INCONCLUSIVE (untested), while their raw density/correlation remains a HARD FACT.

## 8. Data Profile Consistency Check

The consolidated `PRE_AUDIT.md` and `data_profile.md` have been cross-checked.

- Stale numbers have been purged.
- The GAUC weight definition is uniform.
- All evidence labels match.

## 9. Candidate Findings for constraints.md

| Finding                                                                        | Final Classification | Recommendation | Reason                                                                          |
| ------------------------------------------------------------------------------ | -------------------- | -------------- | ------------------------------------------------------------------------------- |
| 42.22% of validation users are invariant (all-pos or all-neg)                  | HARD FACT            | APPROVE        | Mathematical constraint of the GAUC/nDCG evaluation.                            |
| `is_click` and `play_time_ms` are dense and strongly correlated (r=0.75, 0.63) | HARD FACT            | APPROVE        | Validated diagnostic fact. Efficacy for multi-task is explicitly left untested. |
| 85% of users have >=10 prior interactions, but exact user-video repeat is <2%  | HARD FACT            | APPROVE        | Critical context for sequence modeling constraints.                             |
| Train volume decays 13.9x from peak to end                                     | HARD FACT            | APPROVE        | Demonstrates temporal non-stationarity in traffic.                              |
| Random exposure log entirely post-dates the train window                       | STRONG NEGATIVE      | APPROVE        | Training on this data violates temporal split rules.                            |
| Static CWM feature stuffing (8-13 fields) didn't improve baseline              | STRONG NEGATIVE      | APPROVE        | Supports existing constraint C5.                                                |

## 10. Remaining Questions for Autonomous Agent

- Which loss function (pointwise vs. pairwise/listwise) best optimizes nDCG/GAUC for this task?
- How should historical sequences be modeled given the sparsity of exact item repeats but the density of tag/author repeats?
- Can multi-task learning with `is_click` or `play_time_ms` avoid negative transfer?
- Does temporal recency weighting improve performance despite volume decay?

## 11. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

The merged package resolves prior numeric and scoping conflicts, completely excises test-set leakage, and establishes a robust factual foundation for the autonomous agent.
