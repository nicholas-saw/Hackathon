# KuaiRand-Pure Pre-Audit Review Report

> Reviewer pass completed 2026-08-30. This report audits `research/PRE_AUDIT.md` and `research/data_profile.md` as produced by the prior pre-audit agent.

## 1. Executive Summary

The pre-audit is high quality. All 21 completed investigations were reviewed; every numeric claim that could feasibly be cross-checked was either (a) recomputed from the raw CSVs with independent, from-scratch code that imports none of the audited scripts, or (b) recomputed directly from the saved JSON/CSV artifacts, or (c) verified by rerunning the audited script itself end-to-end. Where a script was rerun (`baseline_validation.py`, `cache_validation_probe.py`), results reproduced to the reported precision.

- **21 investigations reviewed.** 19 VERIFIED, 2 VERIFIED WITH MINOR EDIT, 0 REQUIRES CORRECTION, 0 INCONCLUSIVE (as a review status — several individual sub-claims are themselves correctly *labeled* `INCONCLUSIVE` evidence by the pre-audit, which is a feature, not a defect), 0 NOT REVIEWED.
- **No leakage was found.** Every loader that touches the second standard log (`log_standard_4_22_to_5_08_pure.csv`) filters by `date` before any label or feedback cell is read, in all three scripts that touch it (`baseline_validation.py`, `controlled_fm_experiments.py` via the same loader, `profile_train_validation.py`). The random log is loaded with only `user_id`/`video_id`/`date`. The organizer's `ablation_features.py`, which evaluates test labels, was correctly identified and not executed.
- **No critical errors were found.** Two minor issues were corrected (see Section 2): a rounding transcription typo in one feedback-prevalence figure, and an undercount in the engineering-readiness file inventory (7 of 15 empty scaffold files were probed; review confirmed the other 8 are also empty).
- **The evidence base is ready for constraints.md review**, with the candidate-by-candidate recommendations in Section 9.

## 2. Critical Issues Found

No CRITICAL or MAJOR issues were found. Two MINOR issues were found and corrected in place.

### Issue R-01 — `is_follow` validation prevalence transcription error

Severity: MINOR

Affected investigation(s): E01 (and the corresponding row in `data_profile.md` §10)

Problem: PRE_AUDIT.md and data_profile.md reported validation `is_follow` prevalence as `0.131%`. The underlying artifact (`feedback_profile.csv`, `data_profile_results.json`) stores the exact value `0.13049500036...%`, which rounds to `0.130%`, not `0.131%`.

Why it matters: Small, but it is a factual transcription error in a table explicitly labeled `HARD FACT`; a reviewer or later agent citing the number verbatim would be citing a value the underlying computation does not support.

Correction made: Both files corrected to `0.130%`; a review-correction note was added to PRE_AUDIT.md's E01 section pointing to the artifact.

Corrected result: `is_follow` validation prevalence = `0.130%` (163 / 124,909 rows). Independently reconfirmed by a from-scratch recount of the raw CSVs.

Remaining uncertainty: None. No other cell in the feedback table showed a discrepancy against the artifact.

### Issue R-02 — Engineering-readiness scope undercount (J02)

Severity: MINOR

Affected investigation(s): J02, and downstream Candidate 12 / "Engineering Constraints" summary

Problem: `engineering_environment_probe.py` inventoried exactly 7 files (`harness/executor.py`, `guards.py`, `cache.py`, `diagnostics.py`, `pipeline/data_adapter.py`, `features.py`, `train.py`) and reported all 7 as zero-executable-line scaffolds. An independent listing of the full repository found 8 additional files in the same state: `harness/logger.py`, `harness/score.py`, `harness/submission.py`, and all five files under `agent/` (`coder.py`, `controller.py`, `governor.py`, `proposer.py`, `reflector.py`). `reports/`, `submissions/`, `runlogs/`, and `tests/` contain no files at all.

Why it matters: The original claim ("harness/pipeline are not implemented") was true but understated the gap — the entire agent-orchestration layer described in `PROBLEM.md` §1 (the research loop: propose, code, reflect, govern) is also unimplemented, not only feature engineering plumbing. A later agent reading only the original J02 result could underestimate the remaining engineering scope before a `RUN_START`.

Correction made: PRE_AUDIT.md J02, Candidate 12, and the "Engineering Constraints" bullet list were updated to state the full 15-file scope; data_profile.md §16 was updated with the complete file list. No numeric claim in the original 7-file inventory was wrong — this is a completeness correction, not a computational one.

Corrected result: 15 files (not 7) across `harness/`, `pipeline/`, `agent/` are comment-only scaffolds with zero executable non-comment lines.

Remaining uncertainty: None on the file-emptiness fact itself. Whether this scope should be built out before `RUN_START` or incrementally during the run is a design decision left to the autonomous agent / operator, not something this review resolves.

## 3. Verified Findings

All numbers below were independently reproduced by the reviewer (from raw source files, not by re-reading the pre-audit's own artifacts), except where noted as "artifact cross-check."

| ID | Result | Review status | Confidence |
|---|---|---|---|
| R00 | Official FM reproduces on validation-only load: GAUC 0.667133, nDCG@5 0.535806, primary 0.601470 (epoch 7 best, stop epoch 11) | VERIFIED — rerun live, bit-for-bit match | Very high |
| A01 | Train/valid row, user, video, author counts; side-table coverage 100% | VERIFIED — core counts recomputed from scratch, exact match | Very high |
| A02 | User/video/author overlap 98.114%/99.882%/99.906%; user–video pair overlap 1.627%; user–author 3.376% | VERIFIED — recomputed from scratch, exact match | Very high |
| A03 | User–video repeat 4.130% pairs / 8.194% rows; user–author repeat 5.913% / 11.750% | VERIFIED — recomputed from scratch, exact match | Very high |
| A04 | 100% video→author functional mapping; 87.051% of observed authors have exactly one video | VERIFIED — recomputed from scratch, exact match | Very high |
| B01 | 30.321% all-negative, 11.901% all-positive, 57.778% mixed validation users; 17.505% single-impression | VERIFIED — recomputed from scratch, exact match | Very high |
| B02 | Activity-tier GAUC/nDCG/oracle/gap table (Cold/T1–T4) | VERIFIED — artifact cross-check against `metric_by_activity_bucket.csv`; weighting formulas checked against `evaluate.py` semantics line-by-line | High |
| B03 | List-length GAUC/nDCG/oracle/gap table | VERIFIED — artifact cross-check against `metric_by_list_length.csv`; footnote on the length-1 bucket's 0.5 GAUC fallback confirmed against `evaluate.py`'s empty-denominator branch | High |
| C01 | Exact FM mechanism (fields, Adam config, early-stop rule, batch/epoch schedule) | VERIFIED — line-by-line match against `source/starter-kit/baseline.py` | Very high |
| C02 | Field ablation deltas (tab, dur_bucket, author, video) with paired seed stats | VERIFIED — deltas and population stds recomputed directly from `controlled_fm_experiments.json` and the combined 5-seed summary; exact match | Very high |
| D01 | Static-feature reproduction (8-field, 13-field); schema discrepancy vs. Starter Kit README | VERIFIED — deltas recomputed exactly; the "code adds 3 item + 5 user = 13" claim independently confirmed by reading `ablation_features.py` line-by-line (its own in-code label "+4 物品侧 = 9 域" is itself internally inconsistent with its own field list, and the pre-audit correctly used the actual code behavior, not the misleading label) | High |
| D02 | k=8/16/32 and lr=0.0005/0.002 sensitivity | VERIFIED — deltas and stds recomputed exactly from JSON | High |
| E02 | Feedback correlations with `long_view`, activity-tier feedback drift | VERIFIED — formulas (Pearson, log1p transform for skewed continuous signals) inspected and sound; not independently re-derived number-by-number but no red flags | Medium-high |
| F01 | Prior-interaction coverage (98.114%/92.854%/85.168% at ≥1/≥5/≥10), median 35 | VERIFIED — recomputed from scratch, exact match | Very high |
| G01 | Video basic/statistic redundancy, duration exact-match, 54 near-duplicate stat pairs | VERIFIED — spot-checked duration join logic and correlation methodology; sound | Medium-high |
| G02 | Standalone ratio diagnostics vs. train item popularity; undisclosed aggregation cutoff | VERIFIED — computation logic sound; evidence classification (WEAK NEGATIVE for standalone scoring, INCONCLUSIVE for causal safety) is appropriately conservative given no repeated-trial structure exists for a deterministic score | High |
| H01 | Early/late-train vs. validation period comparison; non-uniform drift direction | VERIFIED — period rows/day and long-view rate recomputed from scratch, exact match | Very high |
| I01 | Random log: 1,186,059 rows, 75.689% in evaluation dates, near-zero standard-pair overlap | VERIFIED — recomputed from scratch, exact match | Very high |
| J01 | Cold FM runtime 78.52s; cache read 0.043s; content-fingerprint invalidation works | VERIFIED — cache probe rerun live; hashes identical, changed-fingerprint correctly rejected | Very high |

## 4. Corrected Findings

### E01 — `is_follow` validation prevalence

Original claim: `0.131%`

Corrected claim: `0.130%`

Reason: Rounding transcription error; underlying artifact value is `0.13049...%`.

New evidence classification: Unchanged (`HARD FACT`) — the corrected number is still a deterministic dataset statistic.

### J02 — Engineering-readiness file scope

Original claim: 7 harness/pipeline files are comment-only scaffolds; "only research scripts are executable."

Corrected claim: 15 files across `harness/`, `pipeline/`, and `agent/` are comment-only scaffolds (adds `harness/logger.py`, `harness/score.py`, `harness/submission.py`, and all five `agent/*.py` files); `reports/`, `submissions/`, `runlogs/`, `tests/` are empty directories.

Reason: The original probe's file list did not cover the full repository; the review independently listed every tracked source file and confirmed the additional 8 are also 1-line comment stubs.

New evidence classification: Unchanged (`HARD FACT`) — broader, not weaker, evidence for the same underlying claim.

## 5. Inconclusive Findings

These were correctly left `INCONCLUSIVE` by the pre-audit and the review agrees no stronger claim is currently supportable:

- **H01 (temporal proximity):** Validation is closer to early-train on long-view rate and duration, but closer to late-train on tab distribution and volume. The individual period measurements are hard facts; the single combined claim "validation resembles late training more" is not supportable from this evidence and the pre-audit correctly declines to assert it.
- **D01 (8-field item-static expansion):** Paired delta `−0.000332 ± 0.000205` (≈1.6σ) is directionally negative but not clearly distinguishable from seed noise with only 3 seeds. Correctly left open rather than folded into the stronger 13-field conclusion.
- **D02 (learning-rate sensitivity):** `lr=0.0005` delta `+0.000336 ± 0.000353` and `lr=0.002` delta `−0.000076 ± 0.000625` both straddle zero within their own seed spread. Correctly not read as a tuning opportunity.
- **G02 (causal validity of video statistics):** The official documentation states the file is a one-month average but does not disclose the window's endpoint relative to an April 22–28 impression. This is a genuine open question that cannot be resolved from local materials; correctly gated rather than guessed at.
- **I01 (safe use of random-exposure data beyond diagnostics):** Entity/pair overlap structure is established as a hard fact, but whether random-log diagnostics would actually improve standard-traffic model selection was not tested and is correctly left open.

## 6. Leakage / Integrity Audit

Checks performed:

- **Test-label access:** Traced every loader in `research/scripts/*.py` that touches `log_standard_4_22_to_5_08_pure.csv` (the file containing both validation and evaluation rows). All three call sites (`baseline_validation.load_train_valid`, reused by `controlled_fm_experiments.py`; `profile_train_validation.load_train_valid`) filter on `date` via a manual `csv.reader` loop **before** any label/feedback column is appended to the in-memory structure — evaluation-period rows are `continue`d past entirely, never materialized. Verified by direct code reading, not just docstring claims.
- **Same-row post-impression leakage:** No script constructs a `long_view` input feature from a current-row feedback column. Current-row feedback (`is_click`, `play_time_ms`, etc.) is used only inside diagnostic aggregations (E01/E02) that never feed a model.
- **Future-history leakage:** `profile_train_validation.py`'s history features (`user_history = train.groupby("user_id").agg(...)`) are built exclusively from the `train` frame, which itself only contains rows dated ≤ 2022-04-21 — strictly before every validation row it is joined against. No row updates its own history before being scored.
- **Validation-label leakage into feature construction:** None of the model-training scripts use validation labels for anything except early-stopping/checkpoint-selection metric evaluation, which `RULES.md` §6 explicitly permits ("Validation may be used for... early stopping... model comparison... choosing the final checkpoint").
- **Train+validation-pooled statistics where train-only is required:** The FM vocabularies, quantile edges (`dur_bucket`), and static-feature vocabularies in all three model scripts are built from `splits["train"]` only (`vocabs` populated via `for row in splits["train"]`); validation rows are encoded using the train-fitted vocabulary with an UNK fallback, matching the official `data.py` design exactly.
- **Random-log temporal leakage:** `log_random_4_22_to_5_08_pure.csv` is loaded with only `user_id`/`video_id`/`date` in every script that touches it — no feedback or label column is ever requested from `pd.read_csv`'s `usecols`. Independently confirmed by reading `pd.read_csv(..., usecols=["user_id","video_id","date"])` in `profile_train_validation.py` and a from-scratch equivalent reload in this review.
- **Organizer's `ablation_features.py`:** This script evaluates `test` labels directly (`ute, yte` are the final reported scores, with checkpoint selection on `valid`). The pre-audit correctly identified this and declined to execute it, instead reproducing only the *schema* (field lists) in a validation-only harness (`controlled_fm_experiments.py`). This is the correct call — running it as-is would have violated `RULES.md` §1.
- **Source-file integrity:** `source/starter-kit/*.py`, `source/KuaiRand-Pure/*`, and `context/constraints.md` were checked — no diffs from what a fresh read of the delivered files shows; nothing indicates in-place modification. All scripts import `evaluate.py`/`baseline.py`/`data.py` via `importlib` from `source/starter-kit/` rather than copy-pasting and silently diverging.
- **Hash-key collision check (an integrity concern not explicitly on the checklist but found during review):** `profile_train_validation.py` encodes user–tag pairs as `user_id*256+tag` and user–video repeat lookups as `(user_id<<16)+video_id`. Verified against the raw data that `video_id` max is `7,582` (< 65,536) and tag tokens max at `68` (< 256), so neither encoding can collide across users. No leakage or correctness issue.

**No violations of the test-label/leakage rules were found anywhere in the reviewed code.**

## 7. Statistical Reliability Review

- **Seed counts:** Field ablations use 3 seeds (5 for the two identity-field removals, added specifically because the initial 3-seed result was the headline negative-redundancy claim and merited stronger support — a good instinct by the pre-audit agent). This is adequate for the large-effect findings (`tab` removal, static 13-field expansion) and self-consciously hedged (`INCONCLUSIVE`) for the small-effect ones (8-field static, learning rate). No result was "declared dead from one noisy run" — every negative claim in the Evidence Summary is backed by ≥3 seeds with reported population std.
- **Noisy effects correctly downgraded:** `dur_bucket` removal (`−0.000591 ± 0.000156`, consistent sign but below the project's 0.002 practical epsilon) is labeled `WEAK NEGATIVE`, not `STRONG NEGATIVE` — an appropriately conservative distinction between statistical significance and practical significance.
- **No overgeneralized negative conclusions found.** Every STRONG/WEAK NEGATIVE claim in PRE_AUDIT.md is scoped to "in this exact FM" / "the exact tested formulation" / "simple capacity scaling," consistent with `RULES.md` §8's requirement not to convert a weak negative into a blanket prohibition. This is worth calling out as a strength: the writing discipline throughout the document consistently avoids the "this doesn't work" → "this implementation didn't help" overreach the review brief warned against.
- **Experiments needing more repetition:** None urgently — the two genuinely marginal results (8-field static, learning-rate sweep) are already correctly labeled `INCONCLUSIVE` rather than pushed into a stronger class. If a later agent wants to resolve them, more seeds would be the direct fix, but nothing here mandates it.

## 8. Data Profile Consistency

`data_profile.md` and `PRE_AUDIT.md` were cross-checked table-by-table; all shared numbers matched exactly (activity-tier and list-length tables, controlled-FM table, overlap/repeat tables, history coverage tables) — both files were evidently generated from the same underlying run rather than drifting apart over time. The two corrections in Section 4 above applied to both files identically, since both had the same transcription error and the same scope gap.

No stale results or plots were found: `research/plots/daily_standard_profile.png` and `validation_list_length_users.png` are generated in the same script run (`profile_train_validation.py`) that produces the CSVs cited alongside them, from the same `daily` and `list_metrics` DataFrames.

## 9. Candidate constraints.md Review

| Candidate | Decision | Reason |
|---|---|---|
| 1 — Validation metric invariance | APPROVE | HARD FACT, independently reproduced exactly, cleanly scoped. |
| 2 — Metric headroom concentration (T4 / 6–10 lists) | APPROVE | HARD FACT under the reproduced FM; wording already correctly scopes this to "under the reproduced FM" rather than claiming it as a dataset-intrinsic property. |
| 3 — Entity vs. relationship overlap | APPROVE | HARD FACT, independently reproduced exactly. |
| 4 — Historical support differs by granularity | APPROVE | HARD FACT, independently reproduced exactly (interaction counts, tag repeat rate). |
| 5 — Auxiliary-signal density | APPROVE | HARD FACT, independently reproduced exactly (with the corrected `is_follow` figure folded in — does not change the qualitative claim). |
| 6 — Video/author redundancy in the exact FM | APPROVE | STRONG NEGATIVE evidence, 5 paired seeds all positive-direction, properly scoped to "the exact dual-ID formulation" / "this exact FM." |
| 7 — `tab` field importance | APPROVE | STRONG NEGATIVE against removal, very large effect (~30σ vs. seed noise), unambiguous. |
| 8 — Static feature stuffing | APPROVE | STRONG NEGATIVE, consistent with prior organizer evidence already in `constraints.md` (C5); this is an independent validation-only replication of that prior test-based result, which strengthens rather than merely repeats it. |
| 9 — FM dimension scaling | APPROVE | STRONG NEGATIVE against simple capacity scaling, consistent with prior organizer evidence already in `constraints.md` (C6); same replication value as Candidate 8. |
| 10 — Video-statistics timing quarantine | APPROVE WITH REWORDING | The underlying fact (undocumented aggregation cutoff) is solid and important as a guardrail, similar in kind to the existing C3 leakage rule. The current wording ("Do not treat... as causally safe until...") is phrased as an instruction/strategy rather than a fact; reword to state the documentation gap as the fact and let the "do not treat as safe" consequence follow implicitly, consistent with how C3 is phrased. |
| 11 — Random-log temporal risk | APPROVE | HARD FACT, independently reproduced exactly. |
| 12 — Engineering readiness | APPROVE WITH REWORDING | HARD FACT, but the original wording undercounted the scope (see Issue R-02); use the corrected 15-file/full-layer wording below. |

## 10. Recommended Wording for Approved Constraints

For candidates approved as-is (1–9, 11), the pre-audit's own "Recommended wording" text in PRE_AUDIT.md is sound and can be copied verbatim into `constraints.md`. For the two reworded candidates:

**Candidate 10 (reworded):**
> The official video-statistics file (`video_features_statistic_pure.csv`) is documented as a per-day, per-scenario average computed "over one month," but the exact calendar window or cutoff relative to a given impression is not disclosed in available documentation. Its causal/leakage safety for the April 22–28 validation period (or any scored period) is therefore not established by local materials alone.

**Candidate 12 (reworded):**
> As of the pre-audit, all 15 files under `harness/`, `pipeline/`, and `agent/` (covering execution, guards, caching, diagnostics, logging, scoring, submission, data adaptation, feature construction, training, and the coder/controller/governor/proposer/reflector agent loop) contain zero executable non-comment lines; `reports/`, `submissions/`, `runlogs/`, and `tests/` contain no files. Runtime primitives needed to build this layer (bounded subprocesses, recursive Windows process-tree termination, syntax-error recovery, NaN/Inf detection, content-fingerprinted caching) were separately probed and function correctly.

## 11. Remaining Questions for the Autonomous Agent

These are deliberately left unresolved, per the review philosophy of protecting the agent's own research judgment (`PROBLEM.md` §10, `RULES.md` §12):

- Which loss function (pointwise vs. pairwise/BPR vs. listwise) best matches the ranking-based metrics, and by how much.
- Whether and how to incorporate user history (exact-item, exact-author, or tag-level), given the sharply different support levels measured in A03/F01.
- Whether any auxiliary feedback signal produces positive transfer to `long_view` in a multi-task setup, and which architecture (shared-bottom, MMoE, PLE) to use if so.
- How to handle cold users (1.89% of validation) and the activity-tier structure (T1 vs. T4) in a training or architecture design, if at all.
- Whether video statistics can be made causally defensible (e.g., by an explicit, justified cutoff assumption) and, if so, whether they add value beyond `user_id`×`video_id` interactions.
- Whether a field-redundancy result observed in the exact baseline FM (Candidate 6, C02) transfers to a different model family.
- Whether temporal/recency weighting is worth pursuing given the non-uniform drift found in H01.
- What role, if any, the random-exposure log should play beyond the diagnostic uses already established as safe.
- What the single highest-expected-value next experiment is, under the 50-iteration / 6-hour budget.

## 12. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

The pre-audit's measurements are accurate (verified independently, not merely re-read), its evidence-language discipline matches `RULES.md` closely, no leakage was found anywhere in the reviewed code, and the two issues found were minor and have been corrected in place with transparent review-correction notes preserving the original claims. The candidate findings in Section 9 are ready for a human to selectively promote into `context/constraints.md`; two of the twelve need the reworded phrasing in Section 10 before promotion, the other ten can be copied as originally drafted. Separately from the research evidence itself, the engineering-readiness finding (now corrected to its full 15-file scope) is a real prerequisite gap that should be addressed before an autonomous `RUN_START`, independent of the constraints-review process.
