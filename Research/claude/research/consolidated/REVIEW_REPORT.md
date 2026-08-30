# Consolidated Pre-Audit Review Report

> This report reviews the *consolidation* of three independently-produced KuaiRand-Pure pre-audit
> sets (each already independently reviewed by its own producer) into one merged
> `research/PRE_AUDIT.md` / `data_profile.md`. It does not re-run any experiment; it audits the
> merge itself for completeness, correctness, and internal consistency. See
> `research/consolidated/MERGE_WORKLOG.md` for the full 50-row crosswalk this report summarizes.

## 1. Executive Summary

Three audit sets — Claude (28 original investigations + 1 post-review extension), Gemini (6
original investigations + 3 added on review), and GPT (22 investigations) — were merged into one
consolidated PRE_AUDIT.md, data_profile.md, and this report. All three source sets were themselves
already post-review (each source PRE_AUDIT.md embeds its own producer's "Review correction:" notes),
so this merge is a second-order consistency pass across three already-vetted bodies of work, not a
first review of raw claims.

- **~50 topic areas** were cross-walked (MERGE_WORKLOG.md). Of these, roughly **30 were genuine
  duplicate findings** across two or three audits (several matching to 3-5 decimal places despite
  fully independent codebases — the strongest form of cross-validation available in this exercise),
  **7 involved a real scope difference that could be mistaken for a conflict** (tag-string vs.
  parsed-tag-token granularity; full-video-table vs. observed-only entity scope; different
  activity-tier bucket boundaries across all three audits; different cache-speedup baselines), and
  **13 were present in only one audit** and are retained as single-source findings with clear
  attribution.
- **Zero unresolved *numerical* conflicts remain** — every case where two sources reported
  different numbers for what looked like the same quantity was traceable to a documented scope
  difference (see §4). No case required arbitrarily picking one source's number over another's.
- **Five prior reviewer corrections are preserved** without regression: Claude's evaluation-period-
  contamination removal, Claude's GAUC-weight-denominator fix, Gemini's random-log overlap count
  fix (759→702, independently cross-validated by GPT's separately-computed 702), Gemini's
  video-statistic mean-value fixes (`like_cnt` 158→230.75, plus two previously-unmeasured fields),
  and GPT's `is_follow`-prevalence and engineering-scaffold-scope fixes (both independently
  corroborated by the other two audits — see §3).
- **The merged package is trustworthy for human constraint review**, with the caveat that several
  findings are legitimately single-audit (engineering-scaffold status, GPT-only; joint
  activity×list-length analysis, Claude-only; raw video-stat means, Gemini-only) and should be read
  as such rather than as independently triplicate-verified.

**Readiness: READY FOR HUMAN CONSTRAINT REVIEW.**

## 2. Source Sets Reviewed

| Label | PRE_AUDIT.md | REVIEW_REPORT.md | data_profile.md | Investigations (orig. + review-added) | Own readiness decision |
|---|---:|---:|---:|---:|---|
| Claude | 1,562 lines | 231 lines | 511 lines | 28 + 1 post-review extension (B06) | READY FOR HUMAN CONSTRAINT REVIEW |
| Gemini | 485 lines | 368 lines | 315 lines | 6 + 3 added on review (C01/D01/H01) | READY FOR HUMAN CONSTRAINT REVIEW |
| GPT | 1,397 lines | 183 lines | 431 lines | 22 | READY FOR HUMAN CONSTRAINT REVIEW |

All nine files were read in full for this consolidation, along with `context/PROBLEM.md`,
`context/RULES.md`, `context/DATA_GUIDE.md`, `context/constraints.md`, and `context/references.md`.

## 3. Major Corrections Preserved

These were review corrections *within* one of the three source audits, made before this
consolidation began. This merge verified each is still reflected correctly in the merged output
and did not regress any of them.

1. **Test/evaluation contamination removed (Claude).** The original (pre-review) Claude pre-audit
   locally scored standard-test labels once via the unmodified official baseline script, and
   `phase_i_random_log.py` summarized evaluation-period random-log outcomes — both violations of
   the train+validation-only rule, though neither fed into any retained conclusion. Claude's own
   review removed the test-derived values, hardened the loaders (`common.load_valid_log()` /
   `common.load_random_log()` now check date before materializing complete rows), and regenerated
   the affected artifacts. **Verified in this merge:** no evaluation/test-derived label, rate, or
   metric value appears anywhere in the merged PRE_AUDIT.md or data_profile.md.
2. **GAUC weight-share denominator (Claude).** Claude's original activity-tier and list-length
   "GAUC weight share" tables divided bucket positives by *all* validation positives, including
   uniform-label users the official evaluator excludes entirely (the giveaway: length-1 users, who
   can never enter GAUC, were originally assigned 4.06% GAUC weight). Corrected to the official
   denominator — 34,592 positive rows belonging to mixed-label users only. **Verified in this
   merge:** every GAUC-weight-share figure carried into the merged PRE_AUDIT §4 uses the corrected
   denominator; GPT's independently-computed equivalent tables used the correct denominator from
   the start and serve as external corroboration that Claude's corrected numbers (not its original
   ones) are the right ones to trust.
3. **Random-log overlap count (Gemini), independently cross-validated by GPT.** Gemini's original
   `GI_analysis.py` built its "random UV pairs also in standard logs" statistic from the
   *unfiltered* second standard log (spanning validation+test dates) — a rule-compliance bug
   touching test-period row identities (not labels). Corrected from 759 to 702 pairs. **Verified in
   this merge:** GPT's fully independent implementation of the same comparison also arrives at
   exactly 702, which is strong evidence the corrected value (not the original 759) is right —
   this cross-audit agreement was not available to either Gemini or GPT individually.
4. **Video-statistic mean values (Gemini).** `like_cnt` mean was originally mis-stated as 158;
   direct recomputation gives 230.75. `long_time_play_cnt` and `comment_cnt` means were originally
   left as unmeasured placeholders; filled in as 3,687 and 12.93 respectively. **Verified in this
   merge:** the merged data_profile.md §15 carries only the corrected values, with the original
   wrong value noted for context per the merge instructions.
5. **`is_follow` prevalence and engineering-scaffold scope (GPT).** GPT's own review caught a
   rounding-transcription error (0.131%→0.130% for validation `is_follow` prevalence) and an
   undercount in its engineering-readiness probe (7 files inventoried, 15 actually empty).
   **Verified in this merge, with new cross-audit corroboration not available to GPT's own
   reviewer:** Claude and Gemini's independently-measured `is_follow` figures were already 0.130%
   in their original work, confirming GPT's corrected (not original) value is right. The
   scaffold-scope fix has no cross-audit corroboration available (Claude and Gemini did not probe
   this at all) and is carried forward as a GPT-only finding.

No new corrections were required by this merge itself. The merge did surface and resolve seven
scope-ambiguities that could plausibly have been mistaken for conflicts by a less careful merge
(§4) — none of these required discarding or overriding any source's number.

## 4. Conflicts Resolved

Every case below was investigated per the merge protocol (identify what differs → check for a
scope/definition difference → inspect available evidence → classify). In every case, the
resolution was **"different scope, not a true conflict"** — no case required picking one source's
number as correct and discarding another's as wrong.

| # | Apparent conflict | Resolution |
|---|---|---|
| 1 | User-tag PAIR overlap: 68.14% (Claude, Gemini) vs. 71.913% (GPT) | GPT parses the `tag` field into comma-separated tokens (documenting 46 distinct tokens vs. 110 raw strings elsewhere in its own audit); Claude/Gemini treat the raw tag string as one opaque category. The looser (any-shared-token) definition mechanically yields a higher overlap rate. Both are correct under their own stated definition; PRE_AUDIT §3.2 presents both, explicitly labeled. |
| 2 | User-tag repeat rate, train: 51.77% (Claude) vs. 55.250% (GPT) | Same tag-string-vs-token scope difference as #1. |
| 3 | Row-level "same tag seen before": 73.19% (Claude) vs. 78.413% (GPT) | Same tag-string-vs-token scope difference as #1. |
| 4 | Author redundancy denominator: full video_basic table (86.96%, all 3) vs. observed-in-logs-only (87.05-87.07%, Claude/GPT) | Two legitimately different populations (all authors in the shipped file vs. only authors whose videos appear in train/validation logs). Both retained, explicitly scoped, in PRE_AUDIT §3.3. |
| 5 | Videos/author max: 26 (Claude, full table) vs. 24 (GPT, observed-only) | Direct consequence of #4's scope difference — the most prolific authors in the full catalog are not all represented in train/validation logs. |
| 6 | Activity-tier GAUC weight share for the top tier: 34.79% (Claude) vs. 40.01% (GPT) | Three different tier-boundary schemes across all three audits (Claude: quartiles of users-with-≥1-train-row at edges 17/36/65; GPT: quartiles of all train users at edges 13/31/59; Gemini: fixed absolute thresholds). These are not the same partition of the user population and their percentages are not interchangeable. PRE_AUDIT §4.3 presents all three schemes side by side without merging the numbers, and explicitly states the shared qualitative conclusion (concentration in the top tier) separately from the non-comparable exact percentages. |
| 7 | Cache speedup: 263x (Claude) vs. 72.8x (GPT) | Different baselines: Claude compares cache-read against full re-**encoding** time; GPT compares against raw CSV **load** time (and separately reports a more conservative "effective" 2.27x that includes fingerprint-computation overhead). Not a disagreement about the cache's actual read time (0.018s vs. 0.043s, plausibly just different machines) — a difference in what each is being compared to. All figures retained with their comparison basis stated explicitly. |

Two further near-matches were noted as **minor, unresolved discrepancies** rather than forced into
either "duplicate" or "scope difference": Gemini's row-level same-video/same-author repeat-coverage
figures (1.58%/3.27%) are close to, but not identical to, Claude/GPT's matching figures
(1.62%/3.38% and 1.624%/3.381%), and Claude's tag-cardinality count (111) differs by one from
Gemini/GPT's matching count (110). Neither discrepancy is large enough to affect any downstream
interpretation, and no source documents a scope difference that would explain them, so both are
flagged as open (§5) rather than resolved.

## 5. Conflicts Still Inconclusive

1. **Row-level same-video (1.58% vs. 1.62%/1.624%) and same-author (3.27% vs. 3.38%/3.381%) repeat
   coverage** — Gemini's figures are consistently slightly lower than Claude/GPT's closely-matching
   figures, by amounts too small to matter for any conclusion drawn in this package, but with no
   documented reason for the gap. Flagged as `MINOR DISCREPANCY (unresolved)` in data_profile.md
   §8 rather than silently averaged or dropped.
2. **Tag cardinality: 110 (Gemini, GPT) vs. 111 (Claude).** Off by one, exact scope of Claude's
   count not fully specified in its source material. Flagged as `MINOR DISCREPANCY (unresolved)`
   in data_profile.md §3.
3. **Whether validation "resembles late training more than early training"** — this is not a
   cross-audit conflict (Claude and GPT's own methods each independently concluded the answer is
   mixed/inconclusive) but is preserved as a substantive open question because it is directly
   relevant to any future recency-weighting hypothesis (PRE_AUDIT §10.2).
4. **Whether `psutil`-based recursive process-tree termination fixes the exact Windows
   subprocess-timeout failure Claude found** — Claude and GPT tested different conditions (Claude:
   adversarial, unmanaged grandchild inheriting stdio; GPT: simpler, no grandchild), so GPT's
   passing result is suggestive but not a direct test of a fix for Claude's failure mode (PRE_AUDIT
   §12.3).

## 6. Duplicate Findings Consolidated

Approximately 30 of the ~50 crosswalked topics were genuine duplicates (the same quantity, same
scope, reported by 2 or 3 sources). Selected high-value examples, chosen because independent,
separately-coded implementations matching to several decimal places is the strongest evidence
available in this exercise:

- Baseline reproduction (GAUC/nDCG@5/primary) — GPT and Claude match to 3-4 decimal places on an
  independently-built loader/encoder/training loop.
- Uniform-label validation-user composition (30.32%/11.90%/57.78%) — exact match, 3/3.
- Oracle nDCG@5 by list length — exact match, 3/3, and independently cross-validated by Gemini
  against the organizer's own published `baseline_scores.json`.
- Field-ablation deltas for `tab`, `dur_bucket`, `video_id`, `author_id` — Claude and GPT match to
  4-5 decimal places despite fully independent codebases; the `video_id`/`author_id` result in
  particular is corroborated by 10 combined paired-seed runs (5 from each audit), all pointing the
  same direction.
- Static-feature-expansion and FM-dimension-sweep deltas — Claude and GPT match to ~5 decimal
  places, independently re-verifying organizer evidence already in `constraints.md` (C5/C6) on the
  validation split rather than merely re-quoting the organizer's test-split numbers.
- Feedback-signal prevalence and same-row correlation with `long_view` — exact/near-exact match,
  3/3, including cross-validation of GPT's own `is_follow` correction (§3, item 5 above).
- Historical-availability coverage (≥1/≥5/≥10 prior interactions; median 35) — exact match, 3/3.
- The video-statistics aggregation-window/causal-safety question — not a numeric duplicate, but a
  **convergent qualitative finding reached independently by three different evidence types**
  (Claude: arithmetic ratio; GPT: official documentation citation; Gemini: independent qualitative
  flag) — the single strongest three-way corroboration in the merged package precisely because the
  three methods are so different from each other.
- Random-exposure log date/row counts and full-log pair-overlap count (702) — exact/matching, and
  in the pair-overlap case a Gemini reviewer correction independently cross-validated by GPT.

Full duplicate-vs-scope-difference-vs-single-source classification for every topic is in
`MERGE_WORKLOG.md`.

## 7. Evidence Classification Changes

This merge maps each source's evidence label onto the fuller vocabulary specified for the merged
package (`HARD FACT`, `STRONG POSITIVE/NEGATIVE EVIDENCE`, `WEAK POSITIVE/NEGATIVE EVIDENCE`,
`INCONCLUSIVE`, `ENGINEERING CONSTRAINT`, `INVALID/FORBIDDEN`) without changing any finding's
strength beyond what its own source audit already established. No source finding was upgraded
(e.g. from WEAK to STRONG) during this merge. The mappings applied:

- All three sources' `HARD FACT` → `HARD FACT` (unchanged).
- All three sources' `STRONG NEGATIVE` → `STRONG NEGATIVE EVIDENCE` (relabeled for vocabulary
  consistency only).
- All three sources' `WEAK NEGATIVE` → `WEAK NEGATIVE EVIDENCE` (relabeled only). This includes
  Claude's C01 video/author-redundancy finding, which Claude's own review already scoped narrowly
  ("WEAK NEGATIVE for this exact configuration") — the merge preserves that conservative scoping
  rather than promoting it to STRONG despite 10 combined paired-seed runs across two audits now
  supporting it, because all 10 runs still test only one exact field-set/model/objective
  combination, which is exactly the boundary RULES.md §8 asks audits to respect.
- All three sources' `INCONCLUSIVE` → `INCONCLUSIVE` (unchanged).
- Windows subprocess-timeout and engineering-scaffold findings (previously informally described as
  "HARD FACT (run-specific)" by their sources) → `ENGINEERING CONSTRAINT`, a more precise fit
  since the fuller vocabulary distinguishes environment/tooling facts from dataset facts.
- The random-exposure log's "not for unfiltered training use" conclusion (previously `STRONG
  NEGATIVE` in Gemini's audit) → `STRONG NEGATIVE EVIDENCE`, retained at the same strength; this
  merge did not find grounds to downgrade or upgrade it.

No finding required downgrading. This is consistent with each source audit's own review report,
none of which needed to downgrade a strong claim to weak, or weak to inconclusive, during their
own internal reviews either — the one *near*-miss (Gemini's B01 T4-activity confound with list
length) was already correctly caught and qualified by Gemini's own review before this merge began,
and is preserved as-is in PRE_AUDIT §4.3.

## 8. Data Profile Consistency Check

Every numeric table in the merged `data_profile.md` was cross-checked against the corresponding
finding in the merged `PRE_AUDIT.md`:

- All cardinality, overlap, redundancy, missingness, uniform-label, list-length, feedback-
  prevalence, and historical-availability figures in `data_profile.md` §3-§13 have a matching,
  correctly-scoped statement in `PRE_AUDIT.md` §2-§8. No stale or orphaned numbers were found.
- Both files consistently mark the three activity-tier bucket schemes as non-comparable
  (PRE_AUDIT §4.3 / data_profile §11) rather than presenting one as canonical.
- Both files consistently present the two tag-overlap granularities (PRE_AUDIT §3.2 / data_profile
  §5) and the two random-log overlap comparisons (PRE_AUDIT §11.2 / data_profile §16) as distinct,
  correctly-scoped quantities rather than conflating them.
- The two population-conflation corrections (Gemini's train-activity-median fix, PRE_AUDIT §8.2 /
  data_profile §13; and Gemini's `like_cnt`-mean fix, PRE_AUDIT §9.5 / data_profile §15) are stated
  identically in both files.
- No evaluation/test-derived value appears in either file.
- `MERGE_WORKLOG.md`'s crosswalk table was checked row-by-row against both final files; every row's
  "final decision" is reflected in both `PRE_AUDIT.md` and `data_profile.md` consistently.

No inconsistency was found between the two files as finalized.

## 9. Candidate Findings for constraints.md

| Finding | Final Classification | Recommendation | Reason |
|---|---|---|---|
| 1. Train-log 04-08 gap + volume decay | HARD FACT | APPROVE | Triplicate exact match, deterministic, no strategy directive in the recommended wording |
| 2. Metric headroom concentration (list-length + activity-tier + joint) | HARD FACT | APPROVE WITH CARE | List-length figure is exact-match HARD FACT; activity-tier figures must be quoted with their specific tier scheme attached, never as a single bare percentage |
| 3. Video/author identity redundancy in the exact baseline FM | WEAK NEGATIVE EVIDENCE | APPROVE WITH REWORDING | Strong cross-audit reproduction (10/10 paired seeds positive) but explicitly narrow in scope (this exact FM/field-set/objective only); wording must retain that scope, per RULES.md §8 |
| 4. `tab` importance + static-feature/capacity dead ends (strengthens C5/C6) | STRONG NEGATIVE EVIDENCE | APPROVE | Independently cross-validated by two separately-coded implementations on the validation split; strengthens rather than duplicates existing C5/C6 |
| 5. Same-row feedback correlation with `long_view` | HARD FACT | DO NOT PROMOTE | Correct and triplicate-verified, but redundant with existing constraint C3 — adds dataset-specific numbers to an already-established rule rather than new decision-relevant information |
| 6. Video-statistics aggregation-window/causal-safety gap | HARD FACT (documentation, ratio) + INCONCLUSIVE (safety) | APPROVE WITH REWORDING | Three-way independent convergence is unusually strong evidence for an "unresolved" finding; word as a documented gap + its consequence, not as an instruction ("do not use..."), consistent with how existing C3 is phrased |
| 7. Windows subprocess-timeout risk + partial mitigation evidence | ENGINEERING CONSTRAINT | APPROVE WITH REWORDING | Verified failure condition (Claude) and a verified-but-not-directly-matched mitigation (GPT); wording must not overstate the mitigation as "solving" the exact failure Claude found, since they were not tested together |
| 8. Engineering-readiness gap (15 scaffold-only files) | HARD FACT | APPROVE | Single-source (GPT) but unambiguous, reviewer-corrected, and highly decision-relevant for anything happening before `RUN_START` |
| 9. Population-conflation caution (history-availability populations) | HARD FACT | APPROVE | Directly useful as a standing caution against a specific, already-observed class of error (a population conflation caught by one source's own review); low-risk to promote since it constrains interpretation rather than asserting a new substantive claim |

No candidate in this package required a `REQUIRES MORE EVIDENCE` disposition — every candidate
above is either independently triplicate/duplicate-verified or is a clearly-scoped single-source
finding with no internal red flags.

## 10. Remaining Questions for Autonomous Agent

Preserved from the three source audits' own "Questions the Autonomous Agent Should Resolve Itself"
sections, deduplicated:

- Which multi-task objective/architecture, if any, avoids negative transfer while using
  `is_click`/`play_time_ms` or sparser feedback signals as auxiliary targets.
- Which historical-feature representation (aggregate rate, sequence model, tag-attention) provides
  the highest lift, and at what granularity (video/author/tag, including which tag definition —
  raw string or parsed token, §3.2).
- Whether a pairwise or listwise loss outperforms pointwise logloss given the pointwise/ranking-
  metric mismatch (§13) and the metric-headroom concentration found in §4-§5.
- Whether the video/author-redundancy field-ablation result (§6.2, Candidate 3) generalizes beyond
  the exact tested pointwise FM to any other model family or objective.
- Whether and how to use `video_features_statistic_pure.csv` given its unresolved aggregation-
  window uncertainty (§9.3, Candidate 6) — an explicit risk/benefit judgment left open.
- Whether temporal/recency weighting is worth pursuing given §10.1's volume decay and §10.2's mixed
  early/late-train resemblance evidence.
- Whether the random-exposure log's validation-period slice (§11.3) is worth incorporating as a
  diagnostic/counterfactual set.
- What experiment has the highest expected information gain under the 50-iteration / 6-hour budget,
  given everything above and the engineering-readiness gap (§12.5, Candidate 8) that likely must be
  closed first.

## 11. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

All three source audits independently reached this same decision for their own work, and this
consolidation found no basis to reverse it: no test-label leakage in any merged artifact, all
major reviewer corrections preserved and in two cases independently cross-validated across audits
during this merge (§3), every apparent cross-audit conflict traced to a documented scope
difference rather than requiring an arbitrary choice (§4), and the two files that make up the
merged evidence package agree with each other in full (§8). The nine candidate findings in §9 are
ready for a human to decide on promotion into `context/constraints.md`, using the "APPROVE WITH
REWORDING" guidance above where given. Nothing found during this consolidation requires re-opening
any of the three source audits' own experimental work.
