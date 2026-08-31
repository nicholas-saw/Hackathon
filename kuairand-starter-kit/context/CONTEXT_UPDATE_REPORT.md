# Context Update Report

> Reconciliation of the six-file context package against the final reviewed audit
> evidence. Completed 2026-08-30. No new pre-audit, no new model search, no new
> hypotheses, no manually chosen research direction.
>
> Backup of the pre-update files: `backup/context_before_audit_update/`.

## 1. Sources Used

Read in authority order before any edit was made.

| Authority | Source |
|---|---|
| 1. Official specification | `source/starter-kit/` — `README.md`, `evaluate.py`, `data.py`, `baseline.py`, `baseline_scores.json`, `submit.py`, `ablation_features.py` |
| 1. Official data | `source/KuaiRand-Pure/data/` — all six CSVs (headers, row counts, date coverage, and selected structural values re-verified directly) |
| 2. Final reviewer-corrected findings | `research/consolidated/REVIEW_REPORT.md` |
| 3. Reproduced artifacts | `research/experiment_results/` (baseline reproduction, controlled FM runs, official source notes, inventories) |
| 4. Consolidated pre-audit | `research/consolidated/PRE_AUDIT.md` |
| 4. Merge ledger | `research/consolidated/MERGE_WORKLOG.md` (20-item conflict ledger) |
| 5. Prior context | the six target files, now superseded where they conflicted |

`research/PRE_AUDIT.md`, `research/REVIEW_REPORT.md`, `research/data_profile.md`, and
`research/MERGE_WORKLOG.md` were byte-identical to their `research/consolidated/`
counterparts at the start of this update; the consolidated copies were treated as
authoritative and were **not modified**.

### Independent re-verification performed during this update

Structural facts only — no model was trained, no evaluation label was read.

| Fact | Result |
|---|---|
| Train / validation / evaluation row counts by `date` | 1,141,112 / 124,909 / 170,588 — confirmed |
| Train dates present | 13 dates, 04-09..21; **zero rows on 04-08** — confirmed |
| Daily volume peak / final day | 278,835 on 04-11 / 20,021 on 04-21 (13.9x) — confirmed |
| Early / late train period rows | 891,418 / 249,694 — confirmed |
| Random log by date | 1,186,059 total; 288,338 validation-window; 897,721 evaluation-window — confirmed |
| Side-table row counts | user 27,285; video basic 7,583; video statistic 7,583 — confirmed |
| Author structure | 6,510 authors; 5,661 with exactly one video (86.9585%); max 26 — confirmed |
| `visible_status` | constant 0 across all 7,583 rows — confirmed |
| `is_rand` | constant 0 in standard logs, constant 1 in random log — new structural fact |
| `tab` | 15 values (0–14); tab 1 is ~73% of train rows — confirmed / extended |
| Video-statistic means | `like_cnt` 230.75 (median 57.54); `long_time_play_cnt` 3,687 (978); `comment_cnt` 12.93 (2.46) — confirmed |
| `ablation_features.py` field counts | 5 / **8** / **13**, not 9 / 14 — confirmed by reading the code |
| Scaffold inventory | 15 files under `harness/`, `pipeline/`, `agent/`, all with zero executable lines — confirmed |
| Stay-time zero rates | `profile_stay_time` 99.994% (valid) / 99.989% (train); `comment_stay_time` 95.542% / 94.564% — measured to resolve a wording gap |

One stale artifact was identified and **not** used: `research/experiment_results/video_feature_inventory.csv`
profiles 7,545 video rows (not 7,583) and reports 6,487 authors and `like_cnt` mean
231.84. The reviewer-corrected full-file values (7,583 rows, 6,510 authors, 230.75) are
the ones carried into the context package, and direct measurement confirms them.

---

## 2. PROBLEM.md

**Changes made.**

1. **Oracle note corrected for split.** The file previously stated a single oracle
   primary "near 0.8645", unlabelled. 0.8645 is the **test** oracle; the **validation**
   oracle is 0.8484. Replaced with a split-labelled table and an explicit instruction to
   judge development progress against 0.8484.
2. **Seed standard deviation scoped.** "Approximately 0.0008" is now attributed as the
   organizer's std of the **test** GAUC / nDCG@5 / primary across 5 seeds, and linked to
   the convergence rule.
3. **Published reference ladder added.** The random and item-popularity rows from
   `baseline_scores.json` were missing; the section claimed to describe the official
   baseline but carried only the FM row.
4. **Train boundary clarified.** Added a note that the official train window opens
   2022-04-08 but the raw file has no rows on that date (13 dates, 04-09..21), with an
   explicit warning not to "fix" it by moving a boundary.
5. **Convergence rationale recorded.** `epsilon = 0.002` is roughly 2.5x the published
   0.0008; convergence is judged on validation only.
6. **Official evaluation-split composition added** (23,875 users; 27.1% all-negative,
   9.2% all-positive, 63.7% discriminative) — labelled as published reference material,
   explicitly not to be re-derived locally.
7. **Official submission contract added** — `row_id` ordering, finite-score requirement,
   and the official reason `row_id` is mandatory (3.06% duplicated `(user_id, video_id)`
   pairs in the evaluation split, up to 12 repeats).
8. **Official scale block** now names the side tables that supply the "27K users / 7.6K
   videos" figures (27,285 / 7,583) and states that row counts come from the `date`
   column alone.

**Why.** Items 1 and 2 were genuine inaccuracies of official fact — the unlabelled
0.8645 would have had an agent measuring validation progress against a test-split
ceiling. Items 3–8 are completions or terminology clarifications within sections that
already claimed to hold that official information. **No empirical pre-audit finding was
added to this file.**

---

## 3. RULES.md

**Changes made.**

1. **§1 rewritten as "Test / Evaluation Isolation."** Development is now stated as
   train + validation only, covering the autonomous run, diagnostics, and any audit.
   Added: a prohibition on summarising, profiling, or plotting evaluation-period rows —
   outcomes, features, *or identities*; the two permitted exceptions (date-only row
   counts, published official numbers); an explicit carve-out that applying a frozen
   model to evaluation features for submission is not an inspection; and the requirement
   that accidental reads be logged **and** the derived results discarded.
2. **§1a added — random-exposure log scope.** A table splitting the file into the
   validation window (04-22..28, 288,338 rows, outcomes permitted) and the evaluation
   window (04-29..05-08, 897,721 rows, date-only). Requires filtering by `date` *before*
   materialising any other column. Notes that using the validation slice for training or
   selection is a research decision requiring an explicit logged leakage argument, not a
   rule violation.
3. **§2a added — GAUC semantics as a guardrail.** Any derived per-segment metric,
   weight share, or reweighting must follow `evaluate.py`: only mixed-label users
   contribute; weights are positive counts among contributing users only (34,592 in
   validation); nDCG is equal-user weighted; the two weightings must never be
   multiplied or blended; the 0.5 empty-denominator fallback is not a score.
4. **§2 extended** with the no-wrapping/no-shadowing rule for `evaluate` / `auc` /
   `ndcg_at_k`, and a ban on special-casing an ID because it appears in valid/eval.
5. **§3 rewritten as "Source Integrity."** `source/` is read-only, covering both the
   raw dataset and the official kit; no temporary patches; adapters go in `pipeline/`.
   Added the explicit run-boundary map (read-only `source/`, `context/`, `harness/`;
   editable `pipeline/`; agent-created `runlogs/`, `submissions/`, `reports/`) and the
   statement that the agent never writes `constraints.md`.
6. **§4 strengthened.** The history ordering is now a numbered three-step sequence, plus
   two additions: tied timestamps are **not** predecessors (5.60% of validation rows sit
   in non-unique user/timestamp groups), and encoders/edges/priors must be fitted on the
   past only, never on the period being scored.
7. **§8 evidence vocabulary aligned.** Expanded from four classes to the eight used in
   the consolidated package (adds STRONG/WEAK POSITIVE, ENGINEERING CONSTRAINT,
   INVALID / FORBIDDEN) with rules on scope, single runs, and sub-epsilon deltas.
8. **§13 added** — how to read `constraints.md` and `data_profile.md`: negative results
   bind only the exact formulation tested, INCONCLUSIVE is an invitation to measure, and
   entry order is not a priority ranking.

**Why.** Items 1–3 and 6 directly encode the two review corrections and the tie-semantics
correction. Item 7 was required for cross-file consistency: the constraints file uses
classifications the rules file did not define. **No empirical finding was converted into
a hard rule** — for example, the weak result against the dual-ID FM formulation stayed in
`constraints.md` as evidence and appears nowhere in RULES.md.

---

## 4. DATA_GUIDE.md

**Changes made.** Rewritten as a verified description of the data.

- **Exact schemas** for all six files, with column counts and row counts, verified
  against the CSVs.
- **File-vs-split warning**: `log_standard_4_22_to_5_08_pure.csv` holds both the
  validation (124,909) and evaluation (170,588) windows — split by `date`, never by file.
- **Column roles table** separating pre-impression context from post-impression outcomes,
  with cardinality context (`tab`: 15 values, ~73% concentrated in tab 1).
- **`is_rand` documented** as constant 0 in the standard logs and constant 1 in the
  random log — it distinguishes files and carries no within-file information.
- **Static-field clarification**: all user-table fields are static per user (with
  raw/`_range` duplicate pairs), all video-basic fields static per video.
- **`visible_status` = cardinality 1** — carries no information.
- **§2.4 added**: what the official `data.py` actually exposes (6 columns + joined
  `author_id`); every other column requires an independent CSV read from the pipeline.
- **Tag semantics**: raw tag string vs parsed tokens documented as two
  non-interchangeable representations, with the requirement to state which was used.
- **Video-statistic aggregation uncertainty** stated as four numbered facts (averages not
  totals; undocumented endpoints; undocumented population; 54 near-redundant field
  pairs), with the conclusion pointed at C17 — explicitly "neither a settled ban nor a
  clearance".
- **Exact random-log date coverage table** (288,338 / 897,721) with the permitted scope.
- **`evaluate.py` semantics** summarised where an implementer will read them.
- **`ablation_features.py` field counts corrected** to 5 / 8 / 13, with a note that the
  script's own printed labels and the older "9/14" references are wrong.
- Author/video coupling is described structurally with a pointer to C12 for the
  quantitative finding — no "therefore remove `author_id`".

**Why.** The previous version was a rough map with several "the pre-audit should measure"
placeholders for questions that are now answered. Structural facts an agent needs to
interpret a source were promoted in; experiment results were kept out.

---

## 5. constraints.md

Rebuilt from the reviewed evidence. **6 entries before → 21 active entries after**
(23 IDs allocated; C16 and C17 were retired by human decision after the first pass —
see *Retired* below; C23 was absorbed from the starter-kit package during deployment —
see §13). IDs C1–C6 retain their prior meanings so that existing references
in `PRE_AUDIT.md` and `REVIEW_REPORT.md` (which cite "constraints.md C5 / C6") still
resolve, and retired IDs are never reused.

### Added (15)

| ID | Finding | Classification | Reviewer support |
|---|---|---|---|
| C7 | Evaluation-period information is not development evidence | INVALID / FORBIDDEN | §3 contamination removal, §7, §9 APPROVE |
| C8 | Official baseline reproduces locally; noise scale is 0.0008, epsilon 0.002 | HARD FACT | §3 (published std retained as safer reference) |
| C9 | Validation label composition, GAUC eligibility, oracle 0.848393 | HARD FACT | §9 APPROVE |
| C10 | Corrected GAUC denominator (34,592) and metric/headroom decomposition | HARD FACT (scoped) | §3 correction, §9 APPROVE WITH REWORDING |
| C11 | Scale, side-table coverage, missingness, activity/list distributions | HARD FACT | §3 (median 31 correction), §8 |
| C12 | Warm entities vs novel pairs, incl. author/video redundancy | HARD FACT | §4 scope split, §9 APPROVE |
| C13 | Strictly prior history coverage by granularity | HARD FACT | §3 (81.57% correction), §9 APPROVE |
| C14 | Temporal volume and multidimensional period shift | HARD FACT (components) | §5, §9 APPROVE WITH REWORDING |
| C15 | Removing `tab` from the official FM | STRONG NEGATIVE EVIDENCE | §9 APPROVE WITH REWORDING |
| C18 | Random-log permitted scope and validation-slice structure | HARD FACT (+ C7) | §3, §9 APPROVE |
| C19 | Bare Windows subprocess timeout did not bound the tested tree | ENGINEERING CONSTRAINT | §7, §9 APPROVE WITH REWORDING |
| C20 | Baseline runtime and caching observations | ENGINEERING CONSTRAINT | §4 (run-specific) |
| C21 | Submission and scoring validity contract | ENGINEERING CONSTRAINT | §8, official `submit.py` |
| C22 | 15 comment-only scaffold files; empty subdirectories | ENGINEERING CONSTRAINT | §3, §9 APPROVE |

| C23 | Per-user monotone transforms and global calibration are no-ops | HARD FACT | Absorbed from the `kuairand-starter-kit` package (its C2); mathematical consequence of C1 |

*(C23's ID is out of sequence within section 1 because constraint IDs are append-only.)*

### Retired (2) — removed on your instruction after the first pass

| ID | Was | Where the evidence lives now | Reason |
|---|---|---|---|
| C16 | Dual `video_id` + `author_id` ablation, WEAK NEGATIVE EVIDENCE | `research/data_profile.md` §12 — paired deltas, seed counts, and the reviewer's classification, unchanged | It was the package's only positive-direction result; a positive measurement sitting among established constraints risks reading as a direction to pursue |
| C17 | Video-statistic provenance / causal safety, INCONCLUSIVE | `DATA_GUIDE.md` §6 (the semantic caveat, where an implementer meets it) and `research/data_profile.md` §9 (the measurements, now including the 0.302 correlation, 0.105/0.505 quintile rates, and the non-monotonic like-ratio trend) | It was the one INCONCLUSIVE entry, kept against the general rule; removing it makes that rule absolute — an open question is not an established constraint |

Both IDs remain in `constraints.md` as retirement markers stating where the evidence
moved, so nothing silently re-points. The structural author/video redundancy that C16
referenced is unaffected and remains in C12.

### Updated (6)

- **C1** — expanded from a four-line summary to the exact `evaluate.py` semantics
  (contributor rule, weighting, gain function, 0.5 fallback) plus official split dates.
- **C2** — unchanged in substance; added the organizer's own confirming measurement
  (`item_pop × user bias` scores identically to plain `item_pop`).
- **C3** — evidence added (the same-row correlations that make the leak consequential);
  the permitted uses are unchanged.
- **C4** — added the auxiliary-density spread (~46% click down to ~0.1% follow) under
  "does not establish", so the permission is not read as an endorsement.
- **C5** — was organizer-README-only with no numbers. Now carries the reviewed local
  values (13-field 0.599930, paired delta −0.001510 ± 0.000792, 3 seeds) alongside the
  organizer's published test figures, and the corrected 8/13 field counts.
- **C6** — was "around k = 8/16/32" from the README. Now carries reviewed local values
  for k = 8/16/32/**64** with population stds, plus the organizer's published figures.

### Removed

- No constraint was deleted. The old section **"3. Findings to Add Later"** — a
  placeholder list of categories awaiting evidence — was removed as superseded: those
  categories are now populated as C7–C22.
- The old "What Must NOT Appear Here" section was **kept and extended** (it now also
  bans generalising a negative past its tested formulation, rewriting an INCONCLUSIVE as
  a prohibition or endorsement, any evaluation-derived number, and any GAUC share
  computed over all positive rows).

### Downgraded / reclassified

| Entry | Before | After | Reason |
|---|---|---|---|
| C3 | "HARD FACT" (leakage discipline) | **INVALID / FORBIDDEN** | Matches REVIEW_REPORT §7; a prohibition is not a measurement |
| C5 | "STRONG NEGATIVE against the tested formulation", organizer-sourced | **STRONG NEGATIVE EVIDENCE**, scoped to the exact **13-field** formulation, with the **8-field** variant explicitly INCONCLUSIVE | REVIEW_REPORT §7 separates the two configurations |
| C6 | "STRONG NEGATIVE against simple FM capacity scaling" | **STRONG NEGATIVE EVIDENCE** for simple width scaling, now covering k=64 and locally reproduced | REVIEW_REPORT §7, scoped to width alone |
| C16 (new) | Audit 1 called dual-ID redundancy STRONG NEGATIVE | **WEAK NEGATIVE EVIDENCE**, exact FM only | REVIEW_REPORT §7: five consistent but sub-epsilon deltas |

---

## 6. references.md

**Changes made.**

- **§0 added — "Task Shape".** States that the task is within-user re-ranking of a fixed
  logged list, not full-catalogue retrieval; that there is no candidate-generation stage
  or conventional negative-sampling problem; that retrieval architectures (two-tower +
  ANN, full-catalogue softmax) do not map onto it without repurposing; and that short
  lists (median 4) bound what list-level methods have to work with. This was the main
  terminology gap — the old file described methods without stating the task shape that
  filters them.
- **Every entry restructured** to the four-part form: problem addressed / mechanism /
  assumptions / implementation and runtime considerations.
- **Nothing deleted.** BPR, listwise, LambdaRank, DeepFM/DCN/xDeepFM, DIN, sequential
  models, multi-task, ESMM, CWM, recency weighting, historical aggregates, off-policy
  evaluation, and ensembling all remain. Per the brief, the weak-negative FM results did
  not justify removing any method family.
- **New entries added** for method families the evidence base makes relevant to describe:
  §14 cold-start / unseen-identifier handling (UNK, hashing, coarse backoff — with the
  warning that backoff must never be a lookup keyed on split membership); §16
  experimental methodology under seed noise (matched seeds, paired deltas, the 0.0008 /
  0.002 scales). §12 was extended to cover target encoding and out-of-fold construction;
  §1 now names FFM and higher-order FM as variants.
- **Feasibility notes replaced with measured coverage** where the audit supplies it —
  e.g. DIN's entry now carries 1.624% / 3.381% / 78.413% coverage by granularity and the
  official "incomplete sequential logs" caveat, instead of "must be measured first".
- **Official ranking quoted and labelled.** §17 records the organizer's own "already
  measured, no gain" and "where the headroom probably is" lists, including their judged
  order (1 loss/objective … 7 unbiased validation), explicitly attributed as *the
  organizer's* judgement and accompanied by the statement that it is not a plan handed to
  the agent. This is the one permitted ranking — it describes an official source.
- **Sources expanded** with `evaluate.py`, `baseline_scores.json`, `ablation_features.py`
  (with the 8/13 correction pointer), and the official KuaiRand repository documentation.
- **§18** restates the no-ranking / no-deletion rules for future additions.

**Why.** The revised audit identified a terminology gap (task shape), several
"feasibility must be measured" notes that are now measured, and method families worth
describing given what the data turned out to look like. No method is called
recommended, best, or most promising in this project's voice.

---

## 7. data_profile.md

The consolidated profile was already reviewer-approved and internally consistent; it was
kept as the base and corrected in five places rather than rewritten.

| Change | Detail |
|---|---|
| Integrity scope header | States explicitly that no value derives from evaluation labels or evaluation-period outcomes, that evaluation-window figures are date-only counts, and that every GAUC share uses the mixed-label-user denominator |
| Evaluation row-count provenance | "official row count" → **"date-only row count"**. 170,588 does not appear in any official file; it is reproduced by counting the `date` column, and is now labelled as such |
| File/split boundary | Added: `log_standard_4_22_to_5_08_pure.csv` holds 295,497 rows spanning both windows — split by `date` |
| Official validation ladder | Added: random 0.4834 / item popularity 0.5807 / FM 0.6016 / oracle 0.8484, plus epsilon 0.002, N 3, published std 0.0008. Test-split reference numbers deliberately left in `PROBLEM.md` rather than duplicated into the profile the proposer consumes |
| Stay-time zero rates scoped | Was an unscoped "99.99% / 95.54%", which read as conflicting with the pre-audit's "over 99.98% / about 95%". Both are correct at different scopes; now stated as `profile_stay_time` 99.994% (validation) / 99.989% (train) and `comment_stay_time` 95.542% / 94.564%, measured directly |
| Learning-rate evidence | Added to §12: lr = 0.0003 / 0.0005 / 0.001 / 0.002 / 0.003 / 0.01 with means and stds, classified WEAK NEGATIVE for the tested high rates (principally 0.01) and INCONCLUSIVE among nearby rates. Width table given explicit per-setting stds |

All other values — cardinalities, overlap, activity tiers, corrected GAUC shares,
history coverage, feedback prevalence, video statistics, temporal profile, random-log
scope, engineering timings — were carried through unchanged after spot-verification
against the raw files (see §1).

`research/data_profile.md` now differs from `research/consolidated/data_profile.md` by
these additions only. The consolidated audit artifacts were left untouched, as they are
source material for this update rather than targets of it.

---

## 8. Stale / Invalid Information Removed

**Test-derived values.** None were present in the six target files. The contamination
the review found — locally scored standard-test labels, and summarised evaluation-period
random-log outcomes/features — lived in the source audits and was already excluded during
consolidation. This update's job was to keep it out, and the package now carries three
active defences against reintroduction: `RULES.md` §1 and §1a, `constraints.md` C7, and
the `data_profile.md` integrity header.

**Incorrect GAUC-weight calculations.** No GAUC weight share existed in the previous
context files, so nothing had to be deleted. Every share introduced in this update
(C10 and `data_profile.md` §6) uses the corrected denominator of **34,592 positives from
mixed-label users only**, and the length-1 bucket correctly carries **0.00%**. The
denominator rule is now stated as a standing rule in `RULES.md` §2a, not only as a
number in a table.

**Other superseded values corrected in this update:**

| Was | Now | Where |
|---|---|---|
| Oracle primary "near 0.8645", split unlabelled | 0.8484 validation / 0.8645 test, split-labelled, validation named as the development denominator | PROBLEM.md §6 |
| Seed std 0.0008 unscoped | Organizer's test-split std over 5 seeds; named as the noise reference for epsilon | PROBLEM.md §5, §7 |
| 170,588 implied as an official published count | Reproduced date-only row count | data_profile.md §1 |
| Unscoped stay-time zero rates (read as conflicting) | Scoped to validation and train separately | data_profile.md §7 |
| Static ablation configurations described only as "CWM static fields" | 8 and 13 fields, with the "9/14" labels flagged as wrong | DATA_GUIDE.md §8, constraints.md C5 |
| "The pre-audit should measure…" placeholders | Replaced by verified structure, or by an explicit pointer to the open question | DATA_GUIDE.md throughout |

**Values deliberately not carried over:** the stale
`research/experiment_results/video_feature_inventory.csv` figures (7,545 video rows,
6,487 authors, `like_cnt` mean 231.84). Direct measurement confirms the full-file values
of 7,583 / 6,510 / 230.75 used in the package.

---

## 9. Findings NOT Added to constraints.md

Deliberately excluded, with the reason for each.

| Finding | Class | Why excluded |
|---|---|---|
| Learning-rate sweep (0.0003–0.01) | WEAK NEGATIVE | Not on the reviewer's candidate list. Recorded in `data_profile.md` §12 as a measurement so iterations are not re-spent on it, but it is not a constraint |
| Fixed video-stat ratios as standalone scorers (0.580378 long-play/show, etc.) | WEAK NEGATIVE | Not a reviewer candidate; a standalone-scorer result says little about the same fields inside a model. Retained in `data_profile.md` §9 |
| Feedback density and same-row association | HARD FACT | REVIEW_REPORT §9: **DO NOT PROMOTE** — largely overlaps C3. The numbers stay in `data_profile.md` §7 and appear in C3 only as the reason the leak matters |
| "86.96% of authors have exactly one video" as its own constraint | HARD FACT | REVIEW_REPORT §9: **DO NOT PROMOTE SEPARATELY**. Folded into C12 with both scopes rather than given its own ID |
| `dur_bucket` removal (−0.000591 ± 0.000156) | INCONCLUSIVE | Reviewer reclassified from WEAK NEGATIVE. Left open for the agent |
| 8-field item-only static expansion | INCONCLUSIVE | Appears only inside C5's "does not establish", so C5 cannot be misread as closing that configuration |
| "Validation resembles late train" | INCONCLUSIVE | Component measurements are in C14; the combined verdict is explicitly refused there |
| Validation random log as a model-selection diagnostic | INCONCLUSIVE | C18 carries the structure; usefulness is named as untested |
| Deployable within-validation online history | INCONCLUSIVE | C13 carries the 81.57% availability count and explicitly denies it is a protocol |
| All untested method families — historical aggregates, sequence models, multi-task, pairwise/listwise, alternative architectures, ratio features in a combined model | INCONCLUSIVE | Never measured. These are the agent's research questions; putting them in `constraints.md` would answer them by implication |
| Dual-ID ablation (former C16) | WEAK NEGATIVE | Retired on instruction. The only positive-direction result in the package; kept as a measurement in `data_profile.md` §12, not as a constraint |
| Video-statistic provenance (former C17) | INCONCLUSIVE | Retired on instruction. Preserved as a semantic caveat in `DATA_GUIDE.md` §6 and as measurements in `data_profile.md` §9 |

### Resolved — the one positive-direction result

**Former C16** was the only entry in the package containing a positive measured delta:
removing either identity field from the official FM improved validation primary by
+0.001316 ± 0.000426 (`author_id`, 5 seeds) and +0.001082 ± 0.000585 (`video_id`,
5 seeds).

It was flagged for your decision and you chose to demote it. It is no longer a
constraint. The measurement survives in `research/data_profile.md` §12 with its paired
deltas, seed counts, and the reviewer's WEAK NEGATIVE EVIDENCE classification for the
exact FM formulation — so the agent can still find it, but it no longer sits among
established constraints where it could read as a direction.

---

## 10. Consistency Check

Performed across all six files by scanning every major number.

**Verified consistent:**

- Author/video statistics — 86.96% (full basic file) and 87.051% (train/validation-observed)
  appear only with their scope labels; max 26 and 24 likewise.
- Activity bucket boundaries — the standardised 0 / 1–17 / 18–36 / 37–65 / 66+ tiers
  appear in `constraints.md` C10 and `data_profile.md` §6 and nowhere else. No file
  contains the superseded 1–13/14–31/32–59/60+ or <10/10–49/50–149/150+ boundaries.
- Activity bucket metrics and GAUC weight shares — identical in both files
  (1.69 / 14.67 / 21.35 / 27.50 / 34.79%), denominator 34,592 stated in three places.
- List-length statistics — median 4 / p90 12 / p99 26 / max 74; bucket table identical
  across both files; length-1 GAUC weight is 0.00% in both.
- Uniform-label percentages — 30.321 / 11.901 / 57.778, totalling 42.222% uniform,
  consistent everywhere they appear.
- History coverage — 98.114 / 92.854 / 85.168%; 1.624 / 3.381 / 78.413%; 81.57% with the
  tie caveat attached in every occurrence.
- Feedback prevalence — one source of truth (`data_profile.md` §7); other files refer to
  it approximately ("~46%", "~0.1%") without restating precise values.
- Runtime measurements — 57.5 s / 78.52 s / 0.018 s / 1.384 s / 30.13 s identical in
  `constraints.md` C19–C20 and `data_profile.md` §13, both labelled run-specific.
- Oracle values — 0.8484 (validation, published) and 0.848393 (validation, reproduced)
  are used consistently; **0.8645 appears only in `PROBLEM.md`, explicitly labelled as
  the test split.**
- Constraint cross-references — C2, C3, C12 as cited from `DATA_GUIDE.md`, and
  C3, C4, C6, C8, C9, C12, C13, C14, C18 as cited from `references.md`, all resolve to
  active entries. The former `DATA_GUIDE.md` → C17 pointer was repointed to
  `research/data_profile.md` §9 when C17 was retired; no reference targets a retired ID.
- Evidence vocabulary — the eight classes in `RULES.md` §8 are exactly those used in
  `constraints.md` and `data_profile.md`.

**Intentional scope-labelled differences** (not conflicts; each carries its scope inline):

1. Train activity median **31** (all 26,210 train users) vs **35** (validation users'
   prior train history). Both appear; C13 names the distinction explicitly.
2. Author redundancy **86.96%** (full file) vs **87.051%** (observed videos).
3. User–tag overlap **68.14%** (raw tag string, missing as one category) vs **71.913%**
   (parsed tokens). Flagged in three files as non-interchangeable definitions.
4. Cold-baseline runtime **57.5 s** vs **78.52 s** — different implementations and
   instrumentation, labelled as such in both files.
5. Stay-time zero rates differ between validation and train; both now stated with scope.

**Result: the six files agree.** No stale value survived the scan.

---

## 11. Autonomy Check

**Could the autonomous agent still reasonably choose among multiple research directions?
Yes.**

The package answers *what is true* and leaves *what to do* open. Every direction the
organizer and the review identified remains genuinely undecided by the evidence base:

- **Objective change** (pairwise / listwise) — untested. C8 records that the baseline
  objective is pointwise and the metric rank-based, as a fact about the code, with an
  explicit note that no alternative objective was tested.
- **History and sequence modelling** — C13 gives coverage by granularity spanning 1.624%
  to 78.413% and explicitly denies that any aggregate or sequence model is known to help.
- **Multi-task** — permitted (C4), density measured, transfer unmeasured, negative
  transfer named as a real possibility.
- **Video statistics** — C17 is deliberately unresolved in both directions: neither ban
  nor clearance.
- **Temporal weighting** — C14 refuses the combined early-vs-late verdict and keeps the
  contradictory component measurements.
- **Random-log diagnostics** — C18 gives structure and calls usefulness untested.
- **Model family, tag granularity, cold-start handling, ensembling** — all open.

What is genuinely closed is narrow and well-scoped: `tab` removal, 13-field static
stuffing, simple width scaling, and (weakly, sub-epsilon) the dual-ID formulation — each
bound to the exact official FM, each with an explicit statement of what it does not
generalise to.

Structural safeguards against prescription: every constraint carries a "Does NOT
establish" section; `constraints.md` §9 bans strategy language by example;
`RULES.md` §13 tells the agent that entry order is not a priority ranking and that
INCONCLUSIVE means measure, not avoid; `references.md` describes methods without ranking
them, and attributes the one ranking present to the organizer.

The clearest evidence the package does not encode an answer: the four segments holding
the most GAUC weight, the granularities with the most history coverage, and the
directions the organizer thinks most promising point at **different** things, and the
package does not reconcile them.

---

## 12. Final Status

**READY TO FREEZE CONTEXT**

All six files reflect the final reviewed evidence, both review corrections are enforced
in more than one place, all cross-file numbers agree under labelled scopes, and the
package leaves the research decisions to the agent.

**No items remain open.** All three previously flagged here are resolved:

- **The judging criteria could not be verified against a primary source.** Resolved
  during deployment: `kuairand-starter-kit/context/problem_spec.md` is derived from the
  official Track 2 problem statement (updated 27 Aug 2026) and its judging table —
  35% / 20% / 20% / 15% / 10% with the same descriptions, including the "scored only if
  the submission beats the baseline" condition on Feasibility — matches `PROBLEM.md` §8
  exactly. The same document also corroborates the split row counts, the validation
  oracle 0.8484, the 3.06% duplicate-pair figure, and the convergence rule.

The two items resolved on your instruction:

- The only positive-direction result (former C16) was demoted out of `constraints.md`
  to `research/data_profile.md` §12.
- The only INCONCLUSIVE entry (former C17) was retired from `constraints.md`; its
  caveat lives in `DATA_GUIDE.md` §6 and its measurements in `research/data_profile.md`
  §9. The rule that inconclusive findings stay out of `constraints.md` is now absolute.

Separately, and outside the scope of freezing the context: `C22` records that all 15
files under `harness/`, `pipeline/`, and `agent/` are still comment-only scaffolds. The
context package is ready; the autonomous system is not built. That gap is stated as
evidence, not resolved here.


---

## 13. Deployment to `kuairand-starter-kit`

The reviewed package was **copied** (not moved) into
`Hackathon/kuairand-starter-kit/`. `Research/gpt/` keeps its copy intact, since its
`README.md` and `SETUP_AND_FOLDER_STRUCTURE.md` reference those paths.

Pre-deployment backup of everything altered in the target:
`kuairand-starter-kit/backup/context_before_reviewed_package/` (7 files).

### What was placed

| File | Destination | Collision |
|---|---|---|
| `PROBLEM.md`, `RULES.md`, `DATA_GUIDE.md`, `CONTEXT_UPDATE_REPORT.md` | `context/` | none — new files |
| `constraints.md`, `references.md` | `context/` | **replaced generated files** (see below) |
| `data_profile.md` | `research/` | none — new directory |

### The generation trap, and how it was closed

`context/build_packet.py` held the full text of `constraints.md` and `references.md` as
Python string literals (`CONSTRAINTS`, `REFERENCES`) and **rewrote both files on every
run**, then assembled `packet.md` from them. Copying the reviewed files in without
touching the script would have meant the next `python context/build_packet.py` silently
reverted them.

`build_packet.py` was therefore rewired:

- The two inline literals were deleted and replaced with a `_read()` helper. The script
  now **reads** `constraints.md` and `references.md` from disk and never writes them.
- Its docstring gained a third governing rule — *single source of truth* — recording that
  these files are human-reviewed inputs, not outputs.
- Its console output distinguishes `wrote` from `read … (human-reviewed, not regenerated)`.

### The proposer's blind spot, and the profile wiring

`agent/proposer.py` has no file-read tool: it sees `packet.md` (one cached system block)
and the journal digest, nothing else. Anything absent from the packet is invisible to it.

That made the C16 demotion accidentally destructive here — the dual-ID measurement had
moved to `research/data_profile.md`, which the packet did not include, so demotion became
deletion from the agent's view. `build_packet.py` now also reads
`../research/data_profile.md` into the packet, so the measurement is available to the
agent **as a measurement** rather than as a constraint, which is what the demotion was
for. Verified present in the rebuilt packet.

### Conflict resolution — reviewed values won

| Claim | Starter-kit value (superseded) | Value now in the package |
|---|---|---|
| Drop `author_id` | +0.00157 | +0.001316 ± 0.000426, 5 seeds |
| Drop `video_id` | +0.00136 | +0.001082 ± 0.000585, 5 seeds |
| Noise reference | local 0.00032 / paired 0.0005, "eps ≈ 4 sigma, a null iteration essentially never clears it" | published **0.0008**; eps ≈ 2.5 sigma, the organizers' own derivation |

The dual-ID figures were carried by the old `CONSTRAINTS` literal and disappeared with
it. The noise framing lived in `build_packet.py`'s `build_baseline()` and was rewritten:
`reference_seed_std` is now the published 0.0008, the local 0.00032 is retained but
labelled as the narrower measurement the review ruled over-confident to use as the
reference, and the "4 sigma / essentially never clears it" claim is gone. Verified absent
from the rebuilt packet.

### Content absorbed from the starter-kit package

Its constraints file carried one fact the reviewed package lacked — per-user monotone
transforms and global calibration are no-ops for both metrics. This was added as **C23**
(HARD FACT) to both copies rather than lost.

Its other two entries were **not** lost by omission: the editable-surface and
`harness.adapter` rules (its C10/C11) are stated in `AGENT_RULES.md` §0–§3, which the
packet already includes verbatim, and the non-unique `(user_id, video_id)` fact is C21.

### Verified after rebuild

- `python context/build_packet.py` runs clean and regenerates `packet.md`.
- Superseded +0.00157 / +0.00136: **absent** from the packet (present only in the backup).
- "4 sigma" framing: **absent**.
- Corrected GAUC denominator 34,592: **present**.
- Test-split oracle 0.8645: **absent** from the packet — it exists only in `PROBLEM.md`,
  which is not wired into the packet, correctly labelled as reference material.
- Dual-ID measurement and C23: **present**.

### Two things to know

1. **`packet.md` grew from 26,164 to 84,971 characters** (~6.5k → ~21.2k tokens). It is
   sent as a single cached system block, so the per-iteration cost is a cache read, but
   Feasibility & Practicality is scored partly on tokens. The largest trimmable overlap
   is the generated `data_profile.json` block, whose tier and invariance figures duplicate
   the reviewed profile (they agree — checked cell by cell); its unique content is the
   encoded representation and the adapter-only column list. Say the word and I will trim
   it to just those fields.
2. **`PROBLEM.md`, `RULES.md`, and `DATA_GUIDE.md` are in `context/` but not in the
   packet.** The packet's spec and rules slots are filled by `problem_spec.md` and
   `AGENT_RULES.md`, which are the repo's own equivalents and do not conflict with the
   reviewed files — `problem_spec.md` is in fact derived from the official problem
   statement and is the better spec of the two. The three files sit alongside as human
   reference. `DATA_GUIDE.md` is the one plausible candidate for wiring in, if you want
   the coder to have the verified schemas; that is a token-budget call, not a
   correctness one.

---

## 14. Freeze Record

See `context/FROZEN.md` in both trees for the frozen file list, SHA-256 fingerprints, and
the unfreeze procedure.
