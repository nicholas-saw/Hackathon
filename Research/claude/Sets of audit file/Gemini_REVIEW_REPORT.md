# KuaiRand-Pure Pre-Audit Review Report

> Reviewer pass over the pre-audit AI's work. Scope: `research/PRE_AUDIT.md`, `research/data_profile.md`,
> `research/scripts/*`, and cross-checks against `source/starter-kit/*` and the raw
> `source/KuaiRand-Pure/data/*` files. `research/experiment_results/` and `research/plots/` were both
> empty at review time — no model experiments or plots exist yet to audit beyond the diagnostic FM runs
> embedded in the exploratory scripts.

## 1. Executive Summary

The pre-audit's underlying measurements were, on the whole, competently done and honestly scoped —
every investigation stayed inside train+validation, respected the official date boundaries, and did not
touch `source/` files. Of 6 originally-written investigations (A01, B01, E01, F01, G01, I01), all 6
reproduced correctly at the level of "the reported number matches an independent re-run," and the
oracle-nDCG@5 computation in B01 independently cross-validates against the organizer's own published
`baseline_scores.json` to 4 decimal places (0.6969 recomputed vs 0.6968 published) — strong evidence the
core evaluation logic was used correctly.

That said, review turned up a genuine mix of documentation gaps, a rule-compliance bug, a crash bug, two
factual numeric errors, and several places where interpretation outran the evidence. None of it
constitutes test-label leakage or invalidates the pre-audit's overall direction, but several items
needed correction before this is safe to hand to a human for `constraints.md` promotion decisions.

- Investigations reviewed: 6 original (A01, B01, E01, F01, G01, I01) + 3 added during review (C01, D01,
  H01, whose raw numbers already existed in `data_profile.md` but had no write-up) = 9 total.
- Verified as-is: A01, C01 (new), D01 (new).
- Verified with minor edit / expanded: B01, E01, F01, G01.
- Corrected (factual/methodology error fixed): I01 (rule-compliance bug), plus two data_profile.md
  numeric errors (Section 7 user-activity percentiles, Section 12 `like_cnt` mean) that had no
  corresponding PRE_AUDIT.md investigation to attach to.
- Inconclusive: none of the 9 investigations required an INCONCLUSIVE verdict — all had clean,
  reproducible measurements once corrected.
- Leakage found: **no test-label leakage**. One rule-compliance issue was found (a diagnostic touched
  test-period *rows*, not labels, when it should not have) — see Issue R-02.
- Readiness: **READY FOR HUMAN CONSTRAINT REVIEW** (see §12 for the caveat this comes with).

## 2. Critical Issues Found

### Issue R-01 — Two investigation-adjacent tables had factual numeric errors

Severity: **MAJOR**

Affected investigation(s): none directly (no PRE_AUDIT.md investigation text asserted these specific
wrong numbers), but they lived in `data_profile.md`, which the autonomous agent is expected to trust.

Problem: `data_profile.md` §7 claimed train-side impressions-per-user median = 35 and p99 = ~250;
direct recomputation over all 26,210 train users gives median = 31, p99 = 207. The claimed "35" turns
out to be Investigation F01's *different* statistic (median prior-train-interactions, computed only
over the 22,377 users who also appear in validation) that was apparently copied into the wrong table.
Separately, §12 claimed `like_cnt` mean = 158; direct recomputation gives 230.75. Two other cells in the
same table were left as literal `mean:?` placeholders.

Why it matters: these are exactly the kind of "compact, agent-readable snapshot" numbers the autonomous
agent is told to trust without re-deriving (`data_profile.md`'s own stated purpose). A wrong median or a
mean off by 46% could mislead feature-scaling or capacity decisions downstream, and — more importantly
for this review's purpose — it shows the two conflated quantities (F01's "median prior interactions for
returning users" vs. "median train activity for all users") are legitimately different populations that
future investigations should not casually interchange.

Correction made: `data_profile.md` §7 and §12 corrected with recomputed values and an explanatory note;
`PRE_AUDIT.md` F01 annotated to explain the population difference so it isn't reintroduced.

Corrected result: train activity median = 31 (was 35), p99 = 207 (was ~250), max = 809 (was ">300");
validation activity median = 4 (was ~5), max = 74 (was ">30"); `like_cnt` mean = 230.75 (was 158);
`long_time_play_cnt` mean = 3,687 and `comment_cnt` mean = 12.93 (both previously unmeasured).

Remaining uncertainty: none — these are direct, deterministic recomputations over the full raw files.

### Issue R-02 — Random-log diagnostic touched test-period rows (rule-compliance bug, not label leakage)

Severity: **MAJOR** (process violation) / **MINOR** (practical impact)

Affected investigation(s): I01.

Problem: `research/scripts/GI_analysis.py`'s "Random UV pairs also in standard logs" statistic built
its comparison set from the raw, **unfiltered** `log_standard_4_22_to_5_08_pure.csv`, which spans both
the validation window (04-22..04-28) and the test window (04-29..05-08). The audit's own stated rule
("Use train + validation only") was violated at the level of touching test-period (user_id, video_id)
*identities* — no test `long_view` labels were read or used, and the statistic itself (a structural
ID-overlap count) could not leak label information, but the rule was still broken in letter.

Why it matters: the pre-audit's credibility rests partly on "we never touched test data." Even a
label-free structural touch sets a bad precedent and should be caught and fixed on principle, especially
since a future, less careful copy-paste of this pattern (e.g., extending it to compute a rate or an
association) could accidentally become a real leak.

Correction made: `research/scripts/GI_analysis.py` fixed to filter to validation dates only before
building the comparison set. `PRE_AUDIT.md` I01 updated with a review-correction note. A standalone
reproduction script (`research/review_scripts/I_random_log_review.py`) was added to make the corrected
computation independently auditable.

Corrected result: 702 / 1,186,006 (0.06%) overlap, train+validation only — versus the original 759
(0.06%), of which 58 pairs came only from the test-window contamination. The rounded headline
percentage is unchanged (0.06% either way), so no downstream conclusion was actually affected — this is
a process fix, not a result reversal.

Remaining uncertainty: none.

### Issue R-03 — Investigation write-ups for phases C, D, H were missing despite being marked "completed"

Severity: **MINOR** (documentation completeness, not a correctness error)

Affected investigation(s): C01, D01, H01 (newly added).

Problem: `PRE_AUDIT.md`'s Audit Status line claimed phases A through I were all "completed," but only
A, B, E, F, G, I had a corresponding `## Investigation` write-up with Question/Method/Result/Evidence
classification/Interpretation. The raw numbers for C (missingness) and D (train→validation overlap)
existed in `data_profile.md` (computed by `A_dataset_structure.py`'s A02/A03/A05 blocks) but were never
promoted to a formal, interpreted investigation. H (temporal profile) was worse: the raw daily series
existed in a saved script output, but `data_profile.md`'s own 3-bucket summary of it obscured two
concrete, useful facts (see Issue R-04).

Why it matters: an investigation entry is where evidence classification and "what this does NOT
establish" live. Data sitting in a profile table without that framing is easier to over-trust or
under-use.

Correction made: added `## Investigation C01`, `## Investigation D01`, `## Investigation H01` to
`PRE_AUDIT.md`, each independently re-verified against either the original script output or a fresh
review script.

Remaining uncertainty: none for C01/D01 (pure re-verification of existing numbers). H01's *cause*
(why the early-train spike happened) is explicitly flagged as unknown and unrecoverable from this data.

### Issue R-04 — Temporal profile summary hid a 13.9x volume swing and a zero-row day on the official train start date

Severity: **MAJOR** (a materially useful fact was lost in summarization, not merely undocumented)

Affected investigation(s): H01 (new).

Problem: `data_profile.md` §13 summarized train volume as roughly "~120k/day early, ~30k/day late."
The actual daily series (independently reproduced in
`research/review_artifacts/H_temporal_review_output.txt`) shows a sharp ramp-and-decay: 52,736 (04-09)
→ 227,808 (04-10) → 278,835 peak (04-11) → declining smoothly down to 20,021 by 04-21, continuing into
validation at 17,844/day average. That's a 13.9x peak-to-trough ratio *within the train window alone* —
nothing like a stable "~120k/day." Separately, the raw train log file (`log_standard_4_08_to_4_21_pure.csv`)
has **zero rows dated 2022-04-08**, even though that is the official train start date; the earliest date
actually present is 2022-04-09. Total train row count (1,141,112) is unaffected — this is a coverage
quirk of the source file's date range, not missing/dropped data.

Why it matters: references.md explicitly lists recency weighting and temporal drift as a live research
direction. An agent reasoning about "is there drift" needs the real shape of the series (a strong,
smooth, order-of-magnitude decay continuing into validation), not a two-bucket average that looks like
mild, roughly-stable decline.

Correction made: `PRE_AUDIT.md` Investigation H01 added with the full daily breakdown and
interpretation; `data_profile.md` §13 replaced with the actual per-day table plus a corrected summary.
Promoted to Candidate 04 for `constraints.md` consideration.

Remaining uncertainty: the cause of the spike (logging artifact vs. a real traffic event) is not
determinable from this dataset and is explicitly left open in H01's "What it does NOT establish."

### Issue R-05 — A broken, duplicate implementation of Phase F existed inside BEF_analysis.py

Severity: **MINOR** (code quality / reproducibility, not a result error — the correct standalone script
already existed and is what actually produced the reported numbers)

Problem: `research/scripts/BEF_analysis.py` contained a second, buggy implementation of the historical-
availability analysis that `research/scripts/F_history.py` already implements correctly. Re-running
`BEF_analysis.py` reproduces phases B and E exactly, then crashes with a `ZeroDivisionError` partway
through its own Phase F block; before crashing, its "Cold" bucket printed a nonsensical non-zero repeat-
video rate for users defined to have zero prior train interactions (a data-type/mapping bug), which is
inconsistent with `F_history.py`'s correct output (0.00% for Cold, as it must be by definition).

Why it matters: this is exactly the "duplicated logic that could drift" and "results that cannot be
reproduced from saved scripts" failure modes the review is supposed to catch. Anyone re-running
`BEF_analysis.py` end-to-end today would hit a crash and could wrongly conclude the whole pre-audit
doesn't reproduce.

Correction made: removed the broken duplicate block from `BEF_analysis.py`, replaced with a pointer to
`F_history.py` (the correct, working, standalone implementation, confirmed to exactly reproduce F01's
published numbers). `F_history.py` itself was not modified — it already worked.

Remaining uncertainty: none. F01's actual numbers were never derived from the broken code path.

### Issue R-06 — `generate_pre_audit.py` / `generate_data_profile.py` are not analysis scripts

Severity: **MINOR**

Problem: these two scripts do not compute anything — each one's entire body is a hardcoded Python
string containing a full copy of (the pre-review version of) `PRE_AUDIT.md` / `data_profile.md`, which
it writes to disk verbatim. They are not part of the reproducible measurement chain (that's
`A_dataset_structure.py`, `BEF_analysis.py`, `F_history.py`, `GI_analysis.py`); running them would
silently overwrite this review's corrections with the stale pre-review text.

Why it matters: this is a "duplicated content that could drift" risk — now realized, since this review
edited `PRE_AUDIT.md` and `data_profile.md` directly and did not update these two scripts to match.
Anyone who runs them will undo this review's corrections.

Correction made: none applied (left as-is per "don't rewrite scripts purely for style"; the fix — either
deleting them or resyncing their hardcoded strings — is a judgment call for a human, not a correctness
issue this review should resolve unilaterally). Flagged here so it isn't run by accident.

Separately, `research/scripts/A_dataset_structure.txt` is UTF-16-encoded (an artifact of `>` redirection
under Windows PowerShell), which made it briefly harder to read but not incorrect; the two review-added
output files in `research/review_artifacts/` are UTF-8. No action needed beyond noting it.

## 3. Verified Findings

| Investigation | Result | Review status | Confidence |
|---|---|---|---|
| A01 | Train: 26,210 users / 7,538 videos / 6,482 authors / 110 tags. Valid: 22,377 / 5,951 / 5,315 / 104. 86.96% of authors have exactly 1 video. | VERIFIED | High — exact match on independent rerun |
| C01 (new) | Missingness ≤3.2% across a handful of optional fields; 0% missing in interaction logs and video statistics. | VERIFIED | High |
| D01 (new) | 98-100% entity-ID overlap train→valid; 1.63% exact user-video pair repeat; 68.14% user-tag pair repeat. | VERIFIED | High |
| B01 | 30.3% all-negative / 11.9% all-positive / 57.8% mixed validation users. Oracle nDCG@5 by list length reproduces organizer's published valid oracle (0.6969 recomputed vs 0.6968 published). | VERIFIED, expanded with activity-tier confound note | High for the oracle/invariant-user math (model-independent); Medium for the single-seed FM bucket breakdown |
| E01 | `is_click`/`play_time_ms` dense and correlated with `long_view` (0.75 / 0.63, validation). | VERIFIED WITH MINOR EDIT (wording) | High for the numbers; the "will help as auxiliary tasks" implication was removed |
| F01 | 85% of validation users have ≥10 prior train interactions; item/author repeat rate <2%; is a genuinely different population from A01's plain activity stats. | VERIFIED WITH MINOR EDIT (clarified population, softened one sentence) | High |
| G01 | Video statistics fully populated; wide dynamic range. | VERIFIED WITH MINOR EDIT (two mean values corrected) | High |
| H01 (new) | 13.9x peak/trough daily volume swing within train; zero rows on 2022-04-08. | VERIFIED | High |
| I01 | Random log entirely postdates train (0 rows before 04-22); overlap with standard logs recomputed correctly. | REQUIRES CORRECTION → corrected | High |

## 4. Corrected Findings

**I01 — Random exposure log overlap statistic**
Original claim: "Random UV pairs also in standard logs: 759 (0.06%)" (computed against
train+validation+test).
Corrected claim: 702 / 1,186,006 (0.06%), train+validation only.
Reason: rule-compliance bug — test-period rows were included in the comparison set.
New evidence classification: unchanged (HARD FACT); the underlying STRONG NEGATIVE conclusion about not
training on the random log is strengthened, not weakened, by the correction (see R-02 and I01's
expanded interpretation on why the whole file, not just its test-dated rows, is unsafe for training).

**data_profile.md §7 — User activity percentiles**
Original claim: train median 35, p99 ~250, max >300 (and validation median ~5, max >30).
Corrected claim: train median 31, p99 207, max 809 (validation median 4, max 74).
Reason: numeric error / population conflation with F01's different statistic.
New evidence classification: HARD FACT (unchanged), now reproducible.

**data_profile.md §12 — Video statistic means**
Original claim: `like_cnt` mean 158; `long_time_play_cnt` and `comment_cnt` unmeasured.
Corrected claim: `like_cnt` mean 230.75; `long_time_play_cnt` mean 3,687; `comment_cnt` mean 12.93.
Reason: numeric error / incomplete measurement.
New evidence classification: HARD FACT (unchanged), now reproducible.

## 5. Inconclusive Findings

None. Every investigation in scope had a clean, reproducible measurement once the corrections above
were applied. This pre-audit is exploratory/diagnostic only — no model experiments (multi-task,
historical-feature, or loss-function variants) were run locally, so there is nothing yet to mark
INCONCLUSIVE on statistical grounds; that category will become relevant once the autonomous agent starts
running controlled experiments.

## 6. Leakage / Integrity Audit

- **Test label access:** none found. Every script that constructs a "valid" DataFrame from
  `log_standard_4_22_to_5_08_pure.csv` explicitly filters to `20220422 <= date <= 20220428` before
  computing anything from `long_view` or other labels (`A_dataset_structure.py`, `BEF_analysis.py`,
  `F_history.py` all do this correctly).
- **Same-row outcome leakage:** none. `is_click`/`play_time_ms`/etc. correlations with `long_view` in
  E01 are diagnostic-only (never fed into a trained model as a feature); this matches the explicitly
  allowed "diagnostic variable" use in RULES.md §4.
- **Future-history leakage:** none found. Historical-availability checks (F01, F_history.py) compare
  each validation row's user against *all* of that user's train-window rows, which are all strictly
  earlier than any validation date — no future-relative-to-the-row information is used. No feature
  pipeline that would need the "build-then-update" ordering rule was actually built yet (no historical
  aggregate feature has been implemented), so that rule has not yet been tested against real code.
- **Validation-label leakage into features:** not applicable yet — no features have been constructed
  from validation labels.
- **Random-log temporal leakage:** found and fixed — see Issue R-02. The underlying I01 conclusion
  (don't train on the random log unfiltered) was correct and is now stated more precisely (the *entire*
  file postdates train, not just its test-dated rows).
- **Source-file integrity:** confirmed unmodified. All three `source/` reads were read-only `pd.read_csv`
  calls; no script writes into `source/`. `evaluate.py` and `data.py` were read but not imported into
  any pattern that monkeypatches or reimplements their logic — `BEF_analysis.py` imports and calls the
  real `data.load`/`data.encode`/`baseline.run_fm`/`baseline.evaluate` directly.

**Explicit statement: no test-label leakage was found anywhere in this pre-audit.** The one integrity
issue found (R-02) was a structural/ID-level touch of test-period rows in a diagnostic count, not a
label read, and has been fixed.

## 7. Statistical Reliability Review

- No model comparison experiments (e.g., "feature X helps/hurts") were run locally in this pre-audit —
  all FM training here was for descriptive bucket breakdowns, not KEEP/REVERT decisions, so RULES.md
  §7's "don't declare a method dead from one noisy run" does not yet apply to anything in this file.
- The one place a trained model's numbers feed into the report (B01's activity-tier and list-length
  GAUC/nDCG@5 breakdowns) uses a single seed and a 12-epoch cap rather than the official 40-epoch/
  patience-4 config. The aggregate reproduces the official baseline only approximately (bucket-weighted
  nDCG@5 ≈ 0.533 vs. official 0.5357) — close enough for the *qualitative* headroom-by-bucket story this
  investigation is making, but a caveat was added so this isn't later mistaken for baseline-quality
  numbers. If a future agent wants to cite exact per-tier metrics, it should rerun with the full
  official config and multiple seeds.
- The T4-activity-tier nDCG@5 drop was flagged as confounded with list length (see B01 review addition)
  rather than accepted as an independent "high-activity users are harder" finding — this is exactly the
  kind of overgeneralization RULES.md §8 warns against, caught before promotion to a candidate.
- Oracle-derived numbers (invariant-user %, oracle nDCG@5 by length) are model-independent and exact —
  no seed variance applies to them at all, and they were the ones cross-validated against the organizer's
  published baseline_scores.json.

## 8. Data Profile Consistency

Two `data_profile.md` cells did not match independent recomputation (§7 activity percentiles, §12
`like_cnt` mean) — both corrected, see §4 above. Everything else in `data_profile.md` that could be
cross-checked against either a saved script output or a fresh rerun matched exactly: entity
cardinalities, overlap percentages, author/video redundancy, uniform-label percentages, list-length
buckets (including the oracle column, cross-validated against the organizer's published number),
feedback-signal means/correlations, and historical-availability-by-tier. `data_profile.md` §13
(temporal profile) was expanded rather than merely corrected, since the original summary, while not
technically false, hid the most useful part of the underlying data (see R-04). §15's evidence-link table
was filled in (it previously contained unresolved `#...` anchors).

## 9. Candidate constraints.md Review

| Candidate | Decision | Reason |
|---|---|---|
| 01 — 42% of validation users have uniform labels | APPROVE WITH REWORDING | Math is correct and verified. Original wording embedded a strategy directive ("Focus optimization on..."); constraints.md's own stated policy is to record facts, not strategies. Reworded version in PRE_AUDIT.md §5 candidate 01 removes that clause. |
| 02 — `is_click`/`play_time_ms` dense and correlated with `long_view` | APPROVE WITH REWORDING | Density/correlation numbers are solid. Original wording implied auxiliary-task efficacy, which was not tested. Reworded to state availability only and explicitly flag efficacy as untested (RULES.md §5 requires empirical negative-transfer testing before any efficacy claim). |
| 03 — Rich history volume, low exact-item repeat | APPROVE | Numbers verified exactly; wording is already appropriately conservative (states availability of history, not a modeling recommendation). |
| 04 (new) — Temporal volume non-stationarity + 04-08 zero-row quirk | APPROVE | Independently reproduced HARD FACT with a clear, non-obvious implication for any future recency-weighting hypothesis. Wording in PRE_AUDIT.md is fact-only, no strategy directive. |
| 05 (new) — Entity IDs generalize, exact pairs don't, tags do | APPROVE | Independently reproduced HARD FACT. Directly useful context for feature-granularity decisions without prescribing which granularity to use. |

No candidate warranted DO NOT PROMOTE or REQUIRES MORE EVIDENCE — all five rest on deterministic,
independently-reproduced statistics rather than model-experiment noise.

## 10. Recommended Wording for Approved Constraints

(For a human to copy into `constraints.md` if they concur — not inserted there by this review.)

- **C-candidate:** "~42% of validation users (30.3% all-negative, 11.9% all-positive) have uniform
  `long_view` labels within their impression list; their nDCG@5 is mathematically fixed regardless of
  ranking, and they do not contribute to GAUC. (Evidence: B01, HARD FACT.)"
- **C-candidate:** "`is_click` and `play_time_ms` are dense signals (present on effectively all rows)
  with strong same-row statistical association with `long_view` (validation correlation 0.75 and 0.63
  respectively). This is a density/availability fact only — it does not establish that either is an
  effective multi-task auxiliary target; that requires empirical testing. (Evidence: E01, HARD FACT.)"
- **C-candidate:** "85% of validation users have ≥10 prior train interactions (median 35 among users
  with any train history), but exact (user, video) repeat exposure is <2% and (user, author) repeat is
  <4%. (Evidence: F01, HARD FACT.)"
- **C-candidate:** "Interaction volume decays by roughly an order of magnitude (13.9x peak/trough) within
  the official train window alone, and continues declining smoothly into validation; the raw train log
  file has zero rows on 2022-04-08 despite that being the nominal train start date (earliest actual date:
  2022-04-09). (Evidence: H01, HARD FACT.)"
- **C-candidate:** "User, video, and author IDs generalize from train to validation almost completely
  (>98%), but exact (user, video) pair repetition is rare (1.63%) while (user, tag) pair repetition is
  common (68.14%). (Evidence: D01, HARD FACT.)"

## 11. Remaining Questions for the Autonomous Agent

These are deliberately left unresolved — the review did not attempt to answer them, and doing so
manually would undercut the autonomy the competition is measuring:

- Which multi-task objective/architecture (if any) actually helps `long_view`, given `is_click` and
  `play_time_ms` are available but untested as auxiliary heads.
- Whether historical aggregates (user click rate, user mean play time) built from the rich-but-non-
  repeating history in F01/D01 actually improve ranking, and at what level of granularity
  (video/author/tag).
- Whether a pairwise or listwise loss outperforms pointwise logloss on this metric.
- Whether the strong volume decay found in H01 corresponds to any drift in `long_view` *rate* (as
  opposed to raw traffic) large enough to justify recency weighting — this review only established the
  volume-side fact, not a rate-drift fact.
- Whether coarser (tag/author) generalization keys meaningfully help given D01's overlap numbers, and
  how to combine them with `video_id`/`user_id` without simply reintroducing the redundancy A01 already
  flagged between `author_id` and `video_id`.
- What causes the 2022-04-09..04-11 traffic spike (left explicitly unexplained in H01).

## 12. Readiness Decision

**READY FOR HUMAN CONSTRAINT REVIEW**

The pre-audit's core measurements are sound, reproducible, and free of test-label leakage. All
identified issues (two numeric errors, one rule-compliance bug touching test-period row identities but
no labels, one crash bug in a duplicate code path, and several places where wording outran evidence)
have been corrected in place, with every correction independently re-derived from the raw source files
and left as an auditable script + saved output under `research/review_scripts/` and
`research/review_artifacts/`. The five candidate findings in `PRE_AUDIT.md` §5 are all backed by
deterministic, reproduced statistics and are ready for a human to decide on promotion into
`constraints.md` using the reworded text in §10 above as a starting point. Nothing found here requires
a broader reset of the pre-audit.
