# PRE-AUDIT — KuaiRand-Pure

> Completed 2026-08-30. This notebook gives the later autonomous agent evidence, not a prescribed model.
>
> **Reviewed 2026-08-30** — see `research/REVIEW_REPORT.md`. All numeric claims that could feasibly be independently recomputed were verified against raw source files or by rerunning scripts; two minor corrections were applied in place (marked "Review correction:" below) and no leakage was found. Readiness decision: READY FOR HUMAN CONSTRAINT REVIEW.

## 0. Audit Rules and Leakage Record

- All label-based measurements and experiments used only train (`2022-04-08..2022-04-21`) and validation (`2022-04-22..2022-04-28`).
- Research loaders check `date` before accessing `long_view` or any feedback field in the second standard log. Rows after `2022-04-28` are skipped before label access.
- The random log was loaded with only `user_id`, `video_id`, and `date`; no random-log feedback or labels were loaded.
- The official `source/starter-kit/evaluate.py`, all other files under `source/`, and `context/constraints.md` were not modified.
- Current-row feedback was used only for diagnostics. It was never an input to a current-row `long_view` prediction.
- The organizer's `ablation_features.py` was read but not executed because it selects checkpoints on validation and then evaluates test labels.

Evidence classes used below: `HARD FACT`, `STRONG NEGATIVE`, `WEAK NEGATIVE`, and `INCONCLUSIVE`.

# 1. Audit Status

| ID | Investigation | Status | Evidence class | Short result |
|---|---|---|---|---|
| R00 | Official baseline reproduction | COMPLETE | HARD FACT | Validation primary 0.601470; official neighborhood reproduced |
| A01 | Cardinalities, coverage, missingness | COMPLETE | HARD FACT | Full train/validation counts and feature inventories measured |
| A02 | Train→validation overlap | COMPLETE | HARD FACT | Entity overlap high; exact pair overlap low |
| A03 | Repeat-pair and tag structure | COMPLETE | HARD FACT | Tag histories repeat much more than item/author histories |
| A04 | Video↔author redundancy | COMPLETE | HARD FACT | Functional mapping; 87.05% of observed authors have one video |
| B01 | User/list/label composition | COMPLETE | HARD FACT | 42.22% validation users are ranking-invariant |
| B02 | Activity-bucket metric structure | COMPLETE | HARD FACT | T4 carries 40.01% GAUC weight and largest nDCG gap contribution |
| B03 | List-length metric structure | COMPLETE | HARD FACT | 6–10 lists carry largest aggregate nDCG gap and GAUC share |
| C01 | Baseline mechanism | COMPLETE | HARD FACT | Five-field pointwise FM with Adam and validation early stopping |
| C02 | Field ablations and seed sensitivity | COMPLETE | STRONG/WEAK NEGATIVE | `tab` is material; dual item/author identity is redundant in this FM |
| D01 | Static-field organizer reproduction | COMPLETE | STRONG NEGATIVE | Full 13-field variant −0.001510 ± 0.000792 paired primary |
| D02 | FM dimension and learning-rate checks | COMPLETE | STRONG NEGATIVE / INCONCLUSIVE | k=8/32 gave no mean gain; tested LR effects stayed within noise |
| E01 | Feedback prevalence/distributions | COMPLETE | HARD FACT | Click/play-time dense; most other actions sparse |
| E02 | Feedback associations/activity | COMPLETE | HARD FACT | Click and play time strongly overlap long-view semantics |
| F01 | Strict train-derived history | COMPLETE | HARD FACT | 85.17% of validation users have ≥10 prior train interactions |
| G01 | Video basic/statistical inventory | COMPLETE | HARD FACT | Complete coverage; substantial redundancy and limited missingness |
| G02 | Fixed ratio diagnostics/window safety | COMPLETE | WEAK NEGATIVE / INCONCLUSIVE | Standalone ratios did not beat item popularity; cutoff is undisclosed |
| H01 | Temporal structure | COMPLETE | INCONCLUSIVE | Validation is not uniformly closer to late training |
| I01 | Random-exposure audit | COMPLETE | HARD FACT | 75.69% of random rows fall in evaluation dates |
| J01 | Runtime and deterministic caching | COMPLETE | HARD FACT | Cold FM 78.52s; verified cache hashes match |
| J02 | Windows execution/recovery | COMPLETE | HARD FACT | Probes pass, but all 15 harness/pipeline/agent files remain comment-only scaffolds (review-corrected scope) |

# 2. Investigations

## Investigation R00 — Official Validation Baseline Reproduction

### Question

Does the local environment reproduce the official FM closely enough to trust deeper work?

### Why this matters

All later diagnostics depend on the official split, row order, encoding, model, and evaluator behaving as expected.

### Data used

Train and validation standard logs, `video_features_basic_pure.csv`, and the unchanged official FM/evaluator. No test label was accessed.

### Method

`baseline_validation.py` duplicates only the official loader's train-derived encoding while guarding the second log by date before label access. It imports the official `FM` and `evaluate` implementations unchanged. Configuration: seed 0, k=16, lr=0.001, batch 8192, maximum 40 epochs, patience 4.

### Result

- Rows: train `1,141,112`; validation `124,909`.
- Best epoch: 7; early stop after epoch 11.
- Validation: GAUC `0.667133`, nDCG@5 `0.535806`, primary `0.601470`.
- Published validation: `0.6674 / 0.5357 / 0.6016`.
- Absolute primary discrepancy: `0.000130`.
- Cold run: `78.52s` (`2.88s` load, `8.47s` encode, `66.60s` training plus epoch evaluations, `0.52s` final evaluation).

### Evidence classification

`HARD FACT`.

### Interpretation

The official validation baseline and split counts reproduce within ordinary seed/rounding variation; deeper analysis is trustworthy.

### What it DOES NOT establish

It does not validate test performance, alternative models, or the causal safety of optional feature sources.

### Potential relevance to later agent

Provides a known-good validation-only reference and realistic cold-run budget.

### Artifacts

- `research/scripts/baseline_validation.py`
- `research/experiment_results/baseline_validation.json`
- `research/experiment_results/baseline_validation_predictions.npz`

## Investigation A01 — Cardinalities, Coverage, and Missingness

### Question

What is the train/validation scale, what feature sources cover it, and where are values missing?

### Why this matters

Cardinality and missingness determine representation size, unknown handling, and whether optional sources are feasible.

### Data used

Train, validation, user features, video basic features, and video statistics.

### Method

Exact row counts, `nunique`, blank/NA checks, and join coverage were computed. Feature inventories are restricted to train/validation-observed entities where appropriate.

### Result

| Measurement | Train | Validation |
|---|---:|---:|
| Rows | 1,141,112 | 124,909 |
| Users | 26,210 | 22,377 |
| Videos | 7,538 | 5,951 |
| Authors | 6,482 | 5,315 |
| Tabs | 15 | 15 |
| Dates with rows | 13 | 7 |

The declared train interval starts April 8, but the raw train file has rows on only April 9–21.

Coverage is `100%` for train and validation users in `user_features_pure.csv`, and `100%` for observed videos in both video feature files. The feature files have 27,285 users and 7,583 videos.

Logs and video statistics have no missing cells. Material feature missingness is limited to user `onehot_feat4` (`3.203%`), user `onehot_feat12..17` (`2.617%` each), video duration (`3.152%`), music type (`2.677%`), and tag (`1.266%`). `visible_status` has cardinality 1. Video basic cardinalities include 6,510 authors, 7,202 music IDs, 14 upload types, 3 video types, 110 tag strings, and 46 parsed tag tokens.

### Evidence classification

`HARD FACT`.

### Interpretation

All major side tables are join-feasible; missing-value handling is localized rather than pervasive. Some fields are constant or nearly identifier-like.

### What it DOES NOT establish

Coverage does not imply usefulness, causal validity, or a recommended encoding.

### Potential relevance to later agent

Defines vocabulary sizes, unknown rates, and fields needing explicit missing handling.

### Artifacts

- `research/experiment_results/data_profile_results.json`
- `research/experiment_results/missingness_inventory.csv`
- `research/experiment_results/video_feature_inventory.csv`

## Investigation A02 — Train→Validation Entity and Pair Overlap

### Question

How much validation data is warm at entity and relationship levels?

### Why this matters

Entity overlap supports learned embeddings; relationship overlap determines the direct coverage of memorized affinities.

### Data used

Train/validation IDs plus video author and parsed tag mappings.

### Method

Exact set intersections were computed for entities and unique pairs. Row-level history coverage was also measured.

### Result

| Validation object | Seen in train |
|---|---:|
| Users | 21,955 / 22,377 (`98.114%`) |
| Videos | 5,944 / 5,951 (`99.882%`) |
| Authors | 5,310 / 5,315 (`99.906%`) |
| Unique user–video pairs | 1,974 (`1.627%`) |
| Unique user–author pairs | 4,081 (`3.376%`) |
| Unique user–tag pairs | 68,316 (`71.913%`) |
| Rows with at least one prior user–tag pair | `78.413%` |

Cold validation entities: 422 users (`1.886%`) and 7 videos (`0.118%`).

### Evidence classification

`HARD FACT`.

### Interpretation

The split is overwhelmingly warm at entity level but mostly novel at exact user–item and user–author relationship level. Tag-level affinity has far broader support.

### What it DOES NOT establish

It does not show that any history model improves the metric or that tags should be selected.

### Potential relevance to later agent

Separates embedding generalization feasibility from exact-pair-history feasibility.

### Artifacts

`research/experiment_results/data_profile_results.json`.

## Investigation A03 — Repeat Frequency

### Question

How often do interaction relationships repeat within training?

### Why this matters

Repeated support constrains the reliability of historical rates and candidate-conditioned histories.

### Data used

Training user–video, user–author, and parsed user–tag interactions.

### Method

Group counts were computed for every unique pair.

### Result

- User–video: 1,092,750 unique pairs; median `1`, mean `1.044`, maximum `22`; `4.130%` repeat, representing `8.194%` of rows.
- User–author: 1,070,326 unique pairs; median `1`, mean `1.066`, maximum `22`; `5.913%` repeat, representing `11.750%` of rows.
- User–tag: 345,211 unique pairs; median `2`, mean `3.674`, p90 `8`, p99 `29`, maximum `167`; `55.250%` repeat, representing `87.819%` of tag interactions.

### Evidence classification

`HARD FACT`.

### Interpretation

Exact item/author histories are sparse; category-level histories have materially deeper repeat support.

### What it DOES NOT establish

It does not establish which aggregation, sequence model, or tag parsing is optimal.

### Potential relevance to later agent

Provides support counts for judging the variance of different historical representations.

### Artifacts

`research/experiment_results/data_profile_results.json`.

## Investigation A04 — Video↔Author Redundancy

### Question

How redundant are `video_id` and `author_id` structurally?

### Why this matters

Highly redundant fields can add parameters and interaction terms without adding much distinct information.

### Data used

Video basic features restricted to train/validation-observed videos.

### Method

The video→author functional dependency and author video counts were measured exactly.

### Result

- Every observed video maps to exactly one author (`100%` functional mapping); no basic-feature video maps to multiple authors.
- 6,487 observed authors; videos/author median `1`, mean `1.163`, p90 `2`, p99 `3`, maximum `24`.
- 5,647 authors (`87.051%`) have exactly one observed video.
- Across the full basic table, `86.959%` of authors have exactly one video.

### Evidence classification

`HARD FACT`.

### Interpretation

Author identity is usually a near-alias for video identity, although the minority of multi-video authors can still carry cross-video information.

### What it DOES NOT establish

Structural correlation alone does not prove either field is useless. Controlled ablations are in C02.

### Potential relevance to later agent

Warns that both IDs may create redundant FM interactions and regularization burden.

### Artifacts

`research/experiment_results/data_profile_results.json`.

## Investigation B01 — Validation List and Label Composition

### Question

Which users are movable under the official metrics?

### Why this matters

GAUC ignores uniform-label users, while nDCG is invariant for both all-negative and all-positive users.

### Data used

Validation labels and official evaluator semantics.

### Method

Per-user list length and positive count were computed. Oracle nDCG used labels as scores only to quantify mathematical headroom.

### Result

- Validation users: `22,377`; rows/user min `1`, median `4`, mean `5.582`, p90 `12`, p99 `26`, max `74`.
- Single-impression users: `3,917` (`17.505%`).
- All-negative: `6,785` users (`30.321%`), 21,807 rows (`17.458%`).
- All-positive: `2,663` users (`11.901%`), 4,540 rows (`3.635%`).
- Mixed/movable: `12,929` users (`57.778%`), 98,562 rows (`78.907%`).
- Total invariant users: `42.222%`.
- Mean user positive rate `0.3483`; median `0.2857`. Positive-rate bucket counts are 6,785 at 0, 4,040 in `(0,.25]`, 6,057 in `(.25,.5]`, 2,425 in `(.5,.75]`, 407 in `(.75,1)`, and 2,663 at 1.

### Evidence classification

`HARD FACT`.

### Interpretation

Only 57.78% of users affect rank-sensitive performance; nevertheless, they contain 78.91% of rows.

### What it DOES NOT establish

It does not prescribe reweighting or removing invariant users from training.

### Potential relevance to later agent

Provides the correct denominator for reasoning about metric headroom.

### Artifacts

- `research/experiment_results/validation_user_metric_profile.csv`
- `research/experiment_results/data_profile_results.json`

## Investigation B02 — Metric Structure by Train-Derived Activity

### Question

Where are GAUC weight and movable nDCG headroom concentrated by prior activity?

### Why this matters

An overall score can hide groups with different support, invariance, and remaining headroom.

### Data used

Train interaction counts, validation labels, and official-FM validation predictions.

### Method

Activity thresholds are train-user count quartiles: T1 `1–13`, T2 `14–31`, T3 `32–59`, T4 `60+`; Cold is zero. Each bucket is evaluated with the unchanged official evaluator. Aggregate nDCG-gap contribution is bucket user share × (oracle nDCG − FM nDCG).

### Result

| Tier | Users | Rows | GAUC | nDCG@5 | Primary | Invariant users | GAUC weight | Overall nDCG-gap contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6091 | 57.11% | 1.69% | 0.0021 |
| T1 | 4,351 | 13,437 | 0.6550 | 0.5374 | 0.5962 | 61.25% | 10.55% | 0.0155 |
| T2 | 5,582 | 23,310 | 0.6686 | 0.5409 | 0.6047 | 48.01% | 19.95% | 0.0298 |
| T3 | 5,791 | 32,052 | 0.6624 | 0.5521 | 0.6073 | 37.70% | 27.80% | 0.0429 |
| T4 | 6,231 | 54,120 | 0.6720 | 0.5154 | 0.5937 | 26.95% | 40.01% | 0.0707 |

T4 has both the largest GAUC weight share and the largest aggregate movable nDCG gap.

### Evidence classification

`HARD FACT`.

### Interpretation

Metric opportunity and evidence volume are concentrated among the most active validation users, while T1 has the highest invariant fraction.

### What it DOES NOT establish

It does not establish that activity weighting or a specialized T4 model helps.

### Potential relevance to later agent

Identifies where metric mass and remaining gap coexist.

### Artifacts

`research/experiment_results/metric_by_activity_bucket.csv`.

## Investigation B03 — Metric Structure by Validation List Length

### Question

Which validation list lengths carry metric weight and movable nDCG gap?

### Why this matters

Long lists have more ranking freedom, but aggregate importance also depends on user count and GAUC positive weight.

### Data used

Validation labels and official-FM predictions.

### Method

Users were bucketed by validation list length and evaluated with the official evaluator.

### Result

| List length | Users | GAUC | nDCG@5 | Oracle nDCG | Gap | GAUC weight | Overall gap contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.0000 | 0.00% | 0.0000 |
| 2–3 | 6,218 | 0.6472 | 0.5413 | 0.6086 | 0.0673 | 10.27% | 0.0187 |
| 4–5 | 4,119 | 0.6645 | 0.6185 | 0.7492 | 0.1307 | 16.36% | 0.0241 |
| 6–10 | 5,225 | 0.6756 | 0.5913 | 0.8536 | 0.2623 | 36.39% | 0.0613 |
| 11–20 | 2,346 | 0.6677 | 0.5037 | 0.9182 | 0.4145 | 27.08% | 0.0435 |
| 21+ | 552 | 0.6596 | 0.3934 | 0.9420 | 0.5486 | 9.90% | 0.0135 |

`*` No single-impression user contributes to GAUC; 0.5 is the evaluator's empty-denominator return.

The 6–10 bucket carries the largest aggregate nDCG gap contribution and GAUC weight. The 21+ bucket has the largest per-user nDCG gap but far fewer users.

### Evidence classification

`HARD FACT`.

### Interpretation

The most movable per-user lists and the most important aggregate bucket are not the same.

### What it DOES NOT establish

It does not prescribe listwise training or bucket-specific optimization.

### Potential relevance to later agent

Prevents optimizing for a visually large per-user gap with little total metric mass.

### Artifacts

- `research/experiment_results/metric_by_list_length.csv`
- `research/plots/validation_list_length_users.png`

## Investigation C01 — Exact Baseline Mechanism

### Question

What does the official FM optimize and how is it trained?

### Why this matters

Later hypotheses need an exact reference mechanism, not merely the label “FM.”

### Data used

Official `data.py`, `baseline.py`, and `evaluate.py`.

### Method

Direct source inspection plus baseline reproduction.

### Result

- Fields: `user_id`, `video_id`, `author_id`, `tab`, and ten-bin train-quantile `dur_bucket`.
- Unknown validation categories map to one reserved slot per field.
- FM: first-order weights plus all pairwise embedding interactions; k=16.
- Objective: pointwise binary cross-entropy/logistic loss for `long_view`.
- Optimizer: hand-written Adam, β1=0.9, β2=0.999, ε=1e−8; lr=0.001; L2=1e−6 on W and V.
- Batch 8,192; shuffled each epoch; maximum 40 epochs.
- Early stop on validation primary after four non-improving epochs; improvement threshold 1e−5.
- Bias is updated by plain gradient descent, while W/V use Adam.
- Seed controls initialization and epoch permutations.

### Evidence classification

`HARD FACT`.

### Interpretation

The training objective is pointwise while both reported metrics are within-user ranking metrics.

### What it DOES NOT establish

It does not establish that a pairwise or listwise objective will improve validation.

### Potential relevance to later agent

Defines the exact bottleneck and fair comparison target for future hypotheses.

### Artifacts

- Official source files (read only)
- `research/experiment_results/baseline_validation.json`

## Investigation C02 — Field Ablations and Seed Sensitivity

### Question

Which baseline fields add distinct value under the exact FM, and how noisy are results?

### Why this matters

Structural redundancy does not prove empirical redundancy; repeated paired validation runs do.

### Data used

Train/validation only, official FM/evaluator, seeds 0–2 for all ablations and seeds 0–4 for identity-field verification.

### Method

One field was removed at a time, holding k/lr/batch/early stopping fixed. Deltas are paired by seed against the five-field baseline.

### Result

Three-seed base: primary `0.601440 ± 0.000275` (population std).

| Variant | Primary mean ± std | Paired delta mean ± std | Classification |
|---|---:|---:|---|
| Remove `tab` | 0.585538 ± 0.000429 | −0.015903 ± 0.000467 | STRONG NEGATIVE against removal |
| Remove `dur_bucket` | 0.600849 ± 0.000225 | −0.000591 ± 0.000156 | WEAK NEGATIVE against removal |
| Remove `author_id` (5 seeds) | 0.602889 ± 0.000451 | +0.001316 ± 0.000426 | STRONG evidence of redundancy in this FM |
| Remove `video_id` (5 seeds) | 0.602654 ± 0.000307 | +0.001082 ± 0.000585 | STRONG evidence of redundancy in this FM |

All five paired identity-field deltas are positive for both removals. The author-removal paired deltas range `+0.000615..+0.001877`; video-removal deltas `+0.000404..+0.002163`.

### Evidence classification

- `STRONG NEGATIVE` against removing `tab` in this FM.
- `WEAK NEGATIVE` against removing `dur_bucket` in this FM.
- `STRONG NEGATIVE` against the exact dual `video_id`+`author_id` formulation relative to either tested four-field alternative.

### Interpretation

`tab` carries large distinct signal. Duration contributes a small, reproducible amount. The near-functional video/author relationship is empirically redundant under this exact FM; retaining both consistently underperformed retaining either one.

### What it DOES NOT establish

It does not prove author or video identity is universally useless, nor does it prescribe a final field set for other architectures.

### Potential relevance to later agent

Provides narrow, repeated field-level prior evidence and an empirical seed scale.

### Artifacts

- `research/scripts/controlled_fm_experiments.py`
- `research/experiment_results/controlled_fm_experiments.json`
- `research/experiment_results/controlled_fm_identity_extra_seeds.json`
- `research/experiment_results/controlled_fm_identity_five_seed_summary.json`

## Investigation D01 — Organizer Static-Feature Reproduction

### Question

Is the organizer-reported static-feature dead end reproducible on validation without test labels?

### Why this matters

It prevents the future agent from spending iterations repeating an exact low-value formulation while preserving broader feature hypotheses.

### Data used

Train/validation logs, user features, video basics, official FM/evaluator, seeds 0–2.

### Method

The organizer code's actual schemas were reproduced: base 5 fields; item-static adds `music_id`, `video_type`, `upload_type` for 8 total; full CWM-style adds five user buckets for 13 total. Validation-selected metrics were reported directly; test code was removed.

### Result

- Base: `0.601440 ± 0.000275`.
- 8-field item-static: `0.601108 ± 0.000461`; paired delta `−0.000332 ± 0.000205`.
- 13-field full static: `0.599930 ± 0.000523`; paired delta `−0.001510 ± 0.000792`; all three seeds lower.
- The Starter Kit prose/code counts are inconsistent: the code adds 3 new item fields and 5 user fields, yielding 5+3+5=13.

### Evidence classification

- Full 13-field formulation: `STRONG NEGATIVE` against meaningful benefit.
- Item-only expansion: `INCONCLUSIVE`; observed delta is small and within the practical seed scale.
- Schema discrepancy: `HARD FACT`.

### Interpretation

The exact full static-feature stuffing formulation is reproducibly low-value in this environment.

### What it DOES NOT establish

It does not invalidate static features in other models, derived feature interactions, missing-aware handling, or a subset chosen for a specific mechanism.

### Potential relevance to later agent

Avoids repeating the organizer's exact formulation while leaving feature research decisions open.

### Artifacts

`research/experiment_results/controlled_fm_experiments.json`.

## Investigation D02 — FM Capacity and Learning-Rate Sensitivity

### Question

Does simple capacity scaling or a nearby learning-rate change materially improve the exact baseline?

### Why this matters

The later agent should know whether basic scalar tuning is a likely source of headroom.

### Data used

Train/validation, seeds 0–2, official FM/evaluator.

### Method

Only k and lr were changed, one at a time.

### Result

| Config | Primary mean ± std | Paired delta mean ± std | Mean train time |
|---|---:|---:|---:|
| k=8 | 0.601110 ± 0.000796 | −0.000330 ± 0.000790 | 39.0s |
| k=16 | 0.601440 ± 0.000275 | reference | 51.3s |
| k=32 | 0.601460 ± 0.000688 | +0.000020 ± 0.000504 | 72.9s |
| lr=0.0005 | 0.601776 ± 0.000280 | +0.000336 ± 0.000353 | 75.6s |
| lr=0.002 | 0.601364 ± 0.000826 | −0.000076 ± 0.000625 | 36.0s |

### Evidence classification

- Simple k=8/16/32 scaling: `STRONG NEGATIVE` against meaningful benefit.
- Learning-rate differences: `INCONCLUSIVE` because deltas are within seed noise/practical epsilon.

### Interpretation

k=32 nearly doubles mean training time versus k=8 without a mean score gain. Tested learning rates trade runtime for no defensible score change.

### What it DOES NOT establish

It does not rule out other regularization, schedules, optimizers, model families, or substantially different configurations.

### Potential relevance to later agent

Provides a runtime/benefit prior for simple FM tuning.

### Artifacts

`research/experiment_results/controlled_fm_experiments.json`.

## Investigation E01 — Post-Impression Feedback Prevalence and Distribution

### Question

Which auxiliary signals have enough support to be plausible targets or histories?

### Why this matters

Very sparse tasks may be hard to learn; dense signals can still be redundant with the main target.

### Data used

Current-row feedback from train and validation for diagnostics only.

### Method

Binary prevalence, continuous zero rate/quantiles, and split drift were measured.

### Result

| Signal | Train mean/prevalence | Validation | Key distribution fact |
|---|---:|---:|---|
| `is_click` | 46.345% | 44.383% | Dense |
| `is_like` | 1.868% | 1.797% | Sparse |
| `is_follow` | 0.101% | 0.130% | Very sparse |
| `is_comment` | 0.257% | 0.233% | Very sparse |
| `is_forward` | 0.100% | 0.078% | Very sparse |
| `is_hate` | 0.042% | 0.062% | Very sparse |
| `is_profile_enter` | 2.539% | 1.945% | Sparse and shifted |
| `play_time_ms` | 23,260.5 | 21,486.8 | train median 4,970; p99 213,231; 13.89% zero |
| `profile_stay_time` | 3.31 | 1.88 | 99.989% / 99.994% zero |
| `comment_stay_time` | 552.9 | 460.3 | 94.56% / 95.54% zero |

### Evidence classification

`HARD FACT`.

**Review correction:** The validation `is_follow` prevalence was transcribed as `0.131%`; the underlying artifact (`feedback_profile.csv`, `data_profile_results.json`) records `0.1304950...%`, which rounds to `0.130%`. Corrected above. No other value in this table was affected; the underlying artifact and all other rows were independently reproduced from raw CSVs and matched exactly.

### Interpretation

Click and play-time targets have broad row support; action targets other than like/profile-entry are extremely sparse, and profile-stay time is almost always zero.

### What it DOES NOT establish

Density alone does not establish auxiliary-task benefit or architecture.

### Potential relevance to later agent

Provides label-support priors and identifies likely imbalance challenges.

### Artifacts

- `research/experiment_results/feedback_profile.csv`
- `research/experiment_results/feedback_by_activity_tier.csv`

## Investigation E02 — Feedback Relationships and Activity Variation

### Question

How do feedback signals relate to `long_view`, one another, and activity?

### Why this matters

Strong overlap can supply shared supervision but can also make an auxiliary target redundant; sparse signals may vary materially by activity.

### Data used

Train and validation current-row feedback for diagnostics.

### Method

Conditional long-view rates and Pearson correlations were computed; continuous signals were also log1p-transformed for the joint correlation matrix.

### Result

Train long-view correlations: click `0.7605`, raw play time `0.6351` (log1p `0.5960`), comment-stay log1p `0.2702`, profile-entry `0.1461`, like `0.0992`, comment `0.0590`, follow `0.0250`, forward `0.0226`, hate `−0.0039`, profile-stay log1p `0.0079`.

Train long-view rates are `0.00263` when click=0 versus `0.72330` when click=1. The official definitions explain part of this: both click/valid-play and long-view are thresholded functions of watch time and duration.

In validation, click prevalence declines with train-derived activity: Cold `48.59%`, T1 `50.83%`, T2 `48.58%`, T3 `46.82%`, T4 `39.38%`; mean play time similarly falls from T1 `29,325ms` to T4 `17,088ms`.

### Evidence classification

`HARD FACT`.

### Interpretation

The dense auxiliary signals contain substantial supervision but are not independent of the target definition. Activity-conditioned distributions differ.

### What it DOES NOT establish

It does not establish positive or negative transfer, a loss weight, or a multi-task architecture.

### Potential relevance to later agent

Supplies target-dependence and cohort-shift evidence for forming a later multi-task hypothesis.

### Artifacts

- `research/experiment_results/feedback_correlation_log_continuous.csv`
- `research/experiment_results/feedback_by_activity_tier.csv`

## Investigation F01 — Strictly Prior Historical Information

### Question

How much behavior exists strictly before each validation interaction?

### Why this matters

History-based methods are infeasible without adequate prior support; exact candidate histories and category histories have different coverage.

### Data used

Only training rows as history and validation IDs as candidates. Every history date precedes validation.

### Method

Per-validation-user training counts and feedback sums were computed. Candidate coverage checks used train user–video, user–author, and user–tag sets.

### Result

- Prior interactions/user: median `35`, mean `47.42`, p90 `103`, p99 `216`, max `809`.
- Users with ≥1/≥5/≥10 prior interactions: `98.114% / 92.854% / 85.168%`.
- Prior clicks ≥1/≥5/≥10: `96.157% / 82.531% / 66.309%`.
- Prior likes: `23.229% / 4.683% / 2.239%`.
- Prior comments: `7.785% / 0.241% / 0.018%`.
- Prior follows: `3.423% / 0.054% / 0.022%`.
- Prior forwards: `3.365% / 0.049% / 0.013%`.
- Prior hates: `1.028% / 0.063% / 0.031%`.
- Users with ≥1/≥5/≥10 positive-play-time rows: `97.640% / 91.053% / 82.111%`.
- Validation row coverage: prior same video `1.624%`, same author `3.381%`, same tag `78.413%`.

| Tier | Median prior rows | ≥1 | ≥5 | ≥10 | Same video rows | Same author rows | Same tag rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 0 | 0% | 0% | 0% | 0% | 0% | 0% |
| T1 | 7 | 100% | 72.95% | 33.42% | 0.68% | 1.48% | 46.97% |
| T2 | 22 | 100% | 100% | 100% | 1.03% | 2.27% | 71.65% |
| T3 | 43 | 100% | 100% | 100% | 1.29% | 2.79% | 81.75% |
| T4 | 89 | 100% | 100% | 100% | 2.37% | 4.80% | 90.04% |

The official KuaiRand documentation cautions that Pure has incomplete sequential logs; 27K/1K are recommended for rigorous sequence research.

### Evidence classification

`HARD FACT`.

### Interpretation

General and click/play-time histories are broadly available, rare actions are not, exact candidate identity repeats are scarce, and tag histories have much stronger coverage. History depth is highly activity-dependent.

### What it DOES NOT establish

It does not establish that historical features, DIN, or any sequential model improves validation, and it does not make Pure a complete behavior history.

### Potential relevance to later agent

Defines which historical signals have enough support for a future hypothesis and where cold handling is required.

### Artifacts

- `research/experiment_results/history_by_activity_tier.csv`
- `research/experiment_results/history_signal_by_activity_tier.csv`
- [Official KuaiRand README](https://github.com/chongminggao/KuaiRand/blob/main/README.md)

## Investigation G01 — Video Feature Inventory and Redundancy

### Question

What distinct basic/statistical information exists, and which fields are redundant?

### Why this matters

Blindly adding correlated counts or duplicate duration fields increases cost without necessarily adding information.

### Data used

Both video feature files and train/validation interaction duration.

### Method

Full field inventory, missingness/cardinality/range summaries, duration joins, and all-pairs Spearman correlations among numeric statistics.

### Result

- Both files have 7,583 rows and `100%` coverage of train/validation videos.
- `duration_ms` exactly equals basic `video_duration` on `100%` of nonmissing joined interaction rows; Spearman `1.0`. Basic video duration is missing on `3.152%` of videos/interactions cover `97.932%` nonmissing.
- `visible_status` is constant.
- Numeric video statistics have no missing values, but 54 field pairs have |Spearman|≥0.95.
- Examples: like count vs like users `0.999865`, follow count vs follow users `0.999754`, long-play count vs long-play users `0.999678`, valid-play count vs valid-play users `0.999499`, and play count vs play users `0.999000`.

### Evidence classification

`HARD FACT`.

### Interpretation

The sources are complete but contain exact and near-exact redundancies. Interaction duration is a fully covered fallback for the missing basic duration on logged rows.

### What it DOES NOT establish

Correlation does not prove a field is useless inside a nonlinear or regularized model.

### Potential relevance to later agent

Provides a defensible basis for avoiding duplicated raw fields and for selecting representative statistics if a later hypothesis uses them.

### Artifacts

- `research/experiment_results/video_feature_inventory.csv`
- `research/experiment_results/video_stat_top_redundancies.csv`

## Investigation G02 — Smoothed Ratio Diagnostics and Aggregation Window

### Question

Do semantically fixed ratios contain distinct standalone ranking information, and is their timing causally defensible?

### Why this matters

Ratios can normalize exposure scale, but undisclosed future aggregation would make them unsafe.

### Data used

Video statistics, train item long-view counts, and validation labels. No weight sweep was run.

### Method

Nine predeclared ratios `(numerator + 20×global_ratio)/(show_cnt+20)` were evaluated as standalone static scores and compared with train-derived smoothed item popularity (`0.580722` primary). Spearman correlation with train item long-view rate was measured.

### Result

- Long-play/show: primary `0.580378`, delta `−0.000344`, Spearman with train item rate `0.7167`.
- Valid-play/show: `0.570874`, delta `−0.009848`, Spearman `0.6543`.
- Complete-play/show: `0.550128`, delta `−0.030594`.
- Play/show: `0.540600`, delta `−0.040122`.
- Like/comment/follow/share ratios: primary `0.483741 / 0.454772 / 0.456476 / 0.448518`.

The official documentation says these are per-day, per-scenario averages “over one month.” It does not disclose the exact month endpoints or a cutoff relative to an April 22–28 impression. Therefore causal validity for this split is not established.

### Evidence classification

- Standalone fixed-ratio scoring: `WEAK NEGATIVE` against these exact formulations.
- Causal safety: `INCONCLUSIVE`.
- One-month averaging semantics: `HARD FACT` from official documentation.

### Interpretation

Long-play/show largely tracks train item popularity as a standalone score; other predefined ratios are weaker. The entire source must remain quarantined from causal training claims until its cutoff is established.

### What it DOES NOT establish

It does not prove the statistics lack incremental value when combined with other features, nor that the ratios are safe or unsafe—only that safety is undocumented.

### Potential relevance to later agent

Provides fixed diagnostics, redundancy evidence, and an explicit leakage question the agent must resolve before use.

### Artifacts

- `research/experiment_results/video_stat_ratio_diagnostics.csv`
- `research/experiment_results/official_source_notes.json`
- [Official field documentation](https://github.com/chongminggao/KuaiRand/blob/main/README.md#4%EF%B8%8F%E2%83%A3-descriptions-of-the-fields-in-video_features_statisticcsv)

## Investigation H01 — Temporal Structure

### Question

Does validation empirically resemble late training more than early training?

### Why this matters

Temporal proximity alone is not evidence for recency weighting; observable distributions must be compared.

### Data used

Train/validation dates, labels, IDs, duration, and tab.

### Method

Daily profiles were computed. Early train is April 8–14 (six dates with rows: April 9–14), late train April 15–21, validation April 22–28. Period means and tab Jensen–Shannon divergence were compared.

### Result

| Period | Rows/day | Long-view rate | Users/day | Videos/day | Mean duration |
|---|---:|---:|---:|---:|---:|
| Early train | 148,570 | 0.33228 | 18,471 | 5,257 | 98,553ms |
| Late train | 35,671 | 0.35211 | 12,424 | 4,207 | 95,477ms |
| Validation | 17,844 | 0.31328 | 9,140 | 3,429 | 102,820ms |

Validation is closer to early train in long-view rate (absolute gap `0.01900` vs late `0.03882`) and mean duration (`4,267ms` vs `7,343ms`), but closer to late train in tab distribution (JS `0.00252` vs early `0.00392`) and volume/user/video counts. Daily volume declines sharply; validation long-view rate falls from `0.3382` on April 23 to `0.2899` on April 28.

### Evidence classification

`INCONCLUSIVE` for the single claim that validation “resembles late train more.” The individual daily/period measurements are `HARD FACT`.

### Interpretation

Temporal drift is real but multidimensional; recency is not a uniformly better distribution match.

### What it DOES NOT establish

It does not establish whether date features, recency weighting, or time-aware validation improve the metric.

### Potential relevance to later agent

Supplies concrete drift dimensions without choosing a temporal strategy.

### Artifacts

- `research/experiment_results/daily_standard_profile.csv`
- `research/experiment_results/temporal_period_summary.csv`
- `research/plots/daily_standard_profile.png`

## Investigation I01 — Random-Exposure Log

### Question

What is the random log's date/entity structure and leakage risk?

### Why this matters

Random exposure may support bias diagnostics, but most rows overlap the forbidden evaluation period.

### Data used

Only random-log `user_id`, `video_id`, and `date`, compared with train/validation standard identifiers.

### Method

Exact date counts and set intersections were computed. No feedback or label column was loaded.

### Result

- Rows `1,186,059`; dates April 22–May 8; users `27,285`; videos `7,583`; unique pairs `1,186,006`.
- Evaluation-period dates April 29–May 8 contain `897,721` rows (`75.689%`).
- Relative to standard train+validation entities: `97.607%` of random users and `99.499%` of random videos overlap.
- Only `702` unique random user–video pairs (`0.0592%`) overlap standard train+validation pairs.
- Official collection mechanism: a normal list item was replaced by a uniformly sampled item from the 7,583-item pool during April 22–May 8.

### Evidence classification

`HARD FACT`.

### Interpretation

The log has broad entity coverage and almost entirely novel pairs. Date filtering is mandatory for any future development use; evaluation-period labels cannot be used for model selection.

### What it DOES NOT establish

It does not establish a valid propensity estimator, whether to train on random rows, or whether random-validation metrics predict standard-validation performance.

### Potential relevance to later agent

Potential safe roles include identifier/date diagnostics and, after explicit design, validation-period-only exposure-bias/OPE diagnostics. Training use remains an open research decision.

### Artifacts

- `research/experiment_results/data_profile_results.json`
- `research/experiment_results/official_source_notes.json`
- [Official KuaiRand paper](https://jiawei-chen.github.io/paper/kuairand.pdf)

## Investigation J01 — Runtime, Memory, and Deterministic Cache

### Question

What iteration cost and cache behavior should the future agent expect?

### Why this matters

The final run has a six-hour budget and must not trade correctness for caching speed.

### Data used

Train/validation loader, official baseline, and research-only cache.

### Method

Cold stages were timed. A pickle cache was keyed by a version plus SHA-256 over only the included header/train/validation lines. Original and cached frames were hashed by columns, dtypes, index, and values. A deliberately changed source fingerprint was tested for rejection.

### Result

- Baseline cold total `78.52s`: load `2.88s`, encode `8.47s`, training plus 11 epoch evaluations `66.60s`, final evaluation `0.52s`.
- Epoch evaluations total `6.53s`; approximate update time `60.08s`.
- Descriptive profile peak process RSS about `1.41GB`; machine physical RAM `16.76GB`, 8 logical CPUs.
- Raw DataFrame load `3.139s`; cache read `0.043s` (`72.8×` raw-read speedup).
- Full safe content fingerprint `1.341s`; fingerprint+read `1.384s` (`2.27×` effective speedup).
- Cache size `81,028,199` bytes.
- Original and cached train/validation hashes are identical; changed fingerprint is rejected.

### Evidence classification

`HARD FACT`.

### Interpretation

The dataset is CPU-manageable, and deterministic caching materially reduces repeated parsing while retaining strong invalidation. Concurrent heavy processes can reduce memory headroom; repeated environment probes observed over 90% system RAM use.

### What it DOES NOT establish

Timings are machine/load specific and do not guarantee future model runtimes. The research cache is not integrated into the final harness.

### Potential relevance to later agent

Supports iteration budgeting and a safe cache design.

### Artifacts

- `research/scripts/cache_validation_probe.py`
- `research/experiment_results/cache_probe.json`
- `research/experiment_results/cache/train_validation_raw.meta.json`

## Investigation J02 — Windows Subprocess and Recovery Environment

### Question

Can the operating environment execute, time out, terminate, and recover from common failures?

### Why this matters

Autonomous research requires bounded subprocesses and explicit failure recovery.

### Data used

Local Windows/Python environment and read-only inspection of current harness/pipeline files.

### Method

Non-destructive subprocess probes covered success, timeout, parent+child process trees, a syntax error followed by a valid run, and NaN/Inf arrays.

### Result

- Windows 11, Python 3.13.7, NumPy 2.3.2, pandas 2.3.2; ARMv8 processor reported through an AMD64-compatible Python environment.
- Normal subprocess succeeded in `0.046s`; a 0.3s timeout fired in `0.313s`.
- Recursive `psutil` termination saw the child PID and left neither parent nor child alive.
- Syntax-error subprocess returned code 1 with captured stderr; the next valid subprocess returned code 0.
- `np.isfinite` detected all three NaN/+Inf/−Inf values; official `submit.py` also explicitly rejects NaN/Inf.
- `harness/executor.py`, `guards.py`, `cache.py`, `diagnostics.py`, and `pipeline/data_adapter.py`, `features.py`, `train.py` contain zero executable non-comment lines.

**Review correction:** The probe only inventoried those 7 files. An independent check of the full repository found that `harness/logger.py`, `harness/score.py`, `harness/submission.py`, and all five `agent/*.py` files (`coder.py`, `controller.py`, `governor.py`, `proposer.py`, `reflector.py`) are also exactly 1 line and comment-only. The complete scaffold-only set is 15 files, not 7: the entire `harness/`, `pipeline/`, and `agent/` layer is unimplemented, not only the two directories originally probed. `reports/`, `submissions/`, `runlogs/`, and `tests/` contain no files at all. This changes the severity, not the direction, of the original finding.

### Evidence classification

`HARD FACT`.

### Interpretation

The OS/runtime can support the required controls, but the project has only scaffold comments for the final harness/pipeline/agent layer; none of the probed behavior is integrated there yet, and the gap is repository-wide rather than limited to harness/pipeline.

### What it DOES NOT establish

It does not implement or validate the final autonomous agent, resume/checkpoint behavior, or six-hour orchestration.

### Potential relevance to later agent

Defines Windows-specific process-tree handling and a major engineering-readiness constraint.

### Artifacts

- `research/scripts/engineering_environment_probe.py`
- `research/experiment_results/engineering_environment_probe.json`

# Evidence Summary

## Hard Facts

- The validation-only official FM reproduces at GAUC `0.667133`, nDCG@5 `0.535806`, primary `0.601470`; row counts match exactly.
- The evaluator weights mixed users' GAUC by positive count and averages nDCG over all users; `42.222%` of validation users are invariant to ranking.
- Entity overlap is high (users `98.114%`, videos `99.882%`, authors `99.906%`), but exact prior pair coverage is low (user–video `1.624%` of validation rows; user–author `3.381%`).
- User–tag history covers `78.413%` of validation rows and has substantially more repeat support than identity pairs.
- T4 carries `40.012%` of GAUC weight and the largest aggregate activity-tier nDCG gap; lists of length 6–10 carry `36.393%` of GAUC weight and the largest list-bucket aggregate gap.
- `tab` has 15 values and carries distinct baseline signal; video↔author is a functional mapping and `87.051%` of observed authors have one video.
- Click/play time are dense; likes are sparse; follow/comment/forward/hate and profile-stay signals are extremely sparse.
- `85.168%` of validation users have at least 10 prior train interactions, but only `2.239%` have at least 10 prior likes and fewer than `0.04%` have 10 prior follow/comment/forward/hate actions.
- Video side tables have complete entity coverage. Interaction and basic video duration match exactly wherever basic duration is present. Video statistics contain many near-duplicate count/user-count pairs.
- The official video statistics are one-month averages, but the exact cutoff is not documented.
- Validation is closer to early training on long-view rate and duration, but closer to late training on tab/volume measures.
- The random log has 1,186,059 rows; `75.689%` fall on evaluation dates; random pair overlap with standard train+validation is only `0.0592%`.
- Cold baseline runtime is `78.52s`; verified cache content matches exactly and effective fingerprint+read time is `1.384s` versus `3.139s` raw load.
- The Windows runtime supports timeout/tree termination and error recovery, but the final harness/pipeline files remain comment-only scaffolds.

## Strong Negative Evidence

- Removing `tab` from the exact FM: paired primary delta `−0.015903 ± 0.000467` across three seeds.
- Retaining both near-redundant `video_id` and `author_id` in the exact five-field FM was lower than removing either field across all five paired seeds: remove-author `+0.001316 ± 0.000426`; remove-video `+0.001082 ± 0.000585`.
- Exact 13-field static CWM-style FM expansion: paired delta `−0.001510 ± 0.000792` across three seeds, with every seed lower.
- Simple FM capacity scaling gave no meaningful mean benefit: k=8 delta `−0.000330 ± 0.000790`; k=32 `+0.000020 ± 0.000504`, while k=32 increased mean runtime.

## Weak Negative Evidence

- Removing `dur_bucket` produced paired delta `−0.000591 ± 0.000156`; the effect is consistent but below the project's `0.002` practical convergence epsilon.
- Nine fixed, standalone smoothed video-stat ratios did not beat train item popularity; long-play/show was essentially tied (`−0.000344` primary), but this does not test incremental model value.

## Inconclusive Questions

- The 8-field item-static FM expansion and nearby learning rates produced deltas within seed noise.
- Whether validation should motivate recency is unresolved because drift measures disagree.
- The exact one-month video-statistic cutoff and causal safety are not documented.
- No multi-task architecture was tested; target density/association evidence alone cannot predict transfer.
- Whether random-validation diagnostics improve standard-traffic model selection was not tested.

## Dataset Opportunities Not Yet Tested

- Dense prior click/play-time histories for most warm users.
- Broad user–tag history and tag-level repeat support.
- Multiple auxiliary feedback targets with sharply different prevalence and dependence.
- Video statistic ratios/aggregates, conditional on establishing a causal cutoff.
- Daily/hour/tab context and measurable temporal drift.
- Random-exposure data restricted to temporally safe diagnostics.

These are availability statements, not recommendations.

## Engineering Constraints

- Windows 11 process trees require recursive descendant termination; killing only a parent is insufficient as a design assumption.
- Python 3.13 on an ARMv8 Windows machine may limit compatibility with older research code; the Starter Kit itself needs only NumPy.
- A cold official FM costs about 79 seconds; k=32 costs about 73 seconds of training versus 51 seconds for k=16 in the controlled matrix.
- A safe cache should include a version and content fingerprint, not only timestamps.
- Peak profiling RSS was about 1.41GB, and observed system memory use was high; avoid unnecessary concurrent heavy runs.
- The final harness/pipeline/agent layer is not implemented; only research scripts are executable. This gap spans all 15 files under `harness/`, `pipeline/`, and `agent/`, not only the 7 files originally probed by J02 (review correction).

## Questions the Autonomous Agent Should Resolve Itself

- Which metric bottleneck it believes is most valuable and why.
- Whether and how to use history, auxiliary tasks, ranking-aware losses, temporal context, or random-exposure diagnostics.
- How to handle cold users and the difference between exact-item and tag-level history.
- Whether video statistics can be made causally defensible.
- Whether a field result from the exact FM transfers to another model family.
- What experiment has the highest expected information gain under runtime and convergence limits.

# Candidate Findings for Human Review

## Candidate 1 — Validation metric invariance

Finding: `42.222%` of validation users have uniform labels and cannot be reordered to change their metric contribution.

Evidence class: `HARD FACT`.

Investigation ID: B01.

Numerical evidence: 6,785 all-negative (`30.321%`) + 2,663 all-positive (`11.901%`); 12,929 mixed users (`57.778%`).

Confidence: Very high.

Recommended wording: “On validation, 42.22% of users are ranking-invariant; GAUC weight and movable nDCG come from the remaining mixed-label users.”

What it establishes: The correct movable-user denominator.

What it does NOT establish: Any training reweighting or sampling rule.

## Candidate 2 — Metric headroom concentration

Finding: The largest aggregate activity-tier headroom is T4, and the largest aggregate list-length headroom is 6–10 impressions.

Evidence class: `HARD FACT`.

Investigation ID: B02, B03.

Numerical evidence: T4 has 40.01% GAUC weight and 0.07068 overall nDCG-gap contribution; lists 6–10 have 36.39% GAUC weight and 0.06125 gap contribution.

Confidence: Very high for the official-FM baseline.

Recommended wording: “Under the reproduced FM, T4 users and validation lists of length 6–10 contain the largest aggregate combination of metric weight and movable nDCG gap.”

What it establishes: Where current baseline headroom is concentrated.

What it does NOT establish: That bucket-specific modeling or weighting helps.

## Candidate 3 — Entity versus relationship overlap

Finding: Validation entities are warm, but exact user–candidate relationships are mostly unseen.

Evidence class: `HARD FACT`.

Investigation ID: A02.

Numerical evidence: 98.11% users, 99.88% videos, and 99.91% authors seen; only 1.63% unique user–video and 3.38% user–author pairs seen.

Confidence: Very high.

Recommended wording: “Validation is warm at entity level but cold at exact user–video/user–author relationship level.”

What it establishes: Memorized pair coverage is low despite entity overlap.

What it does NOT establish: Which representation generalizes best.

## Candidate 4 — Historical support differs by granularity

Finding: General histories are common, exact candidate histories are rare, and tag histories are broad.

Evidence class: `HARD FACT`.

Investigation ID: A03, F01.

Numerical evidence: 85.17% users have ≥10 prior rows; validation row prior same video/author/tag coverage is 1.62%/3.38%/78.41%; 55.25% of train user–tag pairs repeat.

Confidence: High; tag parsing is comma-separated integer tokens from official basic features.

Recommended wording: “Train-derived history is broadly available, but candidate-level repeats are scarce; tag-level history has 78.41% validation-row coverage.”

What it establishes: Feasibility and support at different history granularities.

What it does NOT establish: That a historical or tag-aware model improves score.

## Candidate 5 — Auxiliary-signal density

Finding: Click and play time are dense; rare engagement actions have limited per-user history.

Evidence class: `HARD FACT`.

Investigation ID: E01, F01.

Numerical evidence: train click 46.34%, like 1.87%, follow 0.10%, comment 0.26%, forward 0.10%, hate 0.04%; only 2.24% of validation users have ≥10 prior likes and ≤0.04% have ≥10 of the rare actions.

Confidence: Very high.

Recommended wording: “Auxiliary signals vary by orders of magnitude in density; click/play-time histories are broad, while repeated rare-action histories are uncommon.”

What it establishes: Task-support and imbalance constraints.

What it does NOT establish: Multi-task benefit, weights, or architecture.

## Candidate 6 — Video/author redundancy in the exact FM

Finding: `video_id` and `author_id` are structurally redundant, and using both under the exact baseline FM was consistently lower than using either alone.

Evidence class: `STRONG NEGATIVE` against the exact dual-ID formulation.

Investigation ID: A04, C02.

Numerical evidence: 100% functional video→author mapping; 87.05% authors have one observed video; five-seed paired deltas are +0.001316 ± 0.000426 after removing author and +0.001082 ± 0.000585 after removing video, all positive.

Confidence: High for this exact FM.

Recommended wording: “In the official pointwise FM, jointly using video and author identity is empirically redundant; either four-field ablation outperformed the five-field baseline in all five paired seeds.”

What it establishes: Narrow negative evidence for the exact field/model combination.

What it does NOT establish: That either ID is universally removable in other models.

## Candidate 7 — `tab` field importance

Finding: Removing `tab` materially damages the exact baseline.

Evidence class: `STRONG NEGATIVE` against removal.

Investigation ID: C02.

Numerical evidence: paired validation primary delta `−0.015903 ± 0.000467` over three seeds.

Confidence: Very high.

Recommended wording: “`tab` carries distinct signal in the official FM; its removal costs about 0.0159 primary across paired seeds.”

What it establishes: `tab` is not redundant in this baseline.

What it does NOT establish: How `tab` should be represented elsewhere.

## Candidate 8 — Static feature stuffing

Finding: The exact 13-field CWM-style static expansion has no meaningful benefit and is lower on validation.

Evidence class: `STRONG NEGATIVE`.

Investigation ID: D01.

Numerical evidence: primary `0.599930 ± 0.000523`; paired delta `−0.001510 ± 0.000792`, all three seeds lower.

Confidence: High for the tested formulation.

Recommended wording: “Repeating the Starter Kit's exact 13-field static-feature FM expansion has low expected value.”

What it establishes: The exact formulation is reproducibly low-value.

What it does NOT establish: That all static or derived features are useless.

## Candidate 9 — FM dimension scaling

Finding: k=8/16/32 scaling does not provide a meaningful mean gain.

Evidence class: `STRONG NEGATIVE` against simple capacity scaling.

Investigation ID: D02.

Numerical evidence: k8 delta `−0.000330 ± 0.000790`; k32 `+0.000020 ± 0.000504`; k32 mean train time 72.9s vs k16 51.3s.

Confidence: High for tested values.

Recommended wording: “Simple FM embedding-size scaling across k=8/16/32 is low-value under the official training loop.”

What it establishes: No defensible benefit at these sizes.

What it does NOT establish: Limits of other model capacity or regularization.

## Candidate 10 — Video-statistics timing quarantine

Finding: The file is documented as one-month averages, but its exact cutoff relative to validation is undisclosed.

Evidence class: `INCONCLUSIVE` causal validity.

Investigation ID: G02.

Numerical evidence: 7,583/7,583 video coverage; documentation names one month but gives no endpoints/cutoff.

Confidence: Very high about the documentation gap.

Recommended wording: “Do not treat `video_features_statistic_pure.csv` as causally safe until its aggregation cutoff is established; use only under an explicit leakage decision.”

What it establishes: A required validation gate before feature use.

What it does NOT establish: That the source is definitely leaky or useless.

## Candidate 11 — Random-log temporal risk

Finding: Most random-exposure rows are on evaluation dates.

Evidence class: `HARD FACT`.

Investigation ID: I01.

Numerical evidence: 897,721 / 1,186,059 rows (`75.689%`) are dated April 29–May 8.

Confidence: Very high.

Recommended wording: “Random-log development use must apply an explicit April 28 cutoff; evaluation-period feedback is forbidden for selection.”

What it establishes: Temporal filtering requirement.

What it does NOT establish: Whether validation-period random rows should be used or how.

## Candidate 12 — Engineering readiness

Finding: Runtime primitives work, but the final harness, pipeline, and agent orchestration layer are not implemented.

Evidence class: `HARD FACT`.

Investigation ID: J01, J02.

Numerical evidence: timeout/tree/syntax/NaN probes pass; cold baseline `78.52s`; verified effective cache load `1.384s`. All 15 files under `harness/`, `pipeline/`, and `agent/` (the 7 originally probed plus `harness/logger.py`, `score.py`, `submission.py`, and all five `agent/*.py` files, confirmed by review) have zero executable non-comment lines; `reports/`, `submissions/`, `runlogs/`, `tests/` contain no files.

Confidence: Very high.

Recommended wording: “Before an autonomous run, implement and test the currently scaffold-only harness/pipeline/agent layer (all 15 files), including recursive Windows process termination and content-fingerprinted caching.”

What it establishes: A concrete engineering prerequisite.

What it does NOT establish: The design of the final autonomous agent.
