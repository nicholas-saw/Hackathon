# PRE-AUDIT — Consolidated KuaiRand-Pure

> Consolidates three independently reviewed pre-audits. It records evidence, not a prescribed modeling strategy. All local outcome measurements below use train and/or validation only. Published test facts are reference material and are not development evidence.

## 0. Scope and Rules

- Official task: predict `long_view` scores for within-user ranking over logged impressions.
- Official metrics: GAUC and nDCG@5; `primary = (GAUC + nDCG@5) / 2`.
- Official dates: train 2022-04-08..21, validation 2022-04-22..28, evaluation 2022-04-29..05-08. The raw train file has no row on 04-08.
- `source/starter-kit/evaluate.py` controls scoring. GAUC includes only mixed-label users and weights them by positive count; nDCG@5 averages users equally.
- Current-row feedback is INVALID / FORBIDDEN as an input for current-row `long_view`. Strictly earlier feedback may be historical data; feedback may also be an auxiliary target or diagnostic.
- No source file, Starter Kit file, or `context/constraints.md` was changed during consolidation.

## 1. Baseline Reproduction

### Investigation R00 — Official FM validation reproduction

### Question

Does the local environment reproduce the official validation baseline closely enough to support the audit?

### Data / scope

Official train and validation rows, train-fitted encoding, `video_features_basic_pure.csv`, and the unchanged Starter Kit FM/evaluator. No evaluation label.

### Method

Run the official five-field FM mechanism with seed 0, k=16, lr=0.001, batch 8,192, maximum 40 epochs, and patience 4. Cross-check against the published validation result and a five-seed validation-only rerun.

### Result

| Metric | Published validation | Reproduced seed 0 |
|---|---:|---:|
| GAUC | 0.6674 | 0.667133 |
| nDCG@5 | 0.5357 | 0.535806 |
| Primary | 0.6016 | 0.601470 |

Seed 0 selected epoch 7 and stopped after epoch 11. A separate five-seed validation-only reproduction reported mean primary 0.60157 and population std 0.00032. The organizer’s published 0.0008 seed std remains the safer generic noise reference.

### Evidence classification

HARD FACT.

### Interpretation

The official validation score and split sizes reproduce within seed/rounding variation.

### What it DOES NOT establish

It does not validate any local test result, alternative model, or optional feature source.

### Source provenance

- Audit 1 R00 and Review 1 verification
- Audit 3 baseline reproduction/C02 and Review 3 verification
- Official Starter Kit `baseline.py`, `evaluate.py`, and `baseline_scores.json`

## 2. Dataset Structure

### Investigation A01 — Cardinality, coverage, activity, and missingness

### Question

What is the development-data scale, side-table coverage, activity distribution, and missingness?

### Data / scope

Train and validation standard logs; full user, video-basic, and video-statistic tables.

### Method

Exact row counts, unique counts, per-user counts, missing-value rates, and entity joins. Evaluation rows were counted only where the official split fact required it; their outcome columns were not used.

### Result

| Measurement | Train | Validation |
|---|---:|---:|
| Rows | 1,141,112 | 124,909 |
| Users | 26,210 | 22,377 |
| Videos | 7,538 | 5,951 |
| Authors | 6,482 | 5,315 |
| Tabs | 15 | 15 |
| Dates with rows | 13 | 7 |

The user table has 27,285 rows and both video tables have 7,583 rows; they cover 100% of train/validation users or videos. Train impressions per user have median 31, p90 97, p99 207, and max 809. Validation list length has median 4, p90 12, p99 26, and max 74.

The logs and video-statistic table have no missing cells. Selected missingness: user `onehot_feat4` 3.2032%; `onehot_feat12..17` 2.6168% each; basic video duration 3.1518%; music type 2.6770%; tag 1.2660%. `visible_status` is constant.

### Evidence classification

HARD FACT.

### Interpretation

Side tables are join-feasible and missingness is localized. The corrected all-train-user activity median is 31; the distinct value 35 later in this report is the median train history among validation users.

### What it DOES NOT establish

Coverage and cardinality do not establish usefulness, encoding, or causal validity.

### Source provenance

- Audit 1 A01 and Review 1
- Audit 2 A01/C01 and Review 2 corrections to activity percentiles
- Audit 3 A01/A02 and Review 3

## 3. Entity Overlap and Redundancy

### Investigation A02 — Warm entities, novel relationships, and author redundancy

### Question

How much of validation is warm at entity and relationship levels, and how redundant are video and author identity?

### Data / scope

Train/validation identifiers joined to the full video-basic table. Raw tag strings treat missing as an explicit category; parsed-tag-token results are reported separately.

### Method

Exact set intersections, pair counts, within-split repeat counts, and video→author functional-dependency checks.

### Result

| Validation object | Seen in train |
|---|---:|
| Users | 21,955 / 22,377 (98.114%) |
| Videos | 5,944 / 5,951 (99.882%) |
| Authors | 5,310 / 5,315 (99.906%) |
| Unique user–video pairs | 1,974 / 121,337 (1.627%) |
| Unique user–author pairs | 4,081 / 120,885 (3.376%) |
| Raw user–tag-string pairs | 61,405 / 90,121 (68.14%) |

Every video maps to one author. In the full basic file, 5,661/6,510 authors (86.96%) have exactly one video; median videos/author is 1 and maximum is 26. Restricted to train/validation-observed videos, the corresponding fraction is 5,647/6,487 (87.051%) and maximum 24.

Within train, 4.130% of unique user–video pairs repeat and account for 8.194% of rows; user–author figures are 5.913% and 11.750%. With corrected explicit-missing raw-tag semantics, 51.77% of train user–tag pairs and 24.45% of validation user–tag pairs repeat, accounting for 84.98% and 45.49% of rows. An alternative parsed-token construction yields 71.913% validation pair overlap and 78.413% validation-row coverage; it is not the same definition.

### Evidence classification

HARD FACT.

### Interpretation

Validation is overwhelmingly warm at the entity level but mostly novel at exact user–video and user–author levels. Author and video identity are structurally near-redundant for most of the catalog. Coarser tag relationships have much broader support, with results depending on tag parsing.

### What it DOES NOT establish

It does not establish that any field should be removed, that tag features help, or that sparse exact-pair history is useless.

### Source provenance

- Audit 1 A02–A04 and Review 1
- Audit 2 A01/D01 and Review 2
- Audit 3 A03–A05 and Review 3 correction to tag missing semantics

## 4. Metric Structure

### Investigation B01 — Movable users and oracle ceiling

### Question

Which validation users can affect rank-sensitive metrics, and what is the mathematical validation ceiling?

### Data / scope

Validation `long_view` labels and the official evaluator.

### Method

Compute per-user list length and positive count. Use labels as scores only to compute the oracle ceiling.

### Result

| User type | Users | % users | Rows | % rows |
|---|---:|---:|---:|---:|
| All negative | 6,785 | 30.321% | 21,807 | 17.458% |
| All positive | 2,663 | 11.901% | 4,540 | 3.635% |
| Mixed / movable | 12,929 | 57.778% | 98,562 | 78.907% |

There are 3,917 single-impression users (17.505%). The reproduced validation oracle is GAUC 1.0000, nDCG@5 0.6968, primary 0.848393. All-negative users always receive nDCG 0; all-positive users are ranking-invariant; neither group contributes to GAUC.

### Evidence classification

HARD FACT.

### Interpretation

42.222% of validation users have invariant nDCG, while mixed users contain 78.907% of validation rows.

### What it DOES NOT establish

It does not prescribe training filters, user weights, or an optimization strategy.

### Source provenance

- Audit 1 B01 and Review 1
- Audit 2 B01 and Review 2
- Audit 3 B03 and Review 3
- Official `evaluate.py`

## 5. Activity / List-Length Analysis

### Investigation B02 — Activity tiers, list length, and metric concentration

### Question

Where does the reproduced seed-0 baseline-to-oracle gap lie, and how related are prior activity and validation list length?

### Data / scope

Train-side interaction counts for validation users, validation labels, and full official-configuration seed-0 baseline predictions. Consolidated activity tiers use quartiles among warm validation users: Cold 0, T1 1–17, T2 18–36, T3 37–65, T4 66+.

### Method

Evaluate disjoint activity and list-length buckets with the official evaluator. GAUC shares use the 34,592 positives belonging to mixed-label users only. nDCG gap contribution uses equal user weights. Cross-tab all 5×6 activity/list cells and reconcile totals.

### Result

| Activity | Users | Rows | GAUC | nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 422 | 1,990 | 0.6877 | 0.5305 | 0.6422 | 1.69% |
| T1 | 5,713 | 18,419 | 0.6515 | 0.5352 | 0.6224 | 14.67% |
| T2 | 5,480 | 24,752 | 0.6723 | 0.5450 | 0.6721 | 21.35% |
| T3 | 5,373 | 31,501 | 0.6633 | 0.5521 | 0.7288 | 27.50% |
| T4 | 5,389 | 48,247 | 0.6725 | 0.5112 | 0.7731 | 34.79% |

| List length | Users | GAUC | nDCG@5 | Oracle nDCG | GAUC weight |
|---|---:|---:|---:|---:|---:|
| 1 | 3,917 | 0.5000* | 0.4054 | 0.4054 | 0.00% |
| 2–3 | 6,218 | 0.6472 | 0.5413 | 0.6086 | 10.27% |
| 4–5 | 4,119 | 0.6645 | 0.6185 | 0.7492 | 16.36% |
| 6–10 | 5,225 | 0.6756 | 0.5913 | 0.8536 | 36.39% |
| 11–20 | 2,346 | 0.6677 | 0.5037 | 0.9182 | 27.08% |
| 21+ | 552 | 0.6596 | 0.3934 | 0.9420 | 9.90% |

`*` No length-1 user enters GAUC; 0.5 is the evaluator’s empty-denominator fallback.

Train activity and validation list length have Spearman correlation 0.4620. T3/T4 users with lists of 6+ comprise 25.38% of users, 50.79% of official GAUC weight, and 51.72% of the current seed-0 primary gap. All 30 cells reconcile to 22,377 users, 124,909 rows, 100% of GAUC weight, and the full baseline-to-oracle gaps.

### Evidence classification

HARD FACT for this reproduced baseline and these fixed bucket definitions.

### Interpretation

Metric weight and baseline headroom are unevenly distributed, and activity/list length are related but not interchangeable.

### What it DOES NOT establish

It does not establish causality, attainable gain, or that a particular weighting/loss/model will close the gap. Alternative audit bucket boundaries are not contradictory but were omitted to keep one authoritative definition.

### Source provenance

- Audit 1 B02/B03 and Review 1 (official-denominator cross-check; alternate activity tiers)
- Audit 2 B01 and Review 2 (confound identified; older short-run scores superseded)
- Audit 3 B01–B06 and Review 3 corrected denominator/joint extension

## 6. Baseline Mechanism and Ablations

### Investigation C01 — Exact FM mechanism and controlled ablations

### Question

What does the official FM optimize, and which conclusions survive controlled validation-only repeats?

### Data / scope

Official source; train/validation only. Three matched seeds per configuration unless five are stated.

### Method

Inspect official code and change one field/configuration at a time with the same evaluator and early stopping. Report population std and paired deltas where available.

### Result

The FM uses `user_id`, `video_id`, `author_id`, `tab`, and a ten-bin train-quantile `dur_bucket`; first-order terms plus all pairwise embedding interactions; pointwise binary cross-entropy; k=16; Adam for W/V, a plain bias update, lr=0.001, L2=1e-6, batch 8,192, and validation-primary early stopping.

| Variant | Seeds | Primary mean ± std | Paired delta mean ± std | Final class |
|---|---:|---:|---:|---|
| Base | 3 | 0.601440 ± 0.000275 | reference | HARD FACT |
| Remove `tab` | 3 | 0.585538 ± 0.000429 | −0.015903 ± 0.000467 | STRONG NEGATIVE EVIDENCE against removal |
| Remove `dur_bucket` | 3 | 0.600849 ± 0.000225 | −0.000591 ± 0.000156 | INCONCLUSIVE |
| Remove `author_id` | 5 | 0.602889 ± 0.000451 | +0.001316 ± 0.000426 | WEAK NEGATIVE EVIDENCE against dual-ID formulation |
| Remove `video_id` | 5 | 0.602654 ± 0.000307 | +0.001082 ± 0.000585 | WEAK NEGATIVE EVIDENCE against dual-ID formulation |
| Add three item-static fields (8 total) | 3 | 0.601108 ± 0.000461 | −0.000332 ± 0.000205 | INCONCLUSIVE |
| Add full static bundle (13 total) | 3 | 0.599930 ± 0.000523 | −0.001510 ± 0.000792 | STRONG NEGATIVE EVIDENCE for exact formulation |

The verified static configurations contain 8 and 13 fields.

### Evidence classification

As shown in the table. The pointwise/ranking-objective mismatch is a HARD FACT, not positive evidence for an alternative loss.

### Interpretation

`tab` carries distinct value in this FM. The exact dual video/author-ID formulation underperforms either tested four-field alternative across five matched seeds, but the effect is small relative to the project’s 0.002 practical epsilon. The full tested static bundle is reproducibly low-value.

### What it DOES NOT establish

It does not establish that video, author, duration, or static information is generally useless in other models, encodings, or objectives.

### Source provenance

- Audit 1 C01/C02/D01 and Review 1
- Audit 3 C01/C04 and Review 3 conservative reclassification and schema correction
- Official Starter Kit source

## 7. Post-Impression Feedback Structure

### Investigation D01 — Feedback density and same-row association

### Question

Which feedback outcomes are dense, how are they associated with `long_view`, and what use is forbidden?

### Data / scope

Current-row train/validation feedback for diagnostics only. No feedback value was used as a model input.

### Method

Compute prevalence/means and Pearson associations; use log1p where explicitly stated for skewed continuous joint-correlation analysis.

### Result

| Signal | Train mean | Validation mean | Validation r with `long_view` |
|---|---:|---:|---:|
| `is_click` | 46.345% | 44.383% | 0.7515 |
| `is_like` | 1.868% | 1.797% | 0.0949 |
| `is_follow` | 0.101% | 0.130% | 0.0253 |
| `is_comment` | 0.257% | 0.233% | 0.0587 |
| `is_forward` | 0.100% | 0.078% | 0.0245 |
| `is_hate` | 0.042% | 0.062% | −0.0038 |
| `is_profile_enter` | 2.539% | 1.945% | 0.1271 |
| `play_time_ms` | 23,260.5 | 21,486.8 | 0.6319 raw |

Validation `is_follow` is 163/124,909 = 0.130495%, correctly rounded to 0.130%. Validation click/play-time inter-correlation is 0.5167. Comment/comment-stay correlation is 0.3029. Profile stay is over 99.98% zero; comment stay is about 95% zero.

### Evidence classification

HARD FACT for density and association. INVALID / FORBIDDEN for using any current-row outcome as a current-row input.

### Interpretation

Click and play time are dense and strongly associated with the target, partly because their definitions share watch-time structure. Other active-feedback outcomes are sparse.

### What it DOES NOT establish

It does not establish positive transfer, auxiliary loss weights, architecture, or usefulness as strictly historical features.

### Source provenance

- Audit 1 E01/E02 and Review 1 rounding correction
- Audit 2 E01 and Review 2 wording correction
- Audit 3 D01–D03 and Review 3 correlation correction
- Official feedback-leakage rules

## 8. Historical Information Availability

### Investigation E01 — Strictly prior history coverage

### Question

How much train-period history exists before validation, and at what granularity?

### Data / scope

Training rows as history; validation users/items as candidates. Every train timestamp precedes every validation timestamp.

### Method

Count train interactions and outcomes per validation user; measure validation-row coverage by prior same video, author, and tag.

### Result

| Measurement | Value |
|---|---:|
| Validation users with ≥1 / ≥5 / ≥10 train interactions | 98.114% / 92.854% / 85.168% |
| Median / mean / p90 train interactions per validation user | 35 / 47.42 / 103 |
| Validation rows with prior same video | 1.624% |
| Validation rows with prior same author | 3.381% |
| Validation rows with a prior parsed tag token | 78.413% |

Users with ≥1/≥5/≥10 prior clicks are 96.157%/82.531%/66.309%; corresponding like coverage is 23.229%/4.683%/2.239%. Follows, comments, forwards, and hates are much sparser. In a separate availability diagnostic, 81.57% of validation rows have a strictly earlier same-user validation timestamp; tied timestamps are not ordered, and 5.60% of rows occur in non-unique user/timestamp groups.

### Evidence classification

HARD FACT for train-derived coverage. INCONCLUSIVE for a deployable within-validation online-history protocol.

### Interpretation

General and click/play histories are broadly available, while exact item/author repeats and rare-action histories are sparse. History support depends strongly on activity and granularity.

### What it DOES NOT establish

It does not establish that aggregates, sequence models, tag attention, or within-period outcome updates improve validation. KuaiRand-Pure also has incomplete sequential logs.

### Source provenance

- Audit 1 F01 and Review 1
- Audit 2 F01 and Review 2 population clarification
- Audit 3 D04/D05 and Review 3 tied-timestamp correction

## 9. Video Basic / Statistic Features

### Investigation F01 — Inventory, redundancy, marginal association, and timing uncertainty

### Question

What distinct video information exists, and are the supplied statistic features causally interpretable for validation?

### Data / scope

Full video-basic/statistic files and train/validation logs.

### Method

Inspect cardinality/missingness; join durations; compute numeric Spearman redundancy; reconstruct `show_cnt × counts`; compare with observed train+validation impressions; evaluate fixed ratios as marginal or standalone validation diagnostics.

### Result

Basic duration exactly matches logged `duration_ms` wherever nonmissing; `visible_status` is constant. Fifty-four numeric statistic-field pairs have |Spearman|≥0.95; examples include like count/users 0.999865 and play count/users 0.999000.

`show_cnt × counts` is near-integer for every video. The reconstructed/observed train+validation impression ratio has median 11,465×, p10 5,248×, p90 38,199×, and is never below 1. The official description says per-day/per-scenario averages over one month, but does not identify exact endpoints or source population.

The smoothed long-time-play/show ratio has validation correlation 0.302 with `long_view` and bottom/top quintile rates 0.105/0.505. As an exact standalone score it produces primary 0.580378, delta −0.000344 versus train item popularity 0.580722. Other fixed standalone ratios are lower. The like-ratio trend rises through Q4 and dips slightly in Q5; it is not fully monotonic.

### Evidence classification

- Inventory, redundancy, reconstruction, association, and exact standalone scores: HARD FACT.
- Exact fixed-ratio standalone scorers: WEAK NEGATIVE EVIDENCE.
- Aggregation population/window and causal safety: INCONCLUSIVE.

### Interpretation

The files are complete but redundant, and some ratios have strong marginal target association. Their scale suggests a larger external population/window, but that inference does not establish timing safety.

### What it DOES NOT establish

It does not establish the window endpoint, evaluation-period overlap, incremental value over identity, or whether the statistics are allowed/safe as causal features.

### Source provenance

- Audit 1 G01/G02 and Review 1
- Audit 2 G01 and Review 2 corrected statistic means
- Audit 3 E01/E02 and Review 3 causal-safety ruling

## 10. Temporal Structure

### Investigation G01 — Daily volume and multidimensional drift

### Question

How does train volume evolve, and is validation uniformly closer to early or late train?

### Data / scope

Train and validation dates, labels, IDs, duration, and tab.

### Method

Daily counts and period summaries. Early train is 04-09..14; late train is 04-15..21; validation is 04-22..28.

### Result

The nominal 04-08 train date has zero rows. Volume peaks at 278,835 rows on 04-11 and falls to 20,021 on 04-21, a 13.9× ratio with small daily reversals.

| Period | Rows | Rows/day | `long_view` rate | Mean duration |
|---|---:|---:|---:|---:|
| Early train | 891,418 | 148,570 | 0.33228 | 98,553 ms |
| Late train | 249,694 | 35,671 | 0.35211 | 95,477 ms |
| Validation | 124,909 | 17,844 | 0.31328 | 102,820 ms |

Validation is closer to early train in target rate (gap 0.01900 vs 0.03882) and mean duration (4,267 vs 7,343 ms), but closer to late train in tab distribution, volume, and some entity-set measures.

### Evidence classification

HARD FACT for component measurements; INCONCLUSIVE for the combined claim that validation “resembles late train more.”

### Interpretation

Temporal change is real and multidimensional.

### What it DOES NOT establish

It does not establish whether recency weighting, date features, or dropping early rows helps.

### Source provenance

- Audit 1 H01 and Review 1
- Audit 2 H01 and Review 2 expansion/correction
- Audit 3 A06/H01 and Review 3

## 11. Random-Exposure Audit

### Investigation H01 — Validation-period random-exposure structure

### Question

What part of the random log is eligible for development diagnostics, and how different is it from standard validation?

### Data / scope

Date-only counts for the entire random file. Outcomes/features only for 2022-04-22..28. No evaluation-period outcome or feature was retained.

### Method

Filter by date before materializing non-date columns; compare validation-period random rows with standard validation entities and pairs.

### Result

The file has 1,186,059 rows over 04-22..05-08. Validation dates contain 288,338 rows; evaluation dates contain 897,721 rows, counted by date only. The validation slice has 19,091 users, 7,546 videos, and `long_view` rate 0.08056. It shares 17 of its 288,328 unique user–video pairs (0.006%) with standard validation, whose target rate is 0.31328.

Older figures based on full-file entity/pair inspection or locally inspected evaluation outcomes are excluded because they cross the permitted development scope.

### Evidence classification

HARD FACT for retained validation-slice/date-only measurements. INVALID / FORBIDDEN for evaluation-period outcomes as development evidence. INCONCLUSIVE for diagnostic/model-selection usefulness.

### Interpretation

The eligible validation-period random stream is distributionally distinct and almost pair-disjoint from standard validation.

### What it DOES NOT establish

It does not establish a propensity estimator, unbiased replacement metric, training use, or predictive validity for standard traffic.

### Source provenance

- Audit 1 I01 and Review 1 (label-free full-file audit; narrowed here)
- Audit 2 I01 and Review 2 contamination correction
- Audit 3 F01 and Review 3 controlling validation-slice results

## 12. Engineering Feasibility

### Investigation I01 — Runtime, cache, failure handling, and repository readiness

### Question

What has been demonstrated about iteration cost and recovery, and what remains unimplemented?

### Data / scope

Two audited local pipeline implementations on Windows 11/Python 3.13.7; synthetic subprocess/submission probes; repository file inventory.

### Method

Time load/encode/train stages; write and reload caches; compare arrays/fingerprints; test a child/grandchild timeout and error detection; inspect executable lines in harness/pipeline/agent files.

### Result

One reviewer rerun measured about 57.5s cold baseline time (2.99s load, 4.81s encode, 49.7s train), 0.018s cache read, and bit-identical arrays. A separate fingerprinted implementation measured 78.52s cold time and 1.384s for full-content-fingerprint plus cache read; it rejected a changed source fingerprint. These are implementation/run-specific, not contradictory stable benchmarks.

On the tested Windows inherited-pipe process tree, `subprocess.run(timeout=3, capture_output=True)` returned only after the grandchild’s full 30.13s lifetime. A separate recursive-termination probe successfully removed parent and child. The official submission reader rejects NaN/Inf, and syntax errors return a detectable nonzero status.

All 15 files under `harness/`, `pipeline/`, and `agent/` contain zero executable non-comment lines. The original probe counted only seven; Review 1 confirmed the additional eight.

### Evidence classification

HARD FACT for recorded runs and file state; ENGINEERING CONSTRAINT for bare Windows subprocess timeout and the unimplemented system layer.

### Interpretation

The baseline is CPU-manageable and caching can be deterministic, but safe process-tree control must be explicit and the final autonomous pipeline is not implemented.

### What it DOES NOT establish

It does not validate a six-hour autonomous run, resume/checkpoint behavior, complex-model runtime, or a production cache policy.

### Source provenance

- Audit 1 J01/J02 and Review 1 inventory correction
- Audit 2 engineering profile and Review 2
- Audit 3 G01–G04 and Review 3

## 13. Model / Objective Evidence

### Investigation J01 — Capacity and learning-rate evidence

### Question

Does simple FM width scaling or a nearby learning-rate change materially improve the baseline?

### Data / scope

Train/validation, official FM/evaluator, three seeds per configuration.

### Method

Change only k or lr and compare validation primary.

### Result

| Setting | Mean primary ± population std |
|---|---:|
| k=8 | 0.60111 ± 0.00080 |
| k=16 | 0.60144 ± 0.00027 |
| k=32 | 0.60146 ± 0.00069 |
| k=64 | 0.60099 ± 0.00044 |
| lr=0.0003 | 0.60179 ± 0.00011 |
| lr=0.001 | 0.60144 ± 0.00027 |
| lr=0.003 | 0.60009 ± 0.00084 |
| lr=0.01 | 0.59709 ± 0.00053 |

Separate audited values for lr=0.0005 (0.601776 ± 0.000280) and lr=0.002 (0.601364 ± 0.000826) also show no defensible nearby-rate gain.

### Evidence classification

STRONG NEGATIVE EVIDENCE against meaningful gain from simple k=8/16/32/64 width scaling in this FM; WEAK NEGATIVE EVIDENCE for the tested high learning rates, principally the clear 0.01 degradation; INCONCLUSIVE among nearby rates.

### Interpretation

Simple capacity scaling did not move validation primary beyond run variability. The training objective remains pointwise while evaluation is rank-based, but no alternative objective was tested.

### What it DOES NOT establish

It does not rule out other model families, regularization, schedules, or pairwise/listwise objectives.

### Source provenance

- Audit 1 D02 and Review 1
- Audit 3 C03/C05 and Review 3 conservative wording
- Organizer evidence in `context/constraints.md` C6

## 14. Evidence Summary

### HARD FACT

- Official split/metric semantics; reproduced validation FM; core cardinalities and low missingness.
- Warm entities but rare exact relationships; near-redundant author/video mapping.
- Validation invariant-user composition and oracle ceiling.
- Corrected activity/list-length metric decomposition using the official GAUC denominator.
- Feedback density/association, strictly prior train-history availability, video-field redundancy, temporal measurements, validation-period random-log structure, and recorded engineering behavior.

### STRONG NEGATIVE EVIDENCE

- Removing `tab` from the exact official FM.
- The exact 13-field static stuffing formulation.
- Simple FM width scaling across k=8/16/32/64.

### WEAK NEGATIVE EVIDENCE

- The exact dual `video_id`+`author_id` FM formulation relative to either tested four-field alternative.
- Exact video-stat ratio standalone scorers.
- Tested high FM learning rates, especially lr=0.01.

### INCONCLUSIVE

- `dur_bucket` removal; item-only 8-field expansion; nearby learning rates.
- Video-statistic aggregation window/population/causal safety.
- A single early-vs-late train similarity verdict; benefits of recency.
- Usefulness of the validation random log; deployable within-validation online history.
- Every untested historical, sequence, multi-task, pairwise/listwise, and alternative-model formulation.

### ENGINEERING CONSTRAINT

- Bare Windows timeout did not bound the tested child/grandchild process tree.
- The 15-file harness/pipeline/agent layer is comment-only scaffolding.

### INVALID / FORBIDDEN

- Locally derived standard-test metrics or evaluation-period random-log outcomes as development evidence.
- Current-row click, play time, or any other post-impression outcome as an input for the same row’s `long_view`.
