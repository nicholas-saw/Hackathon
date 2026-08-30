# KuaiRand-Pure — Official Problem Definition

> Purpose: give every AI agent a short, authoritative description of the competition task.
> This file should stay **read-only during autonomous runs**.

## 1. Challenge

Build an **Autonomous Machine Learning Research Agent for Recommender Systems**.

The agent should autonomously perform the research loop:

1. Read the problem.
2. Inspect the data.
3. Engineer features.
4. Train and tune.
5. Evaluate.
6. Reflect and revise.
7. Repeat until convergence.

The agent is judged not only on model quality, but also on how autonomously, robustly, and efficiently it performs this loop.

### The three official task requirements

1. **Reproduce the official baseline.** Stand up a working end-to-end pipeline and
   confirm it reaches the official baseline's reported *validation* score. The official
   baseline is the organizer-provided reference in the Starter Kit — not any starter
   pipeline the agent builds for itself, which is an internal step.
2. **Iterate on the pipeline.** Draw on established industry and academic methods to
   improve any stage of the loop, in code. Development uses the training split and
   public validation feedback only.
3. **Improve over the baseline.** Drive the validation score above the official
   baseline. Improvement need not be monotonic — the trajectory may fluctuate, as with
   real data — but the agent should show a sustained ability to keep improving.

---

## 2. Required Benchmark

**Dataset:** KuaiRand-Pure

KuaiRand-1k and KuaiRand-27k are optional bonus benchmarks. They are not required for the primary score.

### Official scale

- Train rows: **1,141,112**
- Validation rows: **124,909**
- Evaluation/test rows: **170,588**
- User catalogue: **27,285** rows in `user_features_pure.csv`
- Video catalogue: **7,583** rows in `video_features_basic_pure.csv` /
  `video_features_statistic_pure.csv`

Row counts are the number of log rows falling in each official date window; they are
obtained from the `date` column alone and do not require reading any label.

---

## 3. Official Task

### Target label

`long_view`

### Task form

**Within-user ranking over each user's logged impressions.**

This is not a full-catalog retrieval task.

For a given evaluation split, the model produces one score per logged impression. The evaluator ranks impressions **within each user**.

### Metrics

- **GAUC**
- **nDCG@5**

### Primary score

```text
primary = (GAUC + nDCG@5) / 2
```

The Starter Kit's `evaluate.py` is the scoring authority.

> **Known stale line in the official document.** The *Limits* row of section 2.3
> (Constraints & Scope) reads "KuaiRand-Pure: NDCG@10 / Recall@50, click = positive
> (fixed)". This is superseded. The Benchmarks table, the Judging Criteria section,
> Appendix A.4, and the shipped `evaluate.py` all pin the task to `long_view` with
> GAUC / nDCG@5. Appendix A.4 also explains why Recall was dropped: at roughly five
> impressions per user, Recall@50 is 0.999+ for every model including random scoring.
> Use the Starter Kit definition.

---

## 3a. Resource Policy — What the Agent May Use

Deliberately permissive. From section 2.3 and the Benchmarks resource policy:

**In scope**

- **Any** open-source library or framework — PyTorch, RecBole, TorchRec, LightGBM, and
  so on. The Starter Kit is numpy-only because it is a *minimal* reference, not because
  dependencies are restricted.
- Any papers, public solutions, or pretrained weights.
- Changes to **any** pipeline stage — features, model, training strategy, evaluation
  loop, and every upstream and downstream module. Improvements are explicitly not
  limited to the model.

**Out of scope — the one hard rule**

- **No external training data.** Training must rely only on the KuaiRand datasets.
  No augmenting, joining, or pre-training on any other dataset, and no pretrained model
  whose weights were trained on these benchmarks' test labels.
- No hidden-test access during development (train + validation only).

Compute is deliberately *not* the binding constraint: 100 iterations of the official
baseline take about 28 minutes on a single CPU core with no GPU. GPU-hours and LLM
tokens are reported for Feasibility scoring, not capped.

---

## 4. Official Date Splits

```text
TRAIN
2022-04-08 through 2022-04-21

VALIDATION
2022-04-22 through 2022-04-28

EVALUATION / TEST
2022-04-29 through 2022-05-08
```

The split boundaries must never be changed.

> Boundary clarification: the official train window opens on 2022-04-08, but the raw
> train file contains **no rows on that date**. Train rows span 13 dates,
> 2022-04-09 through 2022-04-21. The window is still the official one; only the data
> starts a day later. Do not "fix" this by shifting a boundary.

Development must use **train + validation only**.

Do not inspect or use evaluation/test labels for research, feature construction, model selection, hyperparameter tuning, hypothesis selection, or pre-audit conclusions.

---

## 5. Official Baseline

The organizer-provided baseline is a numpy Factorization Machine.

### Baseline fields

1. `user_id`
2. `video_id`
3. `author_id`
4. `tab`
5. `dur_bucket`

### Baseline configuration

```text
model: Factorization Machine
embedding dimension k: 16
learning rate: 0.001
```

Full published configuration (`baseline_scores.json`): `batch = 8192`,
`max_epochs = 40`, `patience = 4`, early stopping on validation primary.

### Published reference ladder

From `source/starter-kit/baseline_scores.json`. The FM row is the one to beat.

| Reference | valid GAUC | valid nDCG@5 | valid primary | test primary |
|---|---:|---:|---:|---:|
| random (sanity lower bound) | 0.4993 | 0.4675 | 0.4834 | 0.4753 |
| item popularity (trivial) | 0.6387 | 0.5227 | 0.5807 | 0.5715 |
| **FM (official baseline)** | **0.6674** | **0.5357** | **0.6016** | **0.5946** |

Published baseline seed standard deviation is approximately **0.0008** — reported by
the organizer as the standard deviation of the **test** GAUC / nDCG@5 / primary across
5 seeds. It is the reference noise scale behind the convergence rule in section 7.

The hidden-test numbers are reference information only. Do not use local hidden-test labels to compare development experiments.

---

## 6. Metric Details

### GAUC

- Calculated per user.
- Only users with both positive and negative labels contribute.
- Weighted by each contributing user's number of positives.

### nDCG@5

- Calculated per user.
- Users with no positive labels receive nDCG = 0.
- Users with all-positive labels are invariant to ranking.
- Only within-user relative order matters.

### Oracle note

The metrics do not span the full [0,1] interval for this dataset, because all-negative
users score nDCG = 0 under any model and all-positive users are ranking-invariant.

The organizer publishes a **split-specific** oracle (true labels used as scores):

| Split | oracle GAUC | oracle nDCG@5 | oracle primary |
|---|---:|---:|---:|
| validation | 1.0000 | 0.6968 | **0.8484** |
| test | 1.0000 | 0.7289 | **0.8645** |

Use the **validation** oracle (0.8484) as the denominator when judging development
progress; 0.8645 is the test-split figure and is reference information only.

### Evaluation-split composition (official, published)

The organizer reports that of the 23,875 test users, 27.1% are all-negative, 9.2% are
all-positive, and 63.7% are label-discriminative. This is published reference material,
not something to be re-derived from local evaluation labels.

---

## 7. Convergence and Run Limits

### Convergence rule

```text
epsilon = 0.002
N = 3
```

A run is converged when validation primary has not improved by more than `0.002` over the last **3 consecutive iterations**.

The organizer derives `epsilon = 0.002` as approximately 2.5 x the published 0.0008
seed standard deviation. Convergence is judged on **validation** primary only.

A run is converged when validation primary has not improved by more than epsilon over
the last 3 iterations, **or** when it hits the 50-iteration cap, **or** when it hits the
6-hour wall-clock ceiling — whichever comes first.

> **What gets scored is the converged result — not the peak, and not the trajectory.**
> The submission ranked is the **validation-best checkpoint at the point of
> convergence**, evaluated once on the hidden test set. Two consequences: the score
> locks at convergence, so running longer cannot raise it, while wall-clock keeps
> accruing against Feasibility.

### Hard limits

- Maximum **50 iterations** per benchmark run.
- Maximum **6 hours agent wall-clock** per run.

---

## 8. Judging Criteria

### Technical Execution — 35%

**Primary metric.** Scored on the hidden test set as the equal-weighted average of each
metric's *absolute improvement* over the official baseline:

```text
delta(m)       = score_agent(m) - score_baseline(m)
score_dataset  = mean over m of delta(m)
```

with `m` ranging over GAUC and nDCG@5. Falling short of the baseline is scored
continuously — a negative delta, not a disqualification. KuaiRand-Pure determines 100%
of this score; the bonus benchmarks add credit without reducing it.

**Robustness.** Judged on how a failure is *handled*, not on whether one occurs — a
capable agent may fail only on genuinely hard problems. What is scored is recovering,
retrying, or routing around a failed step (code error, timeout, unexpected input) so
that long runs neither crash, stall, nor diverge.

### Innovation & Problem Insight — 20%

What the **agent** identifies as worth trying and why.

### Impact & Relevance — 20%

Autonomy. Primarily measured by the number of manual interventions required to reach convergence.

### Feasibility & Practicality — 15%

Two scored measures:

- **Token consumption** — total input + output tokens across the agent's LLM calls.
- **Agent wall-clock** — total elapsed time to reach the converged result. This
  **replaces GPU-hours** as the scored compute measure: the reference pipeline needs no
  GPU, so GPU-hours would be ~0 for most teams and would only penalise whoever used one.
  Report GPU-hours if any were used, but wall-clock is what is scored.

Two rules make it comparable:

1. It is scored **only** among submissions whose hidden-test primary exceeds the official
   baseline. Without that gate the criterion would fight the Primary metric — an agent
   that stopped after three iterations would look cheapest and score worst.
2. It is graded in **three coarse tiers** (low / medium / high consumption), not as a
   continuous ranking.

Iteration count is reported as a deliverable but is not itself one of the two scored
measures.

### Presentation & Communication — 10%

Final-event criterion.

---

## 9. Deliverables

The project should ultimately provide:

- Public code repository
- README
- Per-iteration run logs containing:
  - hypothesis
  - code diff
  - metrics
  - errors and recovery events
- Manual-intervention summary
- Final submission
- Results summary
- Resource-consumption summary
- Detailed written report
- Optional ~3-minute video

Specifics the official deliverables list calls out:

- **Written project description** naming development tools, APIs, libraries/frameworks,
  and datasets/assets used.
- **README** covering project overview, setup and installation, steps to reproduce, a
  brief reflection on limitations and what you would improve with more time, and team
  member contributions.
- **Results table** reporting the validation-best GAUC / nDCG@5 and the absolute delta
  over the official baseline.
- **Resource usage** to reach convergence: total input + output tokens, total agent
  wall-clock, iterations used out of the 50 cap, and GPU-hours if any were used.
- The video is **not required** on this track; without one, a detailed report is
  strongly encouraged.

---

### Official submission contract

`source/starter-kit/submit.py` defines the accepted format:

```text
row_id,user_id,video_id,score
```

- `row_id` starts at 0 and increases contiguously, following the row order of
  `data.load()[split]` (train file read first, then the 04-22..05-08 file, filtered by
  date, original file order preserved).
- `user_id` / `video_id` are redundant alignment-check columns.
- `score` is any finite real number; only relative order within a user matters.
  `NaN` / `Inf` are rejected.
- `row_id` is required as the key because `(user_id, video_id)` is **not unique** in
  the evaluation split: the organizer reports 3.06% duplicated pairs, repeating up to
  12 times.

---

## 10. Core Design Principle

The final system should demonstrate:

> Given a recommender problem, the agent can inspect evidence, form its own reasonable hypotheses, implement experiments, evaluate results, reflect on failures and successes, recover from errors, and converge with minimal human intervention.

Humans may provide:

- task definition
- rules
- data
- codebase
- verified prior evidence
- research references
- budget

The agent should provide:

- interpretation
- research questions
- hypotheses
- prioritization
- experiment design
- code modifications
- evaluation
- reflection
- next-step decision
