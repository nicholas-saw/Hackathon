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

---

## 2. Required Benchmark

**Dataset:** KuaiRand-Pure

KuaiRand-1k and KuaiRand-27k are optional bonus benchmarks. They are not required for the primary score.

### Official scale

- Train rows: **1,141,112**
- Validation rows: **124,909**
- Evaluation/test rows: **170,588**
- Roughly **27K users**
- Roughly **7.6K videos**

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

> Note: the official information document contains one stale line mentioning click / NDCG@10 / Recall@50. The Starter Kit, benchmark table, judging section, and appendix consistently pin the actual task to `long_view` with GAUC / nDCG@5. Use the Starter Kit definition.

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

### Published validation result

```text
GAUC      ≈ 0.6674
nDCG@5    ≈ 0.5357
primary   ≈ 0.6016
```

### Published hidden-test result

```text
GAUC      ≈ 0.6610
nDCG@5    ≈ 0.5282
primary   ≈ 0.5946
```

Published baseline seed standard deviation is approximately **0.0008**.

The hidden-test baseline number is reference information only. Do not use local hidden-test labels to compare development experiments.

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

The metrics do not span the full [0,1] interval for this dataset.

The organizer reports an oracle primary near **0.8645**, not 1.0.

---

## 7. Convergence and Run Limits

### Convergence rule

```text
epsilon = 0.002
N = 3
```

A run is converged when validation primary has not improved by more than `0.002` over the last **3 consecutive iterations**.

### Hard limits

- Maximum **50 iterations** per benchmark run.
- Maximum **6 hours agent wall-clock** per run.

---

## 8. Judging Criteria

### Technical Execution — 35%

Model performance and robustness.

### Innovation & Problem Insight — 20%

What the **agent** identifies as worth trying and why.

### Impact & Relevance — 20%

Autonomy. Primarily measured by the number of manual interventions required to reach convergence.

### Feasibility & Practicality — 15%

Agent resource consumption:

- LLM input + output tokens
- agent wall-clock
- iteration count
- GPU-hours reported if used

This category is scored only for submissions that beat the official hidden-test baseline.

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
