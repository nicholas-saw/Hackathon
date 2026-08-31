# KuaiRand-Pure — Established Evidence / Constraints

> Purpose: concise prior knowledge that the autonomous research agent may rely on.
> Keep this file conservative.
>
> **Do not put speculative hypotheses here.**
> Add new findings only after they are verified in the pre-audit.

## 1. Official / Structural Facts

### C1 — Official target and metric

- Target: `long_view`
- Task: within-user ranking over logged impressions
- Metrics: GAUC and nDCG@5
- Primary: `(GAUC + nDCG@5) / 2`

**Evidence:** Official Starter Kit and competition specification.

**Classification:** HARD FACT

---

### C2 — User-only additive terms cannot change within-user order

Any feature contribution that is constant for all impressions belonging to the same user does not change the ranking of that user's items.

Therefore, a purely additive first-order user bias cannot directly improve within-user ranking.

This does **not** imply that user information is useless in interactions or shared representations.

**Evidence:** Mathematical consequence of within-user ranking.

**Classification:** HARD FACT

---

### C3 — Same-row post-impression feedback is forbidden as input

Current-row values such as click, like, follow, comment, forward, hate, play time, profile stay, comment stay, and profile enter must not be fed into the current-row `long_view` prediction.

They may be used as:

- auxiliary targets
- strictly historical features
- diagnostics

**Evidence:** Competition framing + leakage discipline.

**Classification:** HARD FACT

---

### C4 — Multi-task learning is permitted

The official appendix explicitly identifies KuaiRand's multi-feedback structure as a legitimate setting for auxiliary-task learning, even though only `long_view` is scored.

**Classification:** HARD FACT

---

## 2. Organizer-Provided Negative Evidence

### C5 — Static feature stuffing showed no meaningful gain in the organizer baseline study

The Starter Kit README reports that expanding the FM from the original baseline fields to the tested CWM static feature fields did not materially improve the primary score.

Interpretation:

- repeating the exact same static-feature formulation has low expected value
- this does **not** prove every static or derived feature is useless

**Classification:** STRONG NEGATIVE against the tested formulation

---

### C6 — Increasing FM embedding dimension did not materially help in organizer tests

The Starter Kit README reports little meaningful benefit across tested FM embedding dimensions around k = 8 / 16 / 32.

Interpretation:

- simple capacity scaling of the same baseline FM is low priority
- this does **not** rule out different model families or objectives

**Classification:** STRONG NEGATIVE against simple FM capacity scaling

---

## 3. Findings to Add Later

The pre-audit may propose additions such as:

- field redundancy
- activity/list-length headroom
- temporal structure
- historical signal coverage
- auxiliary-target density
- video-statistic usefulness
- engineering/runtime constraints
- controlled negative model results

Do not add them automatically.

Each proposed addition should include:

1. finding
2. evidence classification
3. measurement or experiment ID
4. numerical evidence
5. what the finding establishes
6. what it does not establish

---

## 4. What Must NOT Appear Here

Do not write:

- "Try recency weighting"
- "Use video statistics"
- "Use multi-task with play_time"
- "Use BPR"
- "Use DIN"
- "Ensemble 9 models"

Those are hypotheses or strategies.

Instead write the underlying verified evidence and allow the autonomous agent to generate the hypothesis itself.
