# KuaiRand-Pure — Research Reference Index

> Purpose: give the proposer/research agent a compact technical toolbox.
> This file describes methods; it does **not** rank them or recommend a winner.

## 1. Factorization Machines (FM)

Core idea:

- represent sparse categorical fields with latent vectors
- model pairwise feature interactions efficiently

Relevance:

- this is the official baseline family

Use as reference for:

- baseline mechanism
- feature-interaction behavior
- comparison against alternative objectives/models

---

## 2. BPR / Pairwise Ranking

**Bayesian Personalized Ranking (BPR)**

Core idea:

- train on relative preference pairs
- encourage a positive item to score above a negative item

Potential relevance:

- GAUC is a ranking metric
- pairwise objectives optimize ordering rather than calibrated probability

Do not assume it helps; test empirically.

---

## 3. Listwise Ranking

Core idea:

- optimize a group/list jointly rather than independent rows
- examples include listwise softmax objectives and LambdaRank-style methods

Potential relevance:

- nDCG@5 is explicitly list-based
- groups are users and their logged impressions

Important implementation concern:

- training batches must preserve user grouping

---

## 4. LambdaRank / LambdaMART

Core idea:

- use ranking-aware gradient weighting, commonly targeting nDCG
- often implemented with GBDT libraries such as LightGBM

Potential relevance:

- direct ranking objective
- strong CPU implementations are available

Do not assume compatibility or gain without testing.

---

## 5. DeepFM / DCN / xDeepFM

### DeepFM

Combines FM-style low-order interactions with a neural component for nonlinear interactions.

### DCN

Uses explicit feature-crossing layers.

### xDeepFM

Models higher-order explicit and implicit interactions.

Potential relevance:

- alternative representation/interaction capacity

Known caution:

- organizer evidence indicates simple FM capacity scaling alone is not a major bottleneck.

---

## 6. DIN / Behavior-History Attention

**Deep Interest Network (DIN)**

Core idea:

- model user history
- attend to historical behaviors conditioned on the candidate item

Potential relevance:

- recommendation settings with meaningful repeated categories/items/authors

Feasibility depends on actual history coverage and overlap, which must be measured first.

---

## 7. Sequential Recommendation

Examples:

- SASRec
- BERT4Rec-style sequence modeling

Core idea:

- model ordered user behavior history

Potential relevance:

- temporal user state

Feasibility depends on:

- history length
- candidate/history overlap
- runtime
- whether sequential structure adds information in this dataset

---

## 8. Multi-Task Learning

The official appendix explicitly permits multi-feedback learning.

Possible auxiliary targets include:

- click
- like
- follow
- comment
- forward
- watch-time-related targets

### Shared-bottom

Common representation shared by all tasks with task-specific heads.

### MMoE

Mixture-of-Experts architecture intended to share useful information while reducing task interference.

### PLE

Progressive Layered Extraction, another multi-task structure designed to separate shared and task-specific information.

Key issue:

**negative transfer / seesaw effect**

Do not assume all auxiliary tasks help `long_view`.

---

## 9. ESMM-Style Thinking

ESMM is a multi-task framework originally designed around linked feedback funnels.

The KuaiRand task is not classic CVR, but the general principle of sharing information across feedback tasks may still be relevant.

The official appendix specifically notes multi-feedback modeling as legitimate.

---

## 10. Watch-Time Modeling / CWM

Reference from the official material:

**Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time (CWM)**

Core idea:

- treat watch time carefully when completed plays imply censoring/truncation by video duration
- use specialized regression objectives rather than naive squared error

Important caution:

- CWM research code is not the official baseline
- its exact labels/metrics differ from this competition

Use it as a research reference, not as competition truth.

---

## 11. Recency / Temporal Weighting

Core idea:

- assign different training importance to interactions based on time
- useful when train and validation distributions shift

No recommendation should be made until temporal drift is measured.

---

## 12. Historical Aggregates

Examples:

- prior user click rate
- prior user like rate
- prior video engagement rate
- prior user-author affinity
- repeat-exposure count

Critical rule:

Historical features must use only information available **before** the row being scored.

---

## 13. Counterfactual / Off-Policy Evaluation

KuaiRand includes randomized-exposure data.

Relevant concepts:

- propensity
- inverse propensity scoring
- off-policy evaluation
- exposure bias

The random log's date coverage and safe use must be audited before it influences model selection.

---

## 14. Ensemble / Rank Aggregation

Core idea:

- combine multiple models to reduce variance or exploit complementary errors

For within-user ranking, common approaches include:

- score averaging
- rank averaging

Do not fit or select ensemble rules using evaluation/test labels.

---

## 15. Recommended Source Material

Official challenge references:

1. MLE-Bench — autonomous ML engineering benchmark
2. AIDE — AI-driven code exploration / ML research agent
3. AI Scientist-v2 — agentic scientific exploration
4. CWM — duration-bias / watch-time modeling

Starter-level recommender references from the official appendix:

- Google Machine Learning Crash Course — Recommendation Systems overview
- Wang Shusen — Recommender Systems overview

Additional papers may be added here during research.

When adding a reference, summarize:

- what problem it solves
- core mechanism
- assumptions
- expected implementation cost

Do not write "the agent should use this next."
