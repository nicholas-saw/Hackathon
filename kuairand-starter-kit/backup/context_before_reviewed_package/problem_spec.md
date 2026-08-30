# Problem specification — KuaiRand-Pure

Source-derived. Official Track 2 problem statement (updated 27 Aug 2026) and the shipped
Starter Kit. Where the two disagree, the Starter Kit wins: `kit/evaluate.py` is the
scoreboard.

## The task

Within-user ranking over logged impressions. For each row of the evaluation split the
model emits one score; the evaluator ranks impressions **inside each user**. This is not
full-catalogue retrieval, and scores are never compared across users.

Relevance label: **`long_view`** (a native 0/1 column).

> The official document contains one stale line naming `is_click` with NDCG@10 and
> Recall@50. It is superseded. The benchmark table, the judging section, the appendix and
> the shipped `evaluate.py` all pin the task to `long_view` with GAUC / nDCG@5. Appendix
> A.4 explains why Recall was dropped: at roughly five impressions per user, Recall@50 is
> 0.999+ for every model including random scoring.

## Metrics

- **GAUC** — per-user AUC, weighted by that user's positive count. Only users with
  `0 < positives < impressions` contribute.
- **nDCG@5** — per user. Users with no positive score 0 and are included in the mean.
  Gain is `2^rel - 1`, which is the identity under binary labels.
- **primary = mean(GAUC, nDCG@5)** — this is what ranks submissions.

Scoring formula: `delta(m) = agent(m) - baseline(m)` for each metric, averaged. That is
exactly the primary delta against the official baseline on the hidden test set.

## Splits

Fixed, by date, and never to be changed:

    train       20220408 - 20220421     1,141,112 rows
    validation  20220422 - 20220428       124,909 rows
    test        20220429 - 20220508       170,588 rows

Development uses train and validation only. The hidden test set is scored once, on the
final submission.

## Official baseline

A numpy Factorization Machine, k=16, lr=0.001, five categorical fields. Hidden test:
GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946, mean over 5 seeds, std 0.0008. This is the
number to beat — not a baseline you build yourself.

## Reading the metric range

The metrics do not span [0, 1]. On validation, 30.3% of users are all-negative (nDCG is 0
for any model) and 11.9% are all-positive, so perfect ranking reaches only primary 0.8484
(oracle nDCG@5 0.6968). Random scoring sits at 0.4834. Judge progress against the oracle,
not against 1.0.

## Convergence and limits

    epsilon = 0.002 over N = 3 consecutive non-improving iterations
    50 iterations per run (hard cap)
    6 hours agent wall-clock (backstop)

A run is converged when validation primary has not improved by more than epsilon over the
last three iterations, or when it hits either cap — whichever comes first. **The scored
submission is the validation-best checkpoint at that point**, so continuing past
convergence cannot raise the score.

## Submission format

CSV with header `row_id,user_id,video_id,score`. `row_id` is a 0-based strictly increasing
index into the split as produced by `data.load()`. `user_id` / `video_id` are redundant
alignment checks. `score` is any real number — only relative order matters; NaN and Inf
are rejected. `row_id` is required because `(user_id, video_id)` is **not unique**: 3.06%
of evaluation rows repeat a pair, up to 12 times.

## What is scored

| Criterion | Weight | What it rewards |
|---|---|---|
| Technical Execution | 35% | primary delta on hidden test, plus robustness under failure |
| Innovation & Problem Insight | 20% | what the agent identified as worth trying, and why |
| Impact & Relevance | 20% | autonomy, measured in manual interventions |
| Feasibility & Practicality | 15% | LLM tokens and agent wall-clock — scored only if the submission beats the baseline |
| Presentation | 10% | final event only |

Robustness is judged on how a failure is handled, not on whether one occurs.

## The division of labour

Humans may provide: the task definition, rules, data, codebase, verified prior evidence,
research references and budget.

The agent must provide: interpretation, research questions, hypotheses, prioritisation,
experiment design, code changes, evaluation, reflection, and the next decision.
