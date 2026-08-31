# KuaiRand-Pure — Research Reference Index

> Purpose: give the proposer/research agent a compact technical toolbox.
> This file describes methods; it does **not** rank them or recommend a winner.
>
> Entries are grouped by the problem they address, and the ordering within a group is
> arbitrary. Where the **organizer** states a preference, it is quoted as the
> organizer's judgement and labelled as such — that is a description of an official
> source, not this project's research strategy.
>
> A method staying in this file is not an endorsement, and a method whose one tested
> implementation underperformed is not removed: a negative result belongs to the exact
> formulation tested (see `constraints.md`), not to the method family.

## 0. Task Shape — Read This Before Reaching for a Method

The scored task is **re-ranking a fixed, logged candidate list within each user**. It is
not full-catalogue retrieval. Consequences for method selection:

- There is no candidate-generation stage to optimise, and no negative sampling problem
  in the usual sense — the impressions are given.
- Retrieval-oriented architectures (two-tower + ANN index, in-batch softmax over the
  full catalogue) do not map onto this task without being repurposed. Their loss
  formulations may still transfer; their serving structure does not.
- Anything constant within a user's list cannot change that user's score (`constraints.md`
  C2). This is a hard filter on what a method can possibly do here.
- Validation lists are short (median 4 impressions), which bounds how much a list-level
  method has to work with per group.

---

## 1. Factorization Machines (FM)

**Problem addressed:** sparse categorical interactions with limited data.

**Mechanism:** each categorical value gets a latent vector; the model scores first-order
terms plus all pairwise dot products between field embeddings, computed in linear time
via the sum-of-squares identity.

**Assumptions:** interactions are well approximated by inner products of low-rank
embeddings; features are categorical or bucketed.

**Implementation notes:** this is the official baseline family
(`source/starter-kit/baseline.py`) — numpy only, roughly a minute per CPU run. Its exact
mechanism and configuration are recorded in `constraints.md` C8.

**Variants worth knowing:** field-aware FM (FFM) gives each field pair its own embedding,
multiplying parameters by the field count; higher-order FM extends beyond pairwise.

---

## 2. BPR / Pairwise Ranking

**Problem addressed:** the mismatch between a pointwise probability objective and a
rank-based metric.

**Mechanism:** train on (positive, negative) pairs drawn from the same user, maximising
the probability that the positive scores higher. Gradients depend only on score
differences within a user.

**Assumptions:** meaningful pairs exist within a group — a user with uniform labels
generates none.

**Implementation notes:** pair construction must respect user grouping. Note that 42.222%
of validation users have uniform labels (`constraints.md` C9) and would supply no pairs;
what that implies for training composition is untested. Pair count grows with the product
of positives and negatives per user, so sampling policy matters for runtime.

---

## 3. Listwise Ranking

**Problem addressed:** optimising a whole ranked list rather than independent rows.

**Mechanism:** a softmax (or other list-level loss) over the scores of one user's
impressions; the loss sees the entire group at once.

**Assumptions:** the group used in training matches the group used in scoring — here,
one user's logged impressions.

**Implementation notes:** batching must preserve user grouping, which usually means
restructuring the batch builder rather than only the loss. Short lists (median 4) mean
many groups contribute few terms.

---

## 4. LambdaRank / LambdaMART

**Problem addressed:** directly targeting a rank metric such as nDCG.

**Mechanism:** reweight pairwise gradients by the metric change that swapping the pair
would cause, so the optimiser spends effort where the metric is sensitive.

**Assumptions:** the target metric decomposes into pairwise swaps — nDCG does.

**Implementation notes:** commonly implemented over GBDTs (LightGBM `lambdarank`), which
needs group boundaries supplied explicitly and dense/numeric features rather than raw
high-cardinality IDs. LightGBM is explicitly in scope (see the resource policy in §17);
the Starter Kit's numpy-only footprint is not a restriction, though adding any dependency
still costs setup time against the wall-clock budget.

---

## 5. Deep Interaction Models — DeepFM / DCN / xDeepFM

**Problem addressed:** interaction orders and nonlinearities an FM cannot express.

**Mechanism:** DeepFM pairs an FM component with an MLP over the same embeddings; DCN
stacks explicit feature-crossing layers; xDeepFM adds a compressed interaction network
for explicit higher-order interactions.

**Assumptions:** enough data to fit the extra parameters without overfitting.

**Implementation notes:** all three need a tensor framework. Relevant context: simple
width scaling of the official FM produced no gain (`constraints.md` C6) — that result is
about *width in that FM*, and says nothing directly about interaction *structure*, which
is what these models change.

---

## 6. DIN / SIM — Behaviour-History Attention

**Problem addressed:** representing a user's history conditioned on the candidate item.

**Mechanism:** attend over the user's historical behaviour sequence with the candidate as
the query, producing a candidate-specific user representation.

**Assumptions:** history is available, ordered, and shares a vocabulary with the
candidate.

**Implementation notes:** feasibility depends on measured coverage, which differs by
granularity by more than an order of magnitude (`constraints.md` C13): exact same-video
history covers 1.624% of validation rows, same-author 3.381%, parsed tag tokens 78.413%.
The official documentation also notes KuaiRand-**Pure** has incomplete sequential logs
and points to the 27K/1K variants when rigorous sequences are needed. Strict
history-before-row ordering is mandatory (`RULES.md` §4).

---

## 7. Sequential Recommendation

**Problem addressed:** temporal user state and order effects.

**Mechanism:** SASRec applies causal self-attention over the interaction sequence;
BERT4Rec uses masked-item prediction over a bidirectional encoder (bidirectional training
requires care to stay causal at scoring time).

**Assumptions:** sequence order is meaningful and reliably recorded.

**Implementation notes:** tied timestamps are not ordered — 5.60% of validation rows sit
in non-unique user/timestamp groups (`constraints.md` C13), so a "previous item" is not
always well defined. Runtime and sequence-length truncation dominate cost.

---

## 8. Multi-Task Learning

**Problem addressed:** exploiting several feedback signals when only one is scored.

**Mechanism and structures:**
- **Shared-bottom** — one shared representation, per-task heads. Cheapest; most exposed
  to interference.
- **MMoE** — a set of expert subnetworks with per-task gating over them.
- **PLE** — separates shared experts from task-specific experts explicitly, layer by layer.

**Assumptions:** tasks share exploitable structure, and auxiliary labels are dense enough
to train their heads.

**Implementation notes:** permitted by the official rules (`constraints.md` C4). Auxiliary
density spans two orders of magnitude here — click ~46%, like ~1.8%, follow ~0.1%
(`research/data_profile.md` §7) — so head-level loss weighting and the sparse-task
contribution are design decisions with real consequences. **Negative transfer / seesaw is
the standard failure mode** and must be measured, not assumed away. The feedback columns
are auxiliary *targets* only; same-row use as inputs is forbidden (`constraints.md` C3).

---

## 9. ESMM-Style Funnel Modelling

**Problem addressed:** sample-selection bias when one label is only observed downstream
of another.

**Mechanism:** model the full-space product of a funnel (e.g. impression → click →
conversion) so that the downstream task is trained over the whole impression space.

**Assumptions:** a genuine funnel/ordering exists between the labels.

**Implementation notes:** this task is not classic CVR. The official appendix (A.2)
settles the funnel question directly: in KuaiRand the scored label `long_view` is logged
on **every impression**, not only on clicked ones, so classic sample-selection bias does
**not** apply here and the ESMM correction it was designed for has no target. Data
sparsity still applies, and the official text names the multi-feedback structure as a
legitimate reason to use other signals as auxiliary tasks. So the transferable part is
the sharing across feedback tasks, not the funnel correction.

---

## 10. Watch-Time Modelling / CWM

**Problem addressed:** duration bias — watch time is mechanically truncated by video
length, so naive regression conflates interest with duration.

**Mechanism:** treat completed plays as censored observations and use one-sided /
censored-regression losses rather than squared error (the CWM paper's contribution:
*Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time*,
https://github.com/hyz20/CWM).

**Assumptions:** the censoring mechanism is known and duration is observed.

**Implementation notes:** CWM is research code pinned to `torch==1.6.0`, and it optimises
counterfactual watch time against its own reconstructed `long_view2` label — its metric
is **not** this competition's metric. Use as a conceptual reference, not a drop-in.
`play_time_ms` may only be an auxiliary target or strictly historical feature
(`constraints.md` C3).

---

## 11. Recency and Temporal Weighting

**Problem addressed:** distribution shift between training data and the scored period.

**Mechanism:** weight training rows by age, decay old data, restrict the window, or add
explicit time features.

**Assumptions:** the shift is monotone in time and the recent past resembles the scored
period more than the distant past.

**Implementation notes:** that assumption is **not** settled here. Measured drift is
multidimensional and points in different directions: validation is closer to *early*
train in target rate and duration, and closer to *late* train in tab distribution and
volume (`constraints.md` C14). Train volume also falls 13.9x across the window, so a
recency window and a sample-size reduction are confounded.

---

## 12. Historical Aggregates and Target Encoding

**Problem addressed:** turning a user's or item's past behaviour into dense features.

**Mechanism:** counts, rates, and ratios computed over prior interactions — prior user
click rate, prior item engagement rate, prior user–author affinity, repeat-exposure
counts. Target encoding replaces a category with a smoothed statistic of the label.

**Assumptions:** the statistic is computed only from information available before the
scored row.

**Implementation notes:** the ordering rule is non-negotiable — build from strictly
earlier history, score, then update (`RULES.md` §4), with tied timestamps excluded from
"earlier". Low-count categories need smoothing toward a prior; the official item-popularity
baseline uses a prior weight of 20 as one concrete example. Out-of-fold or
time-sliced construction is the standard defence against target leakage.

---

## 13. Counterfactual / Off-Policy Evaluation

**Problem addressed:** logged feedback reflects what the deployed policy chose to show.

**Mechanism:** propensity estimation, inverse propensity scoring, doubly-robust
estimators, off-policy evaluation against a randomised-exposure log.

**Assumptions:** propensities are estimable and bounded away from zero; the randomised
log is genuinely randomised.

**Implementation notes:** the random log's exposure mechanism is a uniform replacement
from the 7,583-item pool with a fixed **undisclosed** probability, which limits exact
propensity reconstruction. Only the 2022-04-22..28 slice is usable in development
(`constraints.md` C18, `RULES.md` §1a); it is distributionally distinct from standard
traffic (0.08056 vs 0.31328 positive rate) and almost pair-disjoint from it.

---

## 14. Cold-Start and Unseen-Identifier Handling

**Problem addressed:** identifiers at scoring time that were unseen or barely seen at
training time.

**Mechanism:** UNK slots (the official encoder's approach), hashing tricks, count-based
backoff to coarser granularity, or embedding regularisation toward a prior.

**Assumptions:** a coarser level exists that generalises when the fine level does not.

**Implementation notes:** relevant given the structure in `constraints.md` C12 — entities
are almost all warm (98–99.9%), but the exact user–video relationship is novel for
98.373% of validation pairs. Backoff must be a genuine feature-space fallback, never a
lookup keyed on membership in the validation or evaluation split (`RULES.md` §2).

---

## 15. Ensembling and Rank Aggregation

**Problem addressed:** variance reduction and combining complementary models.

**Mechanism:** average scores, average ranks, or fit a combiner. For within-user ranking,
rank averaging avoids the scale-calibration problem that score averaging inherits.

**Assumptions:** component models make partly independent errors.

**Implementation notes:** combination weights are model selection and must be fitted on
validation, never on evaluation labels (`RULES.md` §1). Each component multiplies
iteration cost against the 6-hour / 50-iteration budget.

---

## 16. Experimental Methodology Under Seed Noise

**Problem addressed:** distinguishing a real effect from run-to-run variation.

**Mechanism:** matched seeds across configurations, paired deltas rather than raw means,
population std reported alongside, and a decision threshold fixed in advance.

**Assumptions:** seeds are the dominant noise source and configurations are otherwise
identical.

**Implementation notes:** the reference scales for this benchmark are a published seed
std of **0.0008** and a convergence epsilon of **0.002** (`PROBLEM.md` §7,
`constraints.md` C8). A single run cannot establish anything; the audit's own controlled
results used 3–5 matched seeds and still landed several deltas below epsilon. Budget the
seed cost into the iteration plan.

---

## 17. Official Source Material

**Resource policy.** Any open-source library or framework is in scope — the official
text names PyTorch, RecBole, TorchRec and LightGBM — as are any papers, public
solutions, and pretrained weights. The single hard rule is no external training data
(`RULES.md` §1b). Methods below that need a tensor framework or a GBDT library are
therefore admissible; the Starter Kit's numpy-only footprint is a property of the
reference implementation, not a constraint on the agent.

**Competition and starter-kit sources (authoritative):**

- `source/starter-kit/README.md` — task definition, published baseline ladder, and the
  organizer's own two sections: *already measured, no gain* (static feature stuffing;
  embedding width; zero contribution of pure user-side first-order terms) and *where the
  headroom probably is*. The latter is presented in **the organizer's judged order of
  promise**: (1) loss/objective mismatch, (2) user history sequences, (3) multi-task,
  (4) watch-time modelling, (5) alternative model family, (6) time features and drift,
  (7) unbiased validation via the random log. That ordering is the organizer's, and it is
  quoted here because it is official source material — it is not a plan handed to the
  agent, and the agent is expected to form and justify its own priorities.
- `source/starter-kit/evaluate.py` — the metric definition; the scoring authority.
- `source/starter-kit/baseline_scores.json` — published scores, seed variance, and the
  convergence parameters.
- `source/starter-kit/ablation_features.py` — reproduction of the static-feature result
  (note the 8/13 field correction in `DATA_GUIDE.md` §8).
- Official KuaiRand documentation (https://github.com/chongminggao/KuaiRand) — dataset
  description, the video-statistic aggregation description, the random-intervention
  mechanism, and the note that KuaiRand-Pure has incomplete sequential logs.

**Autonomous-agent references from the official challenge material:**

1. MLE-Bench — benchmark for autonomous ML engineering.
2. AIDE — LLM-driven code exploration for ML tasks.
3. AI Scientist-v2 — agentic scientific exploration loop.
4. CWM — duration-bias / watch-time modelling.

**Starter-level recommender references from the official appendix:**

- Google Machine Learning Crash Course — Recommendation Systems overview.
- Wang Shusen — Recommender Systems lecture series.

---

## 18. Adding to This File

When adding a reference, record:

- the problem it addresses
- its mechanism
- its assumptions
- implementation and runtime considerations

Do not write "the agent should use this next", "recommended", or "most promising" in this
project's own voice. Do not delete a method because one implementation of it
underperformed — record that result in `constraints.md` with its exact scope and leave
the method described here.
