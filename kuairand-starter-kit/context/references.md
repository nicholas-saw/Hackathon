# Method index

Short cards. What a method is, what it assumes, what it costs. Deliberately unranked and
without recommendations: choosing among these is the research decision you are here to
make, and a pre-ranked list would make you an executor.

**Pairwise ranking (BPR)** — optimise P(positive ranked above negative) within a user
instead of a pointwise probability. Assumes usable positive/negative pairs per user;
degenerate users contribute nothing. Cost: a sampler plus a change to the gradient; same
order of wall-clock as the baseline. Rendle et al., UAI 2009.

**Listwise softmax / ListNet** — a softmax over each user's impression list, cross-entropy
against the normalised label vector. Assumes lists are meaningful units. Note the
validation median list length is 4. Cao et al., ICML 2007.

**LambdaRank / LambdaMART** — weight pairwise gradients by the nDCG change from swapping
the pair. Directly targets a truncated metric. Needs grouped data and a working
delta-nDCG. Burges, 2010.

**Multi-task / ESMM-style** — auxiliary heads on other feedback signals sharing a
representation with the scored task. Assumes the auxiliary signal correlates with the
target and that shared capacity is not the binding constraint. Ma et al., SIGIR 2018.

**Censored watch-time regression (CWM)** — a completed play truncates the true watch time,
so a one-sided loss rather than squared error. Requires play_time_ms and duration_ms.
Zhao et al., KDD 2024.

**Target attention (DIN)** — score a candidate by attending over the user's history.
Assumes the candidate or its attributes recur in that history.

**Field-aware and deep factorisation (FFM, DeepFM, DCN)** — richer interaction structure
over the same sparse fields.

**Inverse propensity weighting** — reweight by exposure probability to debias a logged
policy. Requires propensities; note `is_rand` is 0 on every standard-log row, and the
random-exposure log has no rows before 20220422.

**Seed ensembling / rank averaging** — combine several models' within-user ranks. Reduces
variance rather than bias.
