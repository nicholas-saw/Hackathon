# KuaiRand-Pure — Hard Rules for AI Agents

> Purpose: non-negotiable rules that every research, coding, and reflection agent must follow.
> This file should stay **read-only during autonomous runs**.

## 1. Test / Evaluation Data

1. Do not use evaluation/test labels during development.
2. Do not compute test metrics during the pre-audit.
3. Do not use test performance to choose:
   - hypotheses
   - models
   - features
   - hyperparameters
   - ensemble weights
   - checkpoints
   - constraints
4. The final evaluation/test read must occur only after final model selection is frozen.
5. If any process accidentally reads test labels during development, log the event and stop the affected experiment.

---

## 2. Official Scoring Logic

1. Do not modify or monkeypatch the official `evaluate.py`.
2. Do not reimplement the metric with different semantics.
3. Do not modify official split boundaries.
4. Do not change row ordering to improve scores.
5. Preserve the submission `row_id` contract.

---

## 3. Raw Data Integrity

The original KuaiRand-Pure files are source data.

Do not:

- edit them
- overwrite them
- re-sort them in place
- delete rows
- rewrite labels
- replace missing values in the source files

All transforms must happen in derived files, caches, or the editable pipeline.

---

## 4. Post-Impression Feedback Leakage

The following are post-impression feedback/outcome signals:

- `long_view`
- `is_click`
- `is_like`
- `is_follow`
- `is_comment`
- `is_forward`
- `is_hate`
- `play_time_ms`
- `profile_stay_time`
- `comment_stay_time`
- `is_profile_enter`

### Forbidden

Do not use the **current row's** post-impression feedback value as an input feature to predict the current row's `long_view`.

Examples of forbidden inputs:

```text
is_click_t      -> predict long_view_t
is_like_t       -> predict long_view_t
play_time_ms_t  -> predict long_view_t
```

Do not hide the same leakage inside a derived "engagement score".

### Allowed

Post-impression signals may be used as:

1. **Auxiliary targets** in multi-task learning.
2. **Historical features** constructed strictly from earlier interactions.
3. **Diagnostic variables** for analysis.

When constructing historical features:

```text
build feature from history BEFORE row t
then update history using row t
```

Never update first and then score the same row.

---

## 5. Multi-Task Learning

Multi-task learning is permitted.

Auxiliary targets may include feedback such as:

- click
- like
- follow
- comment
- forward
- watch time

Only `long_view` is scored.

Do not assume that every auxiliary task helps. Negative transfer must be measured empirically.

---

## 6. Validation Discipline

Validation may be used for:

- early stopping
- model comparison
- hyperparameter tuning
- convergence decisions
- choosing the final checkpoint

Every validation use should be logged.

Do not pretend validation was "touched once" if it was used repeatedly.

---

## 7. Experimental Discipline

Every experiment must record:

- hypothesis
- evidence that motivated it
- mechanism
- exact code diff or commit/fingerprint
- configuration
- runtime
- token cost if an LLM was used
- validation GAUC
- validation nDCG@5
- validation primary
- error/recovery events
- decision: KEEP / REVERT / INCONCLUSIVE / REQUEST_ANALYSIS

Do not declare a method dead from one noisy run.

---

## 8. Evidence Language

Use these evidence classes:

### HARD FACT
Deterministic dataset statistic, official rule/code, or mathematical consequence.

### STRONG NEGATIVE
Controlled repeated validation experiments provide convincing evidence against the tested formulation.

### WEAK NEGATIVE
The tested formulation performed poorly, but the broader idea remains plausible.

### INCONCLUSIVE
Evidence is insufficient, noisy, or conflicting.

Never convert a weak negative into a blanket prohibition.

---

## 9. Human Intervention

Minimize manual interventions during the final autonomous run.

A human may perform setup before `RUN_START`.

After `RUN_START`, log every intervention with a category such as:

- observation only
- environment repair
- code edit
- research steering
- manual model selection
- restart

Do not report "0 interventions" unless the logs support it.

---

## 10. Error Handling

The agent should recover where possible from:

- syntax/import errors
- failed dependencies
- timeout
- NaN / Inf
- memory failure
- invalid feature construction
- invalid submission
- broken subprocess

Every recovery attempt must be logged.

Do not silently patch official scoring or raw data to recover.

---

## 11. Resource Limits

Respect:

```text
50 iterations maximum
6 hours agent wall-clock maximum
```

Track:

- iterations
- wall-clock
- LLM tokens
- GPU-hours if used

---

## 12. Research-Agent Boundary

The agent may be given:

- facts
- constraints
- previous results
- dataset profile
- references

The agent should still independently decide:

- what bottleneck it believes exists
- what hypothesis to test
- why the hypothesis is reasonable
- which experiment has highest expected value
- how the result changes its beliefs

Do not reduce the agent to selecting from a pre-ranked human-written answer key.
