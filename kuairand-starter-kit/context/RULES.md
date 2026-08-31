# KuaiRand-Pure — Hard Rules for AI Agents

> Purpose: non-negotiable rules that every research, coding, and reflection agent must follow.
> This file should stay **read-only during autonomous runs**.

## 1. Test / Evaluation Isolation

Development means **train + validation only**. This applies to the autonomous run, to
every diagnostic script, and to every pre-audit or ad-hoc analysis.

1. Do not use evaluation/test labels during development.
2. Do not compute test metrics during development or during any audit. A locally
   computed test score is not "extra evidence"; it is an integrity violation, and any
   conclusion resting on one must be deleted rather than annotated.
3. Do not use test performance to choose:
   - hypotheses
   - models
   - features
   - hyperparameters
   - ensemble weights
   - checkpoints
   - constraints
4. Do not summarise, profile, aggregate, or plot **evaluation-period rows** —
   outcomes, features, or identities — to inform a development decision. Permitted
   evaluation-period operations during development are:
   - counting rows by `date` (no other column materialised);
   - reading the official published test numbers as reference material.
5. Applying an already-frozen model to evaluation rows to produce a submission is not
   an inspection. Scoring may read evaluation **features**; it must not read
   evaluation **labels**, and its output must not loop back into development.
6. The final evaluation/test read must occur only after final model selection is
   frozen, and must be logged as an explicit `TEST_OPEN` event.
7. If any process accidentally reads test labels during development, log the event,
   stop the affected experiment, and discard every result derived from it.

### 1a. Random-exposure log — permitted and forbidden portions

`log_random_4_22_to_5_08_pure.csv` spans **both** the validation and the evaluation
windows. The split is not optional:

| Portion | Dates | Development status |
|---|---|---|
| Validation slice | 2022-04-22..28 | Outcomes and features **permitted** for development analysis |
| Evaluation slice | 2022-04-29..05-08 | **Date-only row counts permitted; every other column forbidden** |

Filter the file by `date` **before** materialising any other column. Evaluation-period
outcome information must never influence development, and that includes evaluation
-period identities used as a comparison or reference set.

Using the validation slice for training or for model selection is a research decision,
not a rule violation — but it sits inside the validation window, so any such use must
be logged explicitly with the leakage argument that justifies it.

---

## 1b. External Data — the one hard resource rule

The official problem statement is deliberately permissive about resources and states
exactly one hard rule:

> **No external training data.** Training must rely only on the KuaiRand datasets. No
> augmenting, joining, or pre-training on any other dataset, and no pretrained model
> whose weights were trained on these benchmarks' test labels.

What this does **not** restrict — all explicitly in scope:

- any open-source library or framework (PyTorch, RecBole, TorchRec, LightGBM, …);
- any papers, public solutions, or pretrained weights, provided those weights were not
  trained on these benchmarks' test labels;
- changes to any pipeline stage, not just the model.

The Starter Kit being numpy-only is a property of the reference implementation, not a
restriction on the agent. Do not treat it as one.

If a method needs a dependency the environment lacks, that is an environment question,
not a rules question.

---

## 2. Official Scoring Logic

1. Do not modify or monkeypatch the official `evaluate.py`. It is the scoreboard.
2. Do not reimplement the metric with different semantics, and do not wrap, shadow, or
   import-hack `evaluate` / `auc` / `ndcg_at_k` to change their behaviour.
3. Do not modify official split boundaries or which rows map to which split.
4. Do not change row ordering to improve scores.
5. Preserve the submission `row_id` contract. Scores must be finite; `NaN`/`Inf` are
   rejected by the official checker.
6. Do not special-case a `user_id` or `video_id` because it appears in validation or
   evaluation. Features must treat unseen identifiers through the ordinary UNK path.

### 2a. GAUC semantics in derived analyses

Any per-segment metric, weight share, headroom decomposition, or reweighting scheme
must follow `evaluate.py` exactly:

- **GAUC contributors:** only users with `0 < positives < impressions`. All-negative
  and all-positive users are excluded entirely.
- **GAUC weights:** each contributing user's **positive count**. The denominator is
  the total positives **among contributing users only** — never all positive rows in
  the split. (Validation: 34,592 positives from 12,929 mixed-label users.)
- **nDCG@5:** averaged over **all** users with equal weight, including users who score
  a constant 0 or 1.
- Never multiply an nDCG contribution by a GAUC weight share, and never present one
  blended "metric share" that hides which of the two weightings produced it.
- A single-impression user returns the evaluator's empty-denominator fallback of 0.5
  and contributes **zero** GAUC weight. Do not report that 0.5 as model performance.

---

## 3. Source Integrity

`source/` is **read-only**. That covers both `source/KuaiRand-Pure/` (the raw dataset)
and `source/starter-kit/` (the official competition kit).

Do not:

- edit, overwrite, re-sort, or truncate any file under `source/`
- delete rows, rewrite labels, or impute missing values in the source files
- patch the official kit "temporarily" and revert — a measured run cannot tell that a
  change was temporary

All transforms happen in derived files, caches, or the editable pipeline. If the
pipeline needs behaviour the official kit does not provide, build an adapter in
`pipeline/`; do not fork or patch the kit. If a source file appears to have a real
bug, stop and report it rather than working around it.

Read-only during the final autonomous run: `source/`, `context/`, `harness/`.
Agent-editable: `pipeline/`. Agent-created output: `runlogs/`, `submissions/`,
`reports/`. `context/constraints.md` is human-reviewed evidence — the agent reads it
and never writes it.

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

When constructing historical features, the order is fixed:

```text
1. build the feature from history strictly BEFORE row t
2. score row t
3. only then update the history state with row t
```

Never update first and then score the same row, and never build a "history" aggregate
in one vectorised pass over a period that already contains row t.

Two further requirements:

- **Ties are not predecessors.** Rows sharing a `user_id` and `time_ms` are not
  ordered relative to each other. A row may only consume history from a *strictly*
  earlier timestamp. (5.60% of validation rows sit in non-unique user/timestamp
  groups, so this is not a hypothetical.)
- **Fit statistics on the past only.** Bucket edges, vocabularies, encoders, priors,
  and any aggregate used at scoring time must be fitted on train (plus, for validation
  scoring, strictly earlier validation history under the ordering above) — never on
  the period being scored as a whole.

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

Use exactly this vocabulary — it is the same vocabulary used in
`context/constraints.md` and in the consolidated pre-audit.

### HARD FACT
Deterministic dataset statistic, official rule/code, or mathematical consequence.

### STRONG POSITIVE EVIDENCE
Controlled, repeated validation experiments show a gain that is large relative to seed
noise and to the 0.002 practical epsilon.

### WEAK POSITIVE EVIDENCE
A directionally positive controlled result whose magnitude is at or below the
practical epsilon, or which rests on few seeds.

### STRONG NEGATIVE EVIDENCE
Controlled repeated validation experiments provide convincing evidence against the
tested formulation.

### WEAK NEGATIVE EVIDENCE
The tested formulation underperformed, but the effect is small or lightly replicated
and the broader idea remains plausible.

### INCONCLUSIVE
Evidence is insufficient, noisy, or conflicting. An INCONCLUSIVE finding is an open
question, not a soft prohibition.

### ENGINEERING CONSTRAINT
A reproducible property of the environment, runtime, or repository that bounds what an
iteration can do.

### INVALID / FORBIDDEN
Ruled out by official rules or by leakage semantics, regardless of measured effect.

Rules of use:

- Never convert a WEAK NEGATIVE into a blanket prohibition, and never convert an
  INCONCLUSIVE into either a prohibition or an endorsement.
- Always state the **scope** a classification applies to (which model, which fields,
  which split, which seeds). "X does not help" is almost never what was measured;
  "X in this exact FM formulation, over N seeds, moved validation primary by d" is.
- A single run never justifies STRONG anything.
- Deltas below the 0.002 practical epsilon are reported with their magnitude and seed
  count, not rounded up into a conclusion.

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

---

## 13. Using `constraints.md` and `data_profile.md`

These files state **what has been measured**, not what to do next.

1. A HARD FACT bounds what is true; it does not name a method.
2. A negative result applies to the **exact formulation tested**. A different loss,
   encoding, or model family is a new question.
3. An INCONCLUSIVE entry is an invitation to measure, not a warning to avoid.
4. If the evidence base does not answer a question the agent needs answered, the
   correct move is `REQUEST_ANALYSIS` on train + validation — not an assumption, and
   not a peek at evaluation data.
5. Do not treat the ordering of entries in any context file as a priority ranking.
