# SETUP & FOLDER STRUCTURE — Instructions for the AI

> Paste or give this file to the AI that is setting up the fresh project.
>
> Goal: create a clean separation between immutable competition sources, human-provided context, pre-audit research, the editable ML pipeline, the harness, and the final autonomous agent.

# 1. Create This Project Structure

```text
techjam-track2/
│
├── source/
│   ├── KuaiRand-Pure/              # raw dataset — READ ONLY
│   └── starter-kit/                # official competition kit — READ ONLY
│
├── context/
│   ├── PROBLEM.md                  # official task definition — READ ONLY during run
│   ├── RULES.md                    # non-negotiable rules — READ ONLY during run
│   ├── DATA_GUIDE.md               # data map — READ ONLY during run
│   ├── constraints.md              # verified prior evidence — READ ONLY during run
│   └── references.md               # technical toolbox — READ ONLY during run
│
├── research/
│   ├── PRE_AUDIT.md                # pre-run empirical notebook
│   ├── data_profile.md             # compact measurements
│   ├── scripts/                    # pre-audit analysis scripts
│   ├── plots/
│   └── experiment_results/
│
├── pipeline/                       # EDITABLE by the autonomous coding agent
│   ├── data_adapter.py
│   ├── features.py
│   ├── train.py
│   ├── models/
│   └── objectives/
│
├── harness/                        # infrastructure — READ ONLY during final autonomous run
│   ├── score.py
│   ├── executor.py
│   ├── guards.py
│   ├── cache.py
│   ├── diagnostics.py
│   ├── logger.py
│   └── submission.py
│
├── agent/
│   ├── controller.py
│   ├── proposer.py
│   ├── coder.py
│   ├── reflector.py
│   ├── governor.py
│   └── prompts/
│
├── runlogs/
├── submissions/
├── reports/
├── tests/
└── README.md
```

# 2. Source Folders

## `source/KuaiRand-Pure/`

Place the extracted raw dataset here.

Rules:

- never edit raw CSVs
- never overwrite source files
- never change labels
- never re-sort files in place
- all derived data goes elsewhere

## `source/starter-kit/`

Place the untouched official Starter Kit here.

Treat these files as competition truth:

- `README.md`
- `data.py`
- `evaluate.py`
- `baseline.py`
- `baseline_scores.json`
- `submit.py`
- `ablation_features.py`

Do not modify them.

If the working pipeline needs more flexibility, build adapters in `pipeline/`, not patches inside the official kit.

---

# 3. Context Files

Copy the supplied context files into `context/`.

The final autonomous agent may READ these files but should not edit them during the run.

### `PROBLEM.md`

Official task, metrics, baseline, convergence, judging, limits.

### `RULES.md`

Hard safety/integrity rules.

### `DATA_GUIDE.md`

Available data and column semantics.

### `constraints.md`

Only verified prior evidence.

Do not fill it with human-written recommendations.

### `references.md`

Technical method summaries.

Methods are described but not ranked.

---

# 4. Pre-Audit Phase

Before building the final agent, use `research/` for empirical auditing.

The research AI may edit:

```text
research/PRE_AUDIT.md
research/data_profile.md
research/scripts/*
research/plots/*
research/experiment_results/*
```

It must NOT directly edit:

```text
context/constraints.md
```

Workflow:

```text
raw data
   ↓
PRE_AUDIT.md
   ↓
human review
   ↓
verified findings copied into constraints.md
   ↓
final autonomous run
```

The pre-audit should produce evidence, not a manually optimized final model.

---

# 5. Final Autonomous-Run Boundary

At `RUN_START`:

### Read-only

```text
source/
context/
harness/
```

### Agent-editable

```text
pipeline/
```

### Agent-created output

```text
runlogs/
submissions/
reports/
```

The agent code in `agent/` should normally be fixed before the final measured run.

Do not let the final agent alter its own evaluator, guardrails, source data, or prior-evidence files.

---

# 6. Agent Roles

## Proposer / Researcher

Reads:

- `context/PROBLEM.md`
- `context/RULES.md`
- `context/DATA_GUIDE.md`
- `context/constraints.md`
- `context/references.md`
- `research/data_profile.md`
- current pipeline code
- previous `journal.jsonl`

Outputs:

- 2–4 candidate hypotheses
- evidence for each
- mechanism
- expected metric effect
- cost/risk
- chosen next experiment

The proposer should generate hypotheses from evidence rather than selecting from a pre-ranked human answer list.

---

## Coder

Reads only what it needs:

- chosen hypothesis
- relevant rules
- relevant pipeline files
- required interfaces/tests

Outputs:

- code changes
- implementation note

The coder does not need the entire research context every iteration.

---

## Executor / Harness

Runs the experiment.

Must capture:

- exit status
- stdout/stderr
- runtime
- resource usage
- validation metrics
- errors

Must enforce:

- timeout
- process cleanup
- no test evaluation
- score validity
- checkpointing

---

## Reflector

Reads:

- original hypothesis
- before metrics
- after metrics
- relevant diagnostics
- error/recovery logs
- related previous experiments

Outputs one of:

```text
KEEP
REVERT
INCONCLUSIVE
REQUEST_ANALYSIS
```

and explains what the result implies about the mechanism.

---

# 7. Hypothesis Schema

Each proposer hypothesis should contain at least:

```json
{
  "hypothesis": "...",
  "evidence": ["...", "..."],
  "mechanism": "...",
  "target_metric": "...",
  "target_segment": "...",
  "proposed_change": "...",
  "expected_result": "...",
  "risk": "...",
  "cost": "...",
  "invalid_if": "...",
  "files_to_modify": ["..."]
}
```

Do not accept a hypothesis such as:

```text
"try multitask"
```

without evidence and mechanism.

---

# 8. REQUEST_ANALYSIS

The proposer may decide it lacks evidence.

It may output:

```json
{
  "action": "REQUEST_ANALYSIS",
  "question": "...",
  "required_measurement": "...",
  "why_needed": "..."
}
```

The harness/diagnostic layer may perform a safe train/validation-only analysis and return the result.

This supports the competition's required loop:

```text
inspect data
→ reason
→ form hypothesis
→ experiment
→ evaluate
→ reflect
→ revise
```

---

# 9. Logging

Every autonomous iteration should append to:

```text
runlogs/<run_id>/journal.jsonl
```

Record:

- iteration ID
- hypothesis
- rationale
- chosen action
- code diff/fingerprint
- configuration
- metrics
- runtime
- token usage
- errors
- recovery
- reflector verdict
- manual intervention count

Optional but useful:

- hash-chain journal entries
- file fingerprints
- explicit `FINAL_SELECTION` event
- explicit `TEST_OPEN` event

---

# 10. Test Isolation

Development must use train + validation only.

The final workflow should be:

```text
RUN_START
  ↓
autonomous iterations on train + validation
  ↓
convergence
  ↓
FINAL_SELECTION
  ↓
freeze candidate/submission
  ↓
TEST_OPEN / final evaluation only
```

No earlier test metric should influence the research loop.

---

# 11. Resource Tracking

Track:

- number of iterations
- total LLM input tokens
- total LLM output tokens
- total agent wall-clock
- CPU/GPU usage
- number of manual interventions

Official run limits:

```text
50 iterations
6 hours wall-clock
```

---

# 12. Success Condition for the Setup AI

Do not start the final autonomous run until all of the following are true:

- source dataset exists and is untouched
- starter kit exists and baseline reproduces
- official evaluator is unchanged
- context files exist
- pre-audit is complete enough to provide reliable evidence
- constraints.md has been human-reviewed
- pipeline can reproduce baseline
- harness can execute and score validation
- timeout recovery works
- syntax-error recovery works
- NaN recovery works
- logging works
- test evaluation is blocked during development
- resource tracking works
- one rehearsal iteration completes end-to-end

After that, freeze the harness and agent infrastructure and begin the measured autonomous run.
