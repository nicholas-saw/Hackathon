# techjam-track2 — Autonomous ML Research Agent (KuaiRand-Pure)

Autonomous machine learning research agent for recommender systems, built against the
KuaiRand-Pure benchmark. See [context/PROBLEM.md](context/PROBLEM.md) for the official
task definition and [context/RULES.md](context/RULES.md) for non-negotiable constraints.

## Layout

| Path | Contents | Mutability |
|---|---|---|
| [source/](source/) | Raw KuaiRand-Pure dataset + official starter kit | read-only |
| [context/](context/) | Problem definition, rules, data guide, verified constraints, references | read-only during runs |
| [research/](research/) | Pre-audit notebook, data profile, analysis scripts/plots/results | editable pre-run |
| [pipeline/](pipeline/) | Data adapter, features, training, models, objectives | editable by the coding agent |
| [harness/](harness/) | Scoring, execution, guardrails, caching, diagnostics, logging, submission | read-only during the final run |
| [agent/](agent/) | Controller, proposer, coder, reflector, governor, prompts | fixed before the measured run |
| runlogs/ | Per-run iteration journals | agent-created output |
| submissions/ | Frozen candidate/final submissions | agent-created output |
| reports/ | Results and resource-consumption summaries | agent-created output |
| tests/ | Harness/pipeline tests | editable |

## Status

Folder structure created; pre-audit ([research/PRE_AUDIT.md](research/PRE_AUDIT.md),
[research/data_profile.md](research/data_profile.md)) is still all `TODO`. Per
[SETUP_AND_FOLDER_STRUCTURE.md](SETUP_AND_FOLDER_STRUCTURE.md) section 12, the final
autonomous run must not start until the pre-audit is complete, `context/constraints.md`
has been human-reviewed, and the pipeline/harness reproduce the baseline end-to-end.

## Setup

See [SETUP_AND_FOLDER_STRUCTURE.md](SETUP_AND_FOLDER_STRUCTURE.md) for the full
project workflow, agent roles, hypothesis schema, logging format, and run limits.
