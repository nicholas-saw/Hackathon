"""Combine the initial and verification seeds for identity-field ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "experiment_results"


def runs(payload, config_id):
    entry = next(item for item in payload["results"] if item["config"]["id"] == config_id)
    return {int(run["seed"]): run["best_validation"] for run in entry["runs"]}


def main():
    initial = json.loads((RESULTS / "controlled_fm_experiments.json").read_text(encoding="utf-8"))
    extra = json.loads((RESULTS / "controlled_fm_identity_extra_seeds.json").read_text(encoding="utf-8"))
    base = {**runs(initial, "base"), **runs(extra, "base")}
    output = {"test_labels_accessed": False, "seeds": sorted(base), "comparisons": {}}
    for config_id in ("minus_author", "minus_video"):
        variant = {**runs(initial, config_id), **runs(extra, config_id)}
        primary = np.asarray([variant[s]["primary"] for s in sorted(base)])
        deltas = np.asarray([variant[s]["primary"] - base[s]["primary"] for s in sorted(base)])
        output["comparisons"][config_id] = {
            "primary_mean": float(primary.mean()), "primary_population_std": float(primary.std()),
            "paired_primary_deltas": deltas.tolist(),
            "paired_delta_mean": float(deltas.mean()),
            "paired_delta_population_std": float(deltas.std()),
            "all_paired_deltas_positive": bool((deltas > 0).all()),
        }
    path = RESULTS / "controlled_fm_identity_five_seed_summary.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
