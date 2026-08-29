"""Reviewer reproduction of the surprising C01 item-identity ablations.

Runs matched seeds for the full official field set and for individually
dropping video_id or author_id.  Train and validation only.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import common as C
import fm_utils as F


def main() -> None:
    train = C.load_train_log()
    valid = C.load_valid_log()
    vbasic = C.load_video_basic()
    configs = {
        "full_5field": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
        "drop_video_id": ["user_id", "author_id", "tab", "dur_bucket"],
        "drop_author_id": ["user_id", "video_id", "tab", "dur_bucket"],
    }
    out = {}
    for name, fields in configs.items():
        print(f"encoding {name}: {fields}", flush=True)
        Xtr, ytr, Xva, yva, uva, dim = F.encode_fields(train, valid, vbasic, fields)
        runs = []
        for seed in range(5):
            _, _, metrics, history = F.train_fm(
                Xtr, ytr, Xva, yva, uva, dim, seed=seed, verbose=False)
            run = {"seed": seed, "epochs": len(history), **{k: float(v) for k, v in metrics.items() if k in ("GAUC", "nDCG@5", "primary")}}
            runs.append(run)
            print(name, run, flush=True)
        out[name] = {
            "fields": fields,
            "runs": runs,
            "mean_primary": statistics.mean(x["primary"] for x in runs),
            "sample_std_primary": statistics.stdev(x["primary"] for x in runs),
        }

    base = {x["seed"]: x["primary"] for x in out["full_5field"]["runs"]}
    for name in ("drop_video_id", "drop_author_id"):
        deltas = [x["primary"] - base[x["seed"]] for x in out[name]["runs"]]
        out[name]["paired_delta_vs_full"] = {
            "per_seed": deltas,
            "mean": statistics.mean(deltas),
            "sample_std": statistics.stdev(deltas),
            "positive_in_n_of_5": sum(x > 0 for x in deltas),
        }

    path = ROOT / "research" / "review_artifacts" / "c01_ablation_reproduction.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
