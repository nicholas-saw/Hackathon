"""Small, predeclared validation-only FM experiment matrix for the pre-audit.

This reproduces field, static-feature, embedding-size, learning-rate, and seed
checks without ever reading evaluation labels. It is deliberately not a model
search: all configurations are fixed below and use the official FM training
implementation and evaluator unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

import baseline_validation as BV


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source" / "KuaiRand-Pure" / "data"
OUTPUT = ROOT / "research" / "experiment_results" / "controlled_fm_experiments.json"
SEEDS = (0, 1, 2)


CONFIGS = [
    {"id": "base", "feature_mode": "base", "k": 16, "lr": 0.001},
    {"id": "minus_author", "feature_mode": "minus_author", "k": 16, "lr": 0.001},
    {"id": "minus_tab", "feature_mode": "minus_tab", "k": 16, "lr": 0.001},
    {"id": "minus_duration", "feature_mode": "minus_duration", "k": 16, "lr": 0.001},
    {"id": "minus_video", "feature_mode": "minus_video", "k": 16, "lr": 0.001},
    {"id": "static_item", "feature_mode": "static_item", "k": 16, "lr": 0.001},
    {"id": "static_cwm13", "feature_mode": "static_cwm13", "k": 16, "lr": 0.001},
    {"id": "k8", "feature_mode": "base", "k": 8, "lr": 0.001},
    {"id": "k32", "feature_mode": "base", "k": 32, "lr": 0.001},
    {"id": "lr0005", "feature_mode": "base", "k": 16, "lr": 0.0005},
    {"id": "lr002", "feature_mode": "base", "k": 16, "lr": 0.002},
]


BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
USER_STATIC_FIELDS = [
    "follow_user_num_range", "register_days_range", "fans_user_num_range",
    "friend_user_num_range", "user_active_degree",
]
ITEM_STATIC_FIELDS = ["music_id", "video_type", "upload_type"]


def load_static(data_dir: Path):
    user = {}
    with (data_dir / "user_features_pure.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            user[row["user_id"]] = tuple(row[field] for field in USER_STATIC_FIELDS)
    item = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            item[row["video_id"]] = tuple(row[field] for field in ITEM_STATIC_FIELDS)
    return user, item


def fields_for_mode(mode):
    if mode == "base": return BASE_FIELDS
    if mode == "minus_author": return [f for f in BASE_FIELDS if f != "author_id"]
    if mode == "minus_tab": return [f for f in BASE_FIELDS if f != "tab"]
    if mode == "minus_duration": return [f for f in BASE_FIELDS if f != "dur_bucket"]
    if mode == "minus_video": return [f for f in BASE_FIELDS if f != "video_id"]
    if mode == "static_item": return BASE_FIELDS + ITEM_STATIC_FIELDS
    if mode == "static_cwm13": return BASE_FIELDS + ITEM_STATIC_FIELDS + USER_STATIC_FIELDS
    raise ValueError(mode)


def encode(splits, mode, user_static, item_static):
    fields = fields_for_mode(mode)
    edges = np.quantile(
        np.asarray([row[5] for row in splits["train"]]),
        np.linspace(0, 1, 11)[1:-1],
    )
    unknown_user = ("UNK",) * len(USER_STATIC_FIELDS)
    unknown_item = ("UNK",) * len(ITEM_STATIC_FIELDS)

    def values(row):
        base = {
            "user_id": row[1], "video_id": row[2], "author_id": row[3],
            "tab": row[4], "dur_bucket": str(int(np.searchsorted(edges, row[5]))),
        }
        item_values = item_static.get(row[2], unknown_item)
        user_values = user_static.get(row[1], unknown_user)
        base.update(dict(zip(ITEM_STATIC_FIELDS, item_values)))
        base.update(dict(zip(USER_STATIC_FIELDS, user_values)))
        return [base[field] for field in fields]

    vocabularies = [dict() for _ in fields]
    for row in splits["train"]:
        for i, value in enumerate(values(row)):
            if value not in vocabularies[i]:
                vocabularies[i][value] = len(vocabularies[i])
    unknown = [len(vocab) for vocab in vocabularies]
    dims = [len(vocab) + 1 for vocab in vocabularies]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    result = {}
    for split, rows in splits.items():
        X = np.empty((len(rows), len(fields)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for n, row in enumerate(rows):
            for i, value in enumerate(values(row)):
                X[n, i] = vocabularies[i].get(value, unknown[i]) + offsets[i]
            y[n] = row[6]
            users.append(row[1])
        result[split] = X, y, users
    return result, int(sum(dims)), dict(zip(fields, dims)), fields


def train_once(encoded, dim, config, seed, epochs=40, batch_size=8192, patience=4):
    Xtr, ytr, _ = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    model = BV.OFFICIAL_BASELINE.FM(dim, k=config["k"], lr=config["lr"], seed=seed)
    rng = np.random.default_rng(seed)
    best = -1.0
    best_metrics = None
    bad = 0
    trace = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(ytr))
        losses = []
        for start in range(0, len(order), batch_size):
            ix = order[start : start + batch_size]
            losses.append(model.step(Xtr[ix], ytr[ix]))
        metrics = BV.OFFICIAL_EVALUATE.evaluate(uva, yva, model.predict(Xva))
        trace.append({"epoch": epoch, "loss": float(np.mean(losses)), "metrics": metrics})
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            best_metrics = dict(metrics)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return {
        "seed": seed, "best_validation": best_metrics, "epochs_run": len(trace),
        "seconds": time.perf_counter() - started, "trace": trace,
    }


def summarize(config_results, baseline_mean):
    primary = [r["best_validation"]["primary"] for r in config_results]
    gauc = [r["best_validation"]["GAUC"] for r in config_results]
    ndcg = [r["best_validation"]["nDCG@5"] for r in config_results]
    return {
        "n_seeds": len(primary),
        "GAUC_mean": statistics.mean(gauc), "GAUC_population_std": statistics.pstdev(gauc),
        "nDCG@5_mean": statistics.mean(ndcg), "nDCG@5_population_std": statistics.pstdev(ndcg),
        "primary_mean": statistics.mean(primary), "primary_population_std": statistics.pstdev(primary),
        "delta_primary_vs_base_mean": statistics.mean(primary) - baseline_mean,
    }


def write_output(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=BV._json_default), encoding="utf-8")


def main(args):
    global SEEDS
    if args.seeds:
        SEEDS = tuple(args.seeds)
    started = time.perf_counter()
    splits = BV.load_train_valid(args.data_dir)
    if {k: len(v) for k, v in splits.items()} != {"train": 1_141_112, "valid": 124_909}:
        raise RuntimeError("Split count guard failed")
    user_static, item_static = load_static(args.data_dir)
    payload = {
        "purpose": "predeclared controlled FM validation-only evidence",
        "test_labels_accessed": False, "official_source_modified": False,
        "seeds": list(SEEDS), "max_epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "results": [],
        "organizer_ablation_schema_note": {
            "base_fields": BASE_FIELDS,
            "new_item_fields": ITEM_STATIC_FIELDS,
            "new_user_fields": USER_STATIC_FIELDS,
            "static_item_total_fields": len(BASE_FIELDS + ITEM_STATIC_FIELDS),
            "static_cwm_total_fields": len(BASE_FIELDS + ITEM_STATIC_FIELDS + USER_STATIC_FIELDS),
        },
    }
    baseline_mean = None
    selected_configs = [c for c in CONFIGS if not args.configs or c["id"] in args.configs]
    if not selected_configs or selected_configs[0]["id"] != "base":
        raise ValueError("Selected configurations must include base first so deltas are defined")
    for config in selected_configs:
        encode_start = time.perf_counter()
        encoded, dim, field_dims, fields = encode(
            splits, config["feature_mode"], user_static, item_static
        )
        encode_seconds = time.perf_counter() - encode_start
        runs = []
        for seed in SEEDS:
            run = train_once(encoded, dim, config, seed, args.epochs, args.batch_size, args.patience)
            runs.append(run)
            print(
                f"{config['id']} seed={seed} primary={run['best_validation']['primary']:.6f} "
                f"epochs={run['epochs_run']} seconds={run['seconds']:.1f}",
                flush=True,
            )
        if config["id"] == "base":
            baseline_mean = statistics.mean(r["best_validation"]["primary"] for r in runs)
        entry = {
            "config": config, "fields": fields, "field_dims_including_unk": field_dims,
            "total_parameter_ids": dim, "encoding_seconds": encode_seconds,
            "runs": runs, "summary": summarize(runs, baseline_mean),
        }
        payload["results"].append(entry)
        payload["elapsed_seconds"] = time.perf_counter() - started
        write_output(args.output, payload)
    print(
        json.dumps(
            {r["config"]["id"]: r["summary"] for r in payload["results"]},
            indent=2,
            default=BV._json_default,
        ),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--configs", nargs="*", choices=[c["id"] for c in CONFIGS])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
