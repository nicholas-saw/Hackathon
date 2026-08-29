"""Validation-only reproduction of the official KuaiRand-Pure FM baseline.

This research script intentionally never reads the label cell for a standard-log
row dated after 2022-04-28. It imports the official FM and evaluator unchanged,
but replaces the official loader because that loader materializes test labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


# Keep the official source tree physically read-only when importing reference code.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "source" / "starter-kit"
DEFAULT_DATA = ROOT / "source" / "KuaiRand-Pure" / "data"
DEFAULT_OUTPUT = ROOT / "research" / "experiment_results" / "baseline_validation.json"
DEFAULT_PREDICTIONS = (
    ROOT / "research" / "experiment_results" / "baseline_validation_predictions.npz"
)
TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


OFFICIAL_EVALUATE = _load_module("evaluate", STARTER / "evaluate.py")
_load_module("data", STARTER / "data.py")
OFFICIAL_BASELINE = _load_module("official_baseline", STARTER / "baseline.py")


def load_train_valid(data_dir: Path):
    """Load only train/validation; never access labels after validation end."""
    vid2author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        encoding="utf-8-sig", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            vid2author[row["video_id"]] = row["author_id"]

    splits = {"train": [], "valid": []}
    names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for name in names:
        with (data_dir / name).open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ix = {column: i for i, column in enumerate(header)}
            for row in reader:
                date = int(row[ix["date"]])
                if TRAIN_RANGE[0] <= date <= TRAIN_RANGE[1]:
                    split = "train"
                elif VALID_RANGE[0] <= date <= VALID_RANGE[1]:
                    split = "valid"
                else:
                    # Crucially, no access to this row's label or feedback fields.
                    continue
                user = row[ix["user_id"]]
                video = row[ix["video_id"]]
                splits[split].append(
                    (
                        date,
                        user,
                        video,
                        vid2author.get(video, "UNK"),
                        row[ix["tab"]],
                        float(row[ix["duration_ms"]]),
                        1 if row[ix["long_view"]] != "0" else 0,
                    )
                )
    return splits


def encode_train_valid(splits):
    edges = np.quantile(
        np.asarray([row[5] for row in splits["train"]]),
        np.linspace(0, 1, 11)[1:-1],
    )

    def raw(row):
        return [
            row[1],
            row[2],
            row[3],
            row[4],
            str(int(np.searchsorted(edges, row[5]))),
        ]

    vocabs = [dict() for _ in FIELDS]
    for row in splits["train"]:
        for i, value in enumerate(raw(row)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])
    unknown = [len(vocab) for vocab in vocabs]
    field_dims = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    encoded = {}
    for split, rows in splits.items():
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for n, row in enumerate(rows):
            values = raw(row)
            for i, value in enumerate(values):
                X[n, i] = vocabs[i].get(value, unknown[i]) + offsets[i]
            y[n] = row[6]
            users.append(row[1])
        encoded[split] = (X, y, users)
    return encoded, int(sum(field_dims)), field_dims, edges


def run(args):
    t0 = time.perf_counter()
    splits = load_train_valid(args.data_dir)
    load_seconds = time.perf_counter() - t0
    actual_counts = {key: len(value) for key, value in splits.items()}
    expected_counts = {"train": 1_141_112, "valid": 124_909}
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {actual_counts} != {expected_counts}")

    t1 = time.perf_counter()
    encoded, dim, field_dims, edges = encode_train_valid(splits)
    encode_seconds = time.perf_counter() - t1
    Xtr, ytr, _ = encoded["train"]
    Xva, yva, uva = encoded["valid"]

    model = OFFICIAL_BASELINE.FM(dim, k=args.k, lr=args.lr, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best = -1.0
    best_state = None
    bad = 0
    epochs = []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(ytr))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            losses.append(model.step(Xtr[batch], ytr[batch]))
        eval_start = time.perf_counter()
        metrics = OFFICIAL_EVALUATE.evaluate(uva, yva, model.predict(Xva))
        eval_seconds = time.perf_counter() - eval_start
        epochs.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "metrics": metrics,
                "evaluation_seconds": eval_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"epoch {epoch:02d} loss={np.mean(losses):.6f} "
            f"GAUC={metrics['GAUC']:.6f} nDCG@5={metrics['nDCG@5']:.6f} "
            f"primary={metrics['primary']:.6f}"
        )
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("No checkpoint was created")
    model.V, model.W, model.b = best_state
    final_eval_start = time.perf_counter()
    final_scores = model.predict(Xva)
    final_metrics = OFFICIAL_EVALUATE.evaluate(uva, yva, final_scores)
    final_eval_seconds = time.perf_counter() - final_eval_start

    result = {
        "purpose": "official FM reproduction using train and validation only",
        "test_labels_accessed": False,
        "official_source_modified": False,
        "split_ranges": {"train": TRAIN_RANGE, "valid": VALID_RANGE},
        "split_rows": actual_counts,
        "fields": FIELDS,
        "field_dims_including_unk": dict(zip(FIELDS, field_dims)),
        "duration_quantile_edges_ms": edges.tolist(),
        "config": {
            "k": args.k,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "l2": 1e-6,
            "optimizer": "Adam (official numpy implementation)",
            "objective": "pointwise binary cross-entropy/logistic loss",
        },
        "timings_seconds": {
            "load": load_seconds,
            "encode": encode_seconds,
            "train_including_epoch_evaluations": train_seconds,
            "final_evaluation": final_eval_seconds,
            "cold_total": time.perf_counter() - t0,
        },
        "best_validation": final_metrics,
        "epochs": epochs,
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        scores=np.asarray(final_scores, dtype=np.float32),
        labels=np.asarray(yva, dtype=np.float32),
        users=np.asarray(uva, dtype=np.int32),
    )
    print(
        json.dumps(
            {"best_validation": final_metrics, "timings_seconds": result["timings_seconds"]},
            indent=2,
            default=_json_default,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
