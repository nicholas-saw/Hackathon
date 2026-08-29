"""Independent reviewer reproduction of core pre-audit statistics.

Only train and validation rows are materialized.  For the shared standard and
random log files, the CSV row's date is checked before any other field is
accessed; evaluation-period labels/features are never stored or inspected.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source" / "KuaiRand-Pure" / "data"
STARTER = ROOT / "source" / "starter-kit"
OUT = ROOT / "research" / "review_artifacts"

TRAIN_LO, TRAIN_HI = 20220408, 20220421
VALID_LO, VALID_HI = 20220422, 20220428
EVAL_LO, EVAL_HI = 20220429, 20220508


def load_date_range(path: Path, lo: int, hi: int) -> pd.DataFrame:
    """Materialize complete rows only when date is within [lo, hi]."""
    kept: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_i = header.index("date")
        for row in reader:
            date = int(row[date_i])
            if lo <= date <= hi:
                kept.append(row)
    frame = pd.DataFrame(kept, columns=header)
    # Match pandas' normal inference for all columns used below.
    for col in frame.columns:
        try:
            frame[col] = pd.to_numeric(frame[col])
        except (TypeError, ValueError):
            pass
    return frame


def count_dates_only(path: Path) -> dict[str, object]:
    counts: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_i = header.index("date")
        for row in reader:
            date = int(row[date_i])
            counts[date] = counts.get(date, 0) + 1
    return {
        "total_rows": int(sum(counts.values())),
        "min_date": min(counts),
        "max_date": max(counts),
        "counts_by_date": {str(k): v for k, v in sorted(counts.items())},
    }


def repeat_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, float | int]:
    c = df.groupby(cols, dropna=False).size()
    return {
        "unique_pairs": int(len(c)),
        "pct_pairs_repeated": float((c > 1).mean() * 100),
        "pct_rows_in_repeated_pairs": float(c[c > 1].sum() / c.sum() * 100),
        "max_repeat": int(c.max()),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    train_path = DATA / "log_standard_4_08_to_4_21_pure.csv"
    shared_path = DATA / "log_standard_4_22_to_5_08_pure.csv"
    random_path = DATA / "log_random_4_22_to_5_08_pure.csv"

    train = pd.read_csv(train_path)
    valid = load_date_range(shared_path, VALID_LO, VALID_HI)
    shared_dates = count_dates_only(shared_path)
    random_dates = count_dates_only(random_path)
    random_valid = load_date_range(random_path, VALID_LO, VALID_HI)
    vbasic = pd.read_csv(DATA / "video_features_basic_pure.csv")
    vstat = pd.read_csv(DATA / "video_features_statistic_pure.csv")
    ufeat = pd.read_csv(DATA / "user_features_pure.csv")

    vid_author = vbasic.set_index("video_id")["author_id"]
    vid_tag = vbasic.set_index("video_id")["tag"]
    for df in (train, valid):
        df["author_id"] = df["video_id"].map(vid_author)
        df["tag"] = df["video_id"].map(vid_tag)

    result: dict[str, object] = {}
    result["row_counts"] = {
        "train": len(train),
        "validation": len(valid),
        "evaluation_date_rows_date_only": sum(
            n for d, n in ((int(k), v) for k, v in shared_dates["counts_by_date"].items())
            if EVAL_LO <= d <= EVAL_HI
        ),
        "random_validation": len(random_valid),
        "random_evaluation_date_rows_date_only": sum(
            n for d, n in ((int(k), v) for k, v in random_dates["counts_by_date"].items())
            if EVAL_LO <= d <= EVAL_HI
        ),
    }

    result["cardinality"] = {
        "train_users": int(train.user_id.nunique()),
        "valid_users": int(valid.user_id.nunique()),
        "train_videos": int(train.video_id.nunique()),
        "valid_videos": int(valid.video_id.nunique()),
        "train_authors": int(train.author_id.nunique()),
        "valid_authors": int(valid.author_id.nunique()),
        "all_video_basic_authors": int(vbasic.author_id.nunique()),
        "all_video_basic_videos": int(vbasic.video_id.nunique()),
        "all_user_feature_users": int(ufeat.user_id.nunique()),
    }

    def overlap(valid_values, train_values) -> float:
        va, tr = set(valid_values), set(train_values)
        return len(va & tr) / len(va) * 100

    result["overlap_pct"] = {
        "users": overlap(valid.user_id, train.user_id),
        "videos": overlap(valid.video_id, train.video_id),
        "authors": overlap(valid.author_id, train.author_id),
        "user_video_pairs": overlap(zip(valid.user_id, valid.video_id), zip(train.user_id, train.video_id)),
        "user_author_pairs": overlap(zip(valid.user_id, valid.author_id), zip(train.user_id, train.author_id)),
        "user_tag_pairs": overlap(zip(valid.user_id, valid.tag), zip(train.user_id, train.tag)),
    }

    videos_per_author = vbasic.groupby("author_id").video_id.nunique()
    result["author_video"] = {
        "authors": int(len(videos_per_author)),
        "one_video_authors": int((videos_per_author == 1).sum()),
        "one_video_authors_pct": float((videos_per_author == 1).mean() * 100),
        "mean": float(videos_per_author.mean()),
        "median": float(videos_per_author.median()),
        "p90": float(videos_per_author.quantile(0.9)),
        "max": int(videos_per_author.max()),
    }

    result["repeat"] = {}
    for split, df in (("train", train), ("valid", valid)):
        result["repeat"][split] = {
            "user_video": repeat_stats(df, ["user_id", "video_id"]),
            "user_author": repeat_stats(df, ["user_id", "author_id"]),
            "user_tag": repeat_stats(df, ["user_id", "tag"]),
        }

    per_user = valid.groupby("user_id").long_view.agg(["sum", "count"])
    per_user["all_negative"] = per_user["sum"] == 0
    per_user["all_positive"] = per_user["sum"] == per_user["count"]
    per_user["mixed"] = ~(per_user.all_negative | per_user.all_positive)
    result["uniform_users"] = {
        key: {
            "count": int(per_user[key].sum()),
            "pct": float(per_user[key].mean() * 100),
        }
        for key in ("all_negative", "all_positive", "mixed")
    }
    result["list_length"] = {
        "min": int(per_user["count"].min()),
        "median": float(per_user["count"].median()),
        "mean": float(per_user["count"].mean()),
        "p90": float(per_user["count"].quantile(0.9)),
        "p99": float(per_user["count"].quantile(0.99)),
        "max": int(per_user["count"].max()),
    }

    train_counts = train.groupby("user_id").size()
    valid_u = pd.Index(sorted(valid.user_id.unique()))
    counts = pd.Series(valid_u.map(train_counts).fillna(0).astype(int), index=valid_u)
    warm = counts[counts > 0]
    edges = np.quantile(warm.to_numpy(), [0.25, 0.5, 0.75])
    def tier(c: int) -> str:
        if c == 0: return "Cold"
        if c <= edges[0]: return "T1"
        if c <= edges[1]: return "T2"
        if c <= edges[2]: return "T3"
        return "T4"
    tiers = counts.map(tier)
    valid["tier"] = valid.user_id.map(tiers)

    # The official GAUC denominator includes positives from mixed-label users only.
    mixed_users = set(per_user.index[per_user.mixed])
    valid["official_gauc_weight"] = np.where(valid.user_id.isin(mixed_users), valid.long_view, 0)
    total_gauc_weight = float(valid.official_gauc_weight.sum())
    valid["len_bucket"] = valid.user_id.map(pd.cut(
        per_user["count"], bins=[0, 1, 3, 5, 10, 20, np.inf],
        labels=["1", "2-3", "4-5", "6-10", "11-20", "21+"],
    ))
    result["official_gauc_weight_share_pct"] = {
        "activity_tier": {
            k: float(g.official_gauc_weight.sum() / total_gauc_weight * 100)
            for k, g in valid.groupby("tier")
        },
        "list_length": {
            str(k): float(g.official_gauc_weight.sum() / total_gauc_weight * 100)
            for k, g in valid.groupby("len_bucket", observed=True)
        },
        "denominator_positive_rows_from_mixed_users": int(total_gauc_weight),
    }

    # Cached score integrity and official-metric reproduction.
    cached_scores = np.load(ROOT / "research" / "experiment_results" / "baseline_seed0_valid_scores.npy")
    cached_users = np.load(ROOT / "research" / "experiment_results" / "baseline_seed0_valid_users.npy", allow_pickle=True)
    cached_labels = np.load(ROOT / "research" / "experiment_results" / "baseline_seed0_valid_labels.npy")
    sys.path.insert(0, str(STARTER))
    from evaluate import evaluate
    result["cached_baseline"] = {
        "users_exactly_aligned": bool(np.array_equal(cached_users.astype(str), valid.user_id.astype(str).to_numpy())),
        "labels_exactly_aligned": bool(np.array_equal(cached_labels, valid.long_view.to_numpy())),
        "metrics": evaluate(cached_users, cached_labels, cached_scores),
        "oracle": evaluate(valid.user_id, valid.long_view, valid.long_view),
    }

    result["temporal"] = {
        "train_daily_rows": {str(int(k)): int(v) for k, v in train.groupby("date").size().items()},
        "valid_daily_rows": {str(int(k)): int(v) for k, v in valid.groupby("date").size().items()},
        "train_dates": [int(x) for x in sorted(train.date.unique())],
    }

    feedback = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "is_profile_enter"]
    result["feedback_prevalence_valid_pct"] = {c: float(valid[c].mean() * 100) for c in feedback}

    result["history"] = {
        "valid_users_ge1": float((counts >= 1).mean() * 100),
        "valid_users_ge5": float((counts >= 5).mean() * 100),
        "valid_users_ge10": float((counts >= 10).mean() * 100),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "p90": float(counts.quantile(0.9)),
    }

    rv_uv = set(zip(random_valid.user_id, random_valid.video_id))
    vv_uv = set(zip(valid.user_id, valid.video_id))
    result["random_validation_only"] = {
        "rows": len(random_valid),
        "users": int(random_valid.user_id.nunique()),
        "videos": int(random_valid.video_id.nunique()),
        "is_rand_one_pct": float((random_valid.is_rand == 1).mean() * 100),
        "long_view_rate": float(random_valid.long_view.mean()),
        "unique_user_video_pairs": len(rv_uv),
        "shared_pairs_with_standard_valid": len(rv_uv & vv_uv),
        "shared_pair_pct": len(rv_uv & vv_uv) / len(rv_uv) * 100,
    }

    result["source_hashes_sha256"] = {
        name: sha256(STARTER / name)
        for name in ("evaluate.py", "data.py", "baseline.py", "baseline_scores.json", "ablation_features.py", "README.md")
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / "core_reproduction.json"
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
