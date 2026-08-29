"""Reproducible train+validation-only KuaiRand-Pure pre-audit profile.

The second standard log also contains evaluation rows. This script checks the
date before accessing any label or post-impression cell, so evaluation labels
are never inspected or materialized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source" / "KuaiRand-Pure" / "data"
RESULTS = ROOT / "research" / "experiment_results"
PLOTS = ROOT / "research" / "plots"
VALID_END = 20220428
LOG_COLUMNS = [
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click",
    "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
]
BINARY_SIGNALS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter",
]
CONTINUOUS_SIGNALS = ["play_time_ms", "profile_stay_time", "comment_stay_time"]
SIGNALS = BINARY_SIGNALS + CONTINUOUS_SIGNALS


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def load_official_evaluator():
    path = ROOT / "source" / "starter-kit" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("audit_official_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_official_evaluator()


def log_dtypes():
    small = {
        "user_id": "int32", "video_id": "int32", "date": "int32",
        "hourmin": "int16", "time_ms": "int64", "play_time_ms": "int64",
        "duration_ms": "int64", "profile_stay_time": "int64",
        "comment_stay_time": "int64", "tab": "int8", "is_rand": "int8",
    }
    for column in BINARY_SIGNALS + ["long_view"]:
        small[column] = "int8"
    return small


def load_train_valid(data_dir: Path):
    timings = {}
    t0 = time.perf_counter()
    train = pd.read_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=LOG_COLUMNS,
        dtype=log_dtypes(),
    )
    timings["train_csv_load"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    values = {column: [] for column in LOG_COLUMNS}
    path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ix = {column: i for i, column in enumerate(header)}
        for row in reader:
            date = int(row[ix["date"]])
            if date > VALID_END:
                # Do not access label/feedback cells on evaluation-period rows.
                continue
            for column in LOG_COLUMNS:
                values[column].append(row[ix[column]])
    valid = pd.DataFrame(values)
    for column, dtype in log_dtypes().items():
        valid[column] = pd.to_numeric(valid[column], errors="raise").astype(dtype)
    timings["validation_csv_guarded_load"] = time.perf_counter() - t1
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Split mismatch: train={len(train)} valid={len(valid)}")
    if train["date"].max() > 20220421 or valid["date"].max() > VALID_END:
        raise RuntimeError("Date guard failed")
    return train, valid, timings


def describe_numeric(series: pd.Series):
    q = series.quantile([0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "min": float(q.loc[0.0]), "median": float(q.loc[0.5]),
        "mean": float(series.mean()), "std": float(series.std()),
        "p90": float(q.loc[0.9]), "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]), "max": float(q.loc[1.0]),
        "zero_pct": float((series == 0).mean() * 100),
    }


def missing_inventory(name: str, frame: pd.DataFrame):
    rows = []
    for column in frame.columns:
        series = frame[column]
        blank = series.astype("string").str.strip().eq("").fillna(False)
        missing = series.isna() | blank
        rows.append({
            "file_or_split": name,
            "field": column,
            "missing_count": int(missing.sum()),
            "missing_pct": float(missing.mean() * 100),
            "cardinality_nonmissing": int(series[~missing].nunique(dropna=True)),
            "dtype": str(series.dtype),
        })
    return rows


def pair_set(frame: pd.DataFrame, left: str, right: str):
    return set(zip(frame[left].astype(int), frame[right].astype(object)))


def tag_tokens(value):
    if pd.isna(value) or str(value).strip() == "":
        return ()
    return tuple(sorted({int(token.strip()) for token in str(value).split(",") if token.strip()}))


def user_metrics(valid: pd.DataFrame, scores: np.ndarray):
    work = valid[["user_id", "long_view"]].copy()
    work["score"] = scores
    rows = []
    for user, group in work.groupby("user_id", sort=False):
        labels = group["long_view"].to_numpy().tolist()
        preds = group["score"].to_numpy().tolist()
        npos = int(sum(labels))
        n = len(labels)
        order = np.argsort(-np.asarray(preds), kind="stable")
        ordered = np.asarray(labels)[order].tolist()
        mixed = 0 < npos < n
        rows.append({
            "user_id": int(user), "n": n, "npos": npos,
            "positive_rate": npos / n, "mixed": mixed,
            "all_negative": npos == 0, "all_positive": npos == n,
            "auc": EVALUATOR.auc(labels, preds) if mixed else np.nan,
            "ndcg": EVALUATOR.ndcg_at_k(ordered, 5),
            "oracle_ndcg": EVALUATOR.ndcg_at_k(sorted(labels, reverse=True), 5),
        })
    return pd.DataFrame(rows)


def bucket_metrics(valid, scores, users, bucket_name, order):
    total_gauc_weight = float(users.loc[users["mixed"], "npos"].sum())
    total_users = len(users)
    out = []
    user_to_bucket = users.set_index("user_id")[bucket_name]
    row_bucket = valid["user_id"].map(user_to_bucket)
    for bucket in order:
        selected_users = users[users[bucket_name] == bucket]
        mask = row_bucket.eq(bucket).to_numpy()
        if not mask.any():
            continue
        metrics = EVALUATOR.evaluate(
            valid.loc[mask, "user_id"].tolist(),
            valid.loc[mask, "long_view"].tolist(),
            scores[mask].tolist(),
        )
        oracle = EVALUATOR.evaluate(
            valid.loc[mask, "user_id"].tolist(),
            valid.loc[mask, "long_view"].tolist(),
            valid.loc[mask, "long_view"].tolist(),
        )
        gap = oracle["nDCG@5"] - metrics["nDCG@5"]
        weight = float(selected_users.loc[selected_users["mixed"], "npos"].sum())
        out.append({
            "bucket": str(bucket), "users": int(len(selected_users)),
            "validation_rows": int(mask.sum()),
            "GAUC": metrics["GAUC"], "nDCG@5": metrics["nDCG@5"],
            "primary": metrics["primary"], "oracle_nDCG@5": oracle["nDCG@5"],
            "nDCG_gap": gap,
            "overall_nDCG_gap_contribution": gap * len(selected_users) / total_users,
            "invariant_user_pct": float((~selected_users["mixed"]).mean() * 100),
            "GAUC_weight_share_pct": 100 * weight / total_gauc_weight if total_gauc_weight else 0.0,
        })
    return pd.DataFrame(out)


def js_for_categories(a: pd.Series, b: pd.Series):
    cats = sorted(set(a.dropna().unique()) | set(b.dropna().unique()))
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return float(jensenshannon(pa, pb, base=2.0) ** 2)


def period_summary(frame, name):
    daily = frame.groupby("date").agg(
        rows=("user_id", "size"), long_view_rate=("long_view", "mean"),
        unique_users=("user_id", "nunique"), unique_videos=("video_id", "nunique"),
        mean_duration_ms=("duration_ms", "mean"), median_duration_ms=("duration_ms", "median"),
    )
    return {
        "period": name, "days": int(frame["date"].nunique()),
        "rows": int(len(frame)), "rows_per_day": float(daily["rows"].mean()),
        "long_view_rate": float(frame["long_view"].mean()),
        "unique_users_total": int(frame["user_id"].nunique()),
        "unique_videos_total": int(frame["video_id"].nunique()),
        "unique_users_per_day": float(daily["unique_users"].mean()),
        "unique_videos_per_day": float(daily["unique_videos"].mean()),
        "mean_duration_ms": float(frame["duration_ms"].mean()),
        "median_duration_ms": float(frame["duration_ms"].median()),
    }


def score_static(valid, values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=np.nanmedian(values), posinf=0, neginf=0)
    return EVALUATOR.evaluate(valid["user_id"].tolist(), valid["long_view"].tolist(), values.tolist())


def main(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = psutil.Process()
    train, valid, load_timings = load_train_valid(args.data_dir)
    rss_after_logs = process.memory_info().rss
    basic = pd.read_csv(args.data_dir / "video_features_basic_pure.csv")
    stats = pd.read_csv(args.data_dir / "video_features_statistic_pure.csv")
    user_features = pd.read_csv(args.data_dir / "user_features_pure.csv")
    feature_load_seconds = time.perf_counter() - started - sum(load_timings.values())

    predictions = np.load(args.predictions)
    scores = predictions["scores"].astype(float)
    if len(scores) != len(valid):
        raise RuntimeError("Prediction row count does not match validation")
    if not np.array_equal(predictions["labels"].astype(np.int8), valid["long_view"].to_numpy()):
        raise RuntimeError("Prediction artifact label alignment failed")
    if not np.array_equal(predictions["users"].astype(np.int32), valid["user_id"].to_numpy()):
        raise RuntimeError("Prediction artifact user alignment failed")

    vid_author = basic.set_index("video_id")["author_id"]
    for frame in (train, valid):
        frame["author_id"] = frame["video_id"].map(vid_author)

    # A: structure, cardinality, missingness, overlap, repeats.
    cardinalities = {}
    for name, frame in (("train", train), ("validation", valid)):
        cardinalities[name] = {
            column: int(frame[column].nunique(dropna=True))
            for column in ["user_id", "video_id", "author_id", "tab", "hourmin", "date", "duration_ms"]
        }
        cardinalities[name]["rows"] = int(len(frame))

    train_users, valid_users = set(train.user_id), set(valid.user_id)
    train_videos, valid_videos = set(train.video_id), set(valid.video_id)
    train_authors, valid_authors = set(train.author_id.dropna()), set(valid.author_id.dropna())
    train_uv = pair_set(train, "user_id", "video_id")
    valid_uv = pair_set(valid, "user_id", "video_id")
    train_ua = pair_set(train.dropna(subset=["author_id"]), "user_id", "author_id")
    valid_ua = pair_set(valid.dropna(subset=["author_id"]), "user_id", "author_id")

    video_tags = {int(v): tag_tokens(t) for v, t in zip(basic.video_id, basic.tag)}
    train_ut_counts = Counter()
    for u, v in zip(train.user_id.to_numpy(), train.video_id.to_numpy()):
        for tag in video_tags.get(int(v), ()):
            train_ut_counts[int(u) * 256 + tag] += 1
    train_ut = set(train_ut_counts)
    valid_ut = set()
    valid_row_any_prior_tag = np.zeros(len(valid), dtype=bool)
    for i, (u, v) in enumerate(zip(valid.user_id.to_numpy(), valid.video_id.to_numpy())):
        tags = video_tags.get(int(v), ())
        keys = [int(u) * 256 + tag for tag in tags]
        valid_ut.update(keys)
        valid_row_any_prior_tag[i] = any(key in train_ut for key in keys)

    overlap = {
        "validation_users_seen_count": len(valid_users & train_users),
        "validation_users_seen_pct": 100 * len(valid_users & train_users) / len(valid_users),
        "cold_validation_users_count": len(valid_users - train_users),
        "cold_validation_users_pct": 100 * len(valid_users - train_users) / len(valid_users),
        "validation_videos_seen_count": len(valid_videos & train_videos),
        "validation_videos_seen_pct": 100 * len(valid_videos & train_videos) / len(valid_videos),
        "cold_validation_videos_count": len(valid_videos - train_videos),
        "cold_validation_videos_pct": 100 * len(valid_videos - train_videos) / len(valid_videos),
        "validation_authors_seen_count": len(valid_authors & train_authors),
        "validation_authors_seen_pct": 100 * len(valid_authors & train_authors) / len(valid_authors),
        "validation_user_video_pairs_seen_count": len(valid_uv & train_uv),
        "validation_user_video_pairs_seen_pct": 100 * len(valid_uv & train_uv) / len(valid_uv),
        "validation_user_author_pairs_seen_count": len(valid_ua & train_ua),
        "validation_user_author_pairs_seen_pct": 100 * len(valid_ua & train_ua) / len(valid_ua),
        "validation_user_tag_pairs_seen_count": len(valid_ut & train_ut),
        "validation_user_tag_pairs_seen_pct": 100 * len(valid_ut & train_ut) / len(valid_ut),
        "validation_rows_with_any_prior_user_tag_pct": float(valid_row_any_prior_tag.mean() * 100),
    }

    uv_counts = train.groupby(["user_id", "video_id"], sort=False).size()
    ua_counts = train.groupby(["user_id", "author_id"], sort=False).size()
    repeat = {
        "train_unique_user_video_pairs": int(len(uv_counts)),
        "train_user_video_pair_count_distribution": describe_numeric(uv_counts),
        "train_user_video_pairs_repeated_pct": float((uv_counts > 1).mean() * 100),
        "train_rows_in_repeated_user_video_pairs_pct": float(uv_counts[uv_counts > 1].sum() / len(train) * 100),
        "train_unique_user_author_pairs": int(len(ua_counts)),
        "train_user_author_pair_count_distribution": describe_numeric(ua_counts),
        "train_user_author_pairs_repeated_pct": float((ua_counts > 1).mean() * 100),
        "train_rows_in_repeated_user_author_pairs_pct": float(ua_counts[ua_counts > 1].sum() / len(train) * 100),
        "train_unique_user_tag_pairs": int(len(train_ut)),
        "train_user_tag_pair_count_distribution": describe_numeric(pd.Series(list(train_ut_counts.values()))),
        "train_user_tag_pairs_repeated_pct": float(np.mean(np.fromiter(train_ut_counts.values(), dtype=np.int64) > 1) * 100),
        "train_tag_interactions_in_repeated_user_tag_pairs_pct": float(
            sum(count for count in train_ut_counts.values() if count > 1) / sum(train_ut_counts.values()) * 100
        ),
    }

    observed_basic = basic[basic.video_id.isin(train_videos | valid_videos)].copy()
    author_video_counts = observed_basic.groupby("author_id").video_id.nunique()
    full_author_video_counts = basic.groupby("author_id").video_id.nunique()
    author_structure = {
        "observed_train_valid_videos_with_basic_features_pct": 100 * len(set(observed_basic.video_id)) / len(train_videos | valid_videos),
        "observed_authors": int(len(author_video_counts)),
        "videos_per_author": describe_numeric(author_video_counts),
        "authors_with_exactly_one_video_count": int((author_video_counts == 1).sum()),
        "authors_with_exactly_one_video_pct": float((author_video_counts == 1).mean() * 100),
        "full_basic_authors": int(len(full_author_video_counts)),
        "full_basic_authors_with_exactly_one_video_pct": float((full_author_video_counts == 1).mean() * 100),
        "videos_with_multiple_authors_in_basic": int(basic.groupby("video_id").author_id.nunique().gt(1).sum()),
        "observed_video_to_author_functional_mapping_pct": float(
            observed_basic.groupby("video_id").author_id.nunique().eq(1).mean() * 100
        ),
    }
    user_feature_ids = set(user_features.user_id.astype(int))
    feature_coverage = {
        "user_feature_rows": int(len(user_features)),
        "train_users_with_user_features_pct": 100 * len(train_users & user_feature_ids) / len(train_users),
        "validation_users_with_user_features_pct": 100 * len(valid_users & user_feature_ids) / len(valid_users),
        "video_basic_rows": int(len(basic)),
        "train_videos_with_basic_features_pct": 100 * len(train_videos & set(basic.video_id)) / len(train_videos),
        "validation_videos_with_basic_features_pct": 100 * len(valid_videos & set(basic.video_id)) / len(valid_videos),
        "video_statistic_rows": int(len(stats)),
        "train_videos_with_statistic_features_pct": 100 * len(train_videos & set(stats.video_id)) / len(train_videos),
        "validation_videos_with_statistic_features_pct": 100 * len(valid_videos & set(stats.video_id)) / len(valid_videos),
    }

    missing_rows = []
    for name, frame in (("train", train), ("validation", valid), ("user_features", user_features),
                        ("video_basic", basic), ("video_statistics", stats)):
        missing_rows.extend(missing_inventory(name, frame))
    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(RESULTS / "missingness_inventory.csv", index=False)

    # B: metric structure with official evaluator and saved official-FM scores.
    per_user = user_metrics(valid, scores)
    train_activity = train.groupby("user_id").size().rename("train_interactions")
    per_user = per_user.join(train_activity, on="user_id")
    per_user["train_interactions"] = per_user["train_interactions"].fillna(0).astype(int)
    seen_counts = train_activity.to_numpy()
    q25, q50, q75 = np.quantile(seen_counts, [0.25, 0.5, 0.75], method="nearest")
    def activity_tier(n):
        if n == 0: return "Cold"
        if n <= q25: return "T1"
        if n <= q50: return "T2"
        if n <= q75: return "T3"
        return "T4"
    per_user["activity_tier"] = per_user.train_interactions.map(activity_tier)
    per_user["list_bucket"] = pd.cut(
        per_user.n, bins=[0, 1, 3, 5, 10, 20, np.inf],
        labels=["1", "2-3", "4-5", "6-10", "11-20", "21+"], include_lowest=True,
    ).astype(str)
    activity_metrics = bucket_metrics(valid, scores, per_user, "activity_tier", ["Cold", "T1", "T2", "T3", "T4"])
    list_metrics = bucket_metrics(valid, scores, per_user, "list_bucket", ["1", "2-3", "4-5", "6-10", "11-20", "21+"])
    activity_metrics.to_csv(RESULTS / "metric_by_activity_bucket.csv", index=False)
    list_metrics.to_csv(RESULTS / "metric_by_list_length.csv", index=False)
    per_user.to_csv(RESULTS / "validation_user_metric_profile.csv", index=False)

    uniform = {}
    for key in ["all_negative", "all_positive", "mixed"]:
        mask = per_user[key]
        uniform[key] = {
            "users": int(mask.sum()), "users_pct": float(mask.mean() * 100),
            "rows": int(per_user.loc[mask, "n"].sum()),
            "rows_pct": float(per_user.loc[mask, "n"].sum() / len(valid) * 100),
        }
    metric_structure = {
        "validation_users": int(len(per_user)),
        "validation_impressions_per_user": describe_numeric(per_user.n),
        "train_impressions_per_user": describe_numeric(train_activity),
        "positive_rate_per_validation_user": describe_numeric(per_user.positive_rate),
        "positive_rate_user_buckets": {
            "0": int((per_user.positive_rate == 0).sum()),
            "(0,0.25]": int(per_user.positive_rate.gt(0).mul(per_user.positive_rate.le(.25)).sum()),
            "(0.25,0.5]": int(per_user.positive_rate.gt(.25).mul(per_user.positive_rate.le(.5)).sum()),
            "(0.5,0.75]": int(per_user.positive_rate.gt(.5).mul(per_user.positive_rate.le(.75)).sum()),
            "(0.75,1)": int(per_user.positive_rate.gt(.75).mul(per_user.positive_rate.lt(1)).sum()),
            "1": int((per_user.positive_rate == 1).sum()),
        },
        "single_impression_users": int((per_user.n == 1).sum()),
        "single_impression_users_pct": float((per_user.n == 1).mean() * 100),
        "uniform_label_composition": uniform,
        "activity_quantile_cutpoints_train_interactions": {"q25": int(q25), "q50": int(q50), "q75": int(q75)},
        "activity_buckets": activity_metrics.to_dict("records"),
        "list_length_buckets": list_metrics.to_dict("records"),
    }

    # E/F: feedback profile and train-derived historical availability.
    feedback_rows = []
    for split_name, frame in (("train", train), ("validation", valid)):
        for signal in SIGNALS:
            row = {"split": split_name, "signal": signal}
            if signal in BINARY_SIGNALS:
                prevalence = float(frame[signal].mean())
                row.update({
                    "kind": "binary", "mean_or_prevalence": prevalence,
                    "zero_pct": float((frame[signal] == 0).mean() * 100),
                    "long_view_rate_signal_zero": float(frame.loc[frame[signal] == 0, "long_view"].mean()),
                    "long_view_rate_signal_positive": float(frame.loc[frame[signal] != 0, "long_view"].mean()) if (frame[signal] != 0).any() else np.nan,
                    "pearson_with_long_view": float(frame[[signal, "long_view"]].corr().iloc[0, 1]),
                })
            else:
                desc = describe_numeric(frame[signal])
                row.update({
                    "kind": "continuous", "mean_or_prevalence": desc["mean"],
                    "zero_pct": desc["zero_pct"], "median": desc["median"],
                    "p90": desc["p90"], "p99": desc["p99"], "max": desc["max"],
                    "mean_given_long_view_zero": float(frame.loc[frame.long_view == 0, signal].mean()),
                    "mean_given_long_view_one": float(frame.loc[frame.long_view == 1, signal].mean()),
                    "pearson_with_long_view": float(frame[[signal, "long_view"]].corr().iloc[0, 1]),
                    "log1p_pearson_with_long_view": float(np.corrcoef(np.log1p(frame[signal]), frame.long_view)[0, 1]),
                })
            feedback_rows.append(row)
    feedback_df = pd.DataFrame(feedback_rows)
    feedback_df.to_csv(RESULTS / "feedback_profile.csv", index=False)

    corr_frame = pd.concat([train[SIGNALS + ["long_view"]], valid[SIGNALS + ["long_view"]]], ignore_index=True)
    corr_transform = corr_frame.copy()
    for signal in CONTINUOUS_SIGNALS:
        corr_transform[signal] = np.log1p(corr_transform[signal])
    feedback_corr = corr_transform.corr(method="pearson")
    feedback_corr.to_csv(RESULTS / "feedback_correlation_log_continuous.csv")

    user_history = train.groupby("user_id").agg(
        prior_interactions=("video_id", "size"), prior_clicks=("is_click", "sum"),
        prior_likes=("is_like", "sum"), prior_follows=("is_follow", "sum"),
        prior_comments=("is_comment", "sum"), prior_forwards=("is_forward", "sum"),
        prior_hates=("is_hate", "sum"), prior_play_time_rows=("play_time_ms", lambda s: int((s > 0).sum())),
        prior_play_time_ms=("play_time_ms", "sum"),
    )
    history_users = per_user[["user_id", "activity_tier"]].join(user_history, on="user_id").fillna(0)
    history_columns = [c for c in user_history.columns]
    for column in history_columns:
        history_users[column] = history_users[column].astype(np.int64)

    train_uv_key = set((int(u) << 16) + int(v) for u, v in zip(train.user_id, train.video_id))
    valid_repeat_video = np.fromiter(
        (((int(u) << 16) + int(v)) in train_uv_key for u, v in zip(valid.user_id, valid.video_id)),
        dtype=bool, count=len(valid),
    )
    valid_repeat_author = np.fromiter(
        ((int(u), a) in train_ua for u, a in zip(valid.user_id, valid.author_id)),
        dtype=bool, count=len(valid),
    )
    valid_row_tier = valid.user_id.map(per_user.set_index("user_id").activity_tier)
    history_by_tier = []
    history_signal_by_tier = []
    for tier in ["Cold", "T1", "T2", "T3", "T4"]:
        hu = history_users[history_users.activity_tier == tier]
        rm = valid_row_tier.eq(tier).to_numpy()
        history_by_tier.append({
            "tier": tier, "validation_users": int(len(hu)),
            "median_prior_interactions": float(hu.prior_interactions.median()) if len(hu) else np.nan,
            "users_ge_1_prior_pct": float((hu.prior_interactions >= 1).mean() * 100) if len(hu) else np.nan,
            "users_ge_5_prior_pct": float((hu.prior_interactions >= 5).mean() * 100) if len(hu) else np.nan,
            "users_ge_10_prior_pct": float((hu.prior_interactions >= 10).mean() * 100) if len(hu) else np.nan,
            "validation_rows_prior_same_video_pct": float(valid_repeat_video[rm].mean() * 100) if rm.any() else np.nan,
            "validation_rows_prior_same_author_pct": float(valid_repeat_author[rm].mean() * 100) if rm.any() else np.nan,
            "validation_rows_prior_same_tag_pct": float(valid_row_any_prior_tag[rm].mean() * 100) if rm.any() else np.nan,
        })
        for column in history_columns:
            history_signal_by_tier.append({
                "tier": tier, "signal": column, "validation_users": int(len(hu)),
                "mean": float(hu[column].mean()) if len(hu) else np.nan,
                "median": float(hu[column].median()) if len(hu) else np.nan,
                "users_ge_1_pct": float((hu[column] >= 1).mean() * 100) if len(hu) else np.nan,
                "users_ge_5_pct": float((hu[column] >= 5).mean() * 100) if len(hu) else np.nan,
                "users_ge_10_pct": float((hu[column] >= 10).mean() * 100) if len(hu) else np.nan,
            })
    pd.DataFrame(history_by_tier).to_csv(RESULTS / "history_by_activity_tier.csv", index=False)
    pd.DataFrame(history_signal_by_tier).to_csv(RESULTS / "history_signal_by_activity_tier.csv", index=False)
    history_signal_coverage = {}
    for column in history_columns:
        history_signal_coverage[column] = {
            threshold: float((history_users[column] >= int(threshold)).mean() * 100)
            for threshold in ("1", "5", "10")
        }
    history = {
        "validation_user_prior_interactions": describe_numeric(history_users.prior_interactions),
        "validation_user_signal_threshold_pct": history_signal_coverage,
        "validation_rows_prior_same_video_pct": float(valid_repeat_video.mean() * 100),
        "validation_rows_prior_same_author_pct": float(valid_repeat_author.mean() * 100),
        "validation_rows_prior_same_tag_pct": float(valid_row_any_prior_tag.mean() * 100),
        "by_activity_tier": history_by_tier,
        "signal_availability_by_activity_tier": history_signal_by_tier,
    }

    feedback_by_tier_rows = []
    train_tier_map = train_activity.map(activity_tier)
    for split_name, frame, tier_series in (
        ("train", train, train.user_id.map(train_tier_map)),
        ("validation", valid, valid_row_tier),
    ):
        for tier in ["Cold", "T1", "T2", "T3", "T4"]:
            part = frame[tier_series.eq(tier).to_numpy()]
            if part.empty: continue
            for signal in SIGNALS:
                feedback_by_tier_rows.append({
                    "split": split_name, "tier": tier, "signal": signal,
                    "rows": int(len(part)), "mean_or_prevalence": float(part[signal].mean()),
                    "zero_pct": float((part[signal] == 0).mean() * 100),
                })
    pd.DataFrame(feedback_by_tier_rows).to_csv(RESULTS / "feedback_by_activity_tier.csv", index=False)

    # G: video basic/statistical inventory and a fixed set of smoothed ratios.
    inventory_rows = []
    for source_name, frame in (("video_basic", observed_basic), ("video_statistics", stats[stats.video_id.isin(train_videos | valid_videos)])):
        for column in frame.columns:
            s = frame[column]
            row = {
                "source": source_name, "field": column, "rows": int(len(frame)),
                "missing_pct": float(s.isna().mean() * 100),
                "cardinality": int(s.nunique(dropna=True)), "dtype": str(s.dtype),
            }
            if pd.api.types.is_numeric_dtype(s):
                nonmissing = s.dropna()
                row.update({
                    "min": float(nonmissing.min()) if len(nonmissing) else np.nan,
                    "median": float(nonmissing.median()) if len(nonmissing) else np.nan,
                    "mean": float(nonmissing.mean()) if len(nonmissing) else np.nan,
                    "p99": float(nonmissing.quantile(.99)) if len(nonmissing) else np.nan,
                    "max": float(nonmissing.max()) if len(nonmissing) else np.nan,
                })
            inventory_rows.append(row)
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(RESULTS / "video_feature_inventory.csv", index=False)

    duration_check = pd.concat([
        train[["video_id", "duration_ms"]], valid[["video_id", "duration_ms"]]
    ], ignore_index=True).merge(
        basic[["video_id", "video_duration"]], on="video_id", how="left", sort=False
    )
    duration_diff = (duration_check.duration_ms - duration_check.video_duration).abs()
    basic_redundancy = {
        "visible_status_cardinality": int(basic.visible_status.nunique(dropna=True)),
        "tag_combination_cardinality": int(basic.tag.nunique(dropna=True)),
        "tag_token_cardinality": int(len({tag for tags in video_tags.values() for tag in tags})),
        "rows_with_video_duration_pct": float(duration_check.video_duration.notna().mean() * 100),
        "interaction_duration_exactly_matches_basic_pct_of_nonmissing": float(
            duration_diff[duration_check.video_duration.notna()].eq(0).mean() * 100
        ),
        "interaction_vs_basic_duration_median_absolute_difference_ms": float(duration_diff.median()),
        "interaction_vs_basic_duration_spearman": float(
            duration_check[["duration_ms", "video_duration"]].corr(method="spearman").iloc[0, 1]
        ),
    }

    numeric_stats = stats.select_dtypes(include=[np.number]).drop(columns=["video_id"])
    stat_corr = numeric_stats.corr(method="spearman")
    redundant_pairs = []
    columns = list(stat_corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = float(stat_corr.loc[left, right])
            redundant_pairs.append({"left": left, "right": right, "spearman": value, "absolute_spearman": abs(value)})
    stat_redundancy_df = pd.DataFrame(redundant_pairs).sort_values("absolute_spearman", ascending=False)
    stat_redundancy_df.head(100).to_csv(RESULTS / "video_stat_top_redundancies.csv", index=False)

    ratio_specs = {
        "play_per_show": "play_cnt", "complete_play_per_show": "complete_play_cnt",
        "valid_play_per_show": "valid_play_cnt", "long_play_per_show": "long_time_play_cnt",
        "short_play_per_show": "short_time_play_cnt", "like_per_show": "like_cnt",
        "comment_per_show": "comment_cnt", "follow_per_show": "follow_cnt",
        "share_per_show": "share_cnt",
    }
    stat_work = stats.copy()
    ratio_rows = []
    beta = 20.0
    train_item = train.groupby("video_id").long_view.agg(["sum", "count"])
    global_lv = float(train.long_view.mean())
    train_item["train_item_long_view_rate_smoothed"] = (train_item["sum"] + 20 * global_lv) / (train_item["count"] + 20)
    valid_join = valid[["user_id", "video_id", "long_view"]].merge(stat_work, on="video_id", how="left", sort=False)
    valid_join = valid_join.merge(train_item[["train_item_long_view_rate_smoothed"]], on="video_id", how="left", sort=False)
    item_scores = valid_join.train_item_long_view_rate_smoothed.fillna(global_lv).to_numpy()
    item_metrics = score_static(valid_join, item_scores)
    for ratio_name, numerator in ratio_specs.items():
        global_ratio = float(stat_work[numerator].sum() / stat_work.show_cnt.sum())
        stat_work[ratio_name] = (stat_work[numerator] + beta * global_ratio) / (stat_work.show_cnt + beta)
        ratio_map = stat_work.set_index("video_id")[ratio_name]
        vals = valid_join.video_id.map(ratio_map).astype(float)
        fill = float(stat_work[ratio_name].median())
        vals = vals.fillna(fill)
        metrics = score_static(valid_join, vals)
        aligned = pd.DataFrame({"video_id": stat_work.video_id, "ratio": stat_work[ratio_name]}).merge(
            train_item[["train_item_long_view_rate_smoothed"]], on="video_id", how="inner"
        )
        ratio_rows.append({
            "ratio": ratio_name, "numerator": numerator, "denominator": "show_cnt",
            "smoothing_beta": beta, "smoothing_prior_rate": global_ratio,
            "validation_video_coverage_pct": float(valid_join.video_id.isin(set(stat_work.video_id)).mean() * 100),
            "median": float(stat_work[ratio_name].median()), "p99": float(stat_work[ratio_name].quantile(.99)),
            "spearman_with_train_item_long_view_rate": float(aligned[["ratio", "train_item_long_view_rate_smoothed"]].corr(method="spearman").iloc[0, 1]),
            "validation_GAUC": metrics["GAUC"], "validation_nDCG@5": metrics["nDCG@5"],
            "validation_primary": metrics["primary"],
            "delta_primary_vs_train_item_pop": metrics["primary"] - item_metrics["primary"],
        })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(RESULTS / "video_stat_ratio_diagnostics.csv", index=False)
    video_features = {
        "basic_rows": int(len(basic)), "statistic_rows": int(len(stats)),
        "train_valid_video_basic_coverage_pct": float(pd.Index(train_videos | valid_videos).isin(basic.video_id).mean() * 100),
        "train_valid_video_stat_coverage_pct": float(pd.Index(train_videos | valid_videos).isin(stats.video_id).mean() * 100),
        "validation_train_item_pop_metrics": item_metrics,
        "ratio_diagnostics": ratio_rows,
        "basic_redundancy": basic_redundancy,
        "statistic_pairs_absolute_spearman_ge_0_95": int((stat_redundancy_df.absolute_spearman >= .95).sum()),
        "top_statistic_redundancies": stat_redundancy_df.head(15).to_dict("records"),
        "aggregation_window_documented_in_local_source": False,
        "causal_validity": "unclear from local source materials",
    }

    # H: temporal profile and empirical early/late comparisons.
    standard = pd.concat([train, valid], ignore_index=True)
    daily = standard.groupby("date").agg(
        rows=("user_id", "size"), long_view_rate=("long_view", "mean"),
        unique_users=("user_id", "nunique"), unique_videos=("video_id", "nunique"),
        mean_duration_ms=("duration_ms", "mean"), median_duration_ms=("duration_ms", "median"),
        mean_tab=("tab", "mean"),
    ).reset_index()
    daily.to_csv(RESULTS / "daily_standard_profile.csv", index=False)
    early = train[(train.date >= 20220408) & (train.date <= 20220414)]
    late = train[(train.date >= 20220415) & (train.date <= 20220421)]
    period_rows = [period_summary(early, "early_train"), period_summary(late, "late_train"), period_summary(valid, "validation")]
    pd.DataFrame(period_rows).to_csv(RESULTS / "temporal_period_summary.csv", index=False)
    temporal_comparison = {
        "validation_vs_early": {
            "absolute_long_view_rate_difference": abs(float(valid.long_view.mean() - early.long_view.mean())),
            "absolute_mean_duration_difference_ms": abs(float(valid.duration_ms.mean() - early.duration_ms.mean())),
            "tab_js_divergence_bits": js_for_categories(valid.tab, early.tab),
        },
        "validation_vs_late": {
            "absolute_long_view_rate_difference": abs(float(valid.long_view.mean() - late.long_view.mean())),
            "absolute_mean_duration_difference_ms": abs(float(valid.duration_ms.mean() - late.duration_ms.mean())),
            "tab_js_divergence_bits": js_for_categories(valid.tab, late.tab),
        },
    }

    # I: random-exposure identifiers/dates only; never load its labels.
    random_start = time.perf_counter()
    random_ids = pd.read_csv(
        args.data_dir / "log_random_4_22_to_5_08_pure.csv",
        usecols=["user_id", "video_id", "date"],
        dtype={"user_id": "int32", "video_id": "int32", "date": "int32"},
    )
    random_load_seconds = time.perf_counter() - random_start
    random_users, random_videos = set(random_ids.user_id), set(random_ids.video_id)
    random_pairs = set(zip(random_ids.user_id.astype(int), random_ids.video_id.astype(int)))
    standard_pairs = train_uv | valid_uv
    eval_period_mask = random_ids.date.between(20220429, 20220508)
    random_audit = {
        "labels_loaded": False,
        "rows": int(len(random_ids)), "date_min": int(random_ids.date.min()),
        "date_max": int(random_ids.date.max()), "dates": {str(k): int(v) for k, v in random_ids.date.value_counts().sort_index().items()},
        "evaluation_period_rows": int(eval_period_mask.sum()),
        "evaluation_period_rows_pct": float(eval_period_mask.mean() * 100),
        "unique_users": len(random_users), "unique_videos": len(random_videos),
        "unique_pairs": len(random_pairs),
        "users_overlap_train_validation_count": len(random_users & (train_users | valid_users)),
        "users_overlap_train_validation_pct_of_random": 100 * len(random_users & (train_users | valid_users)) / len(random_users),
        "videos_overlap_train_validation_count": len(random_videos & (train_videos | valid_videos)),
        "videos_overlap_train_validation_pct_of_random": 100 * len(random_videos & (train_videos | valid_videos)) / len(random_videos),
        "pairs_overlap_train_validation_count": len(random_pairs & standard_pairs),
        "pairs_overlap_train_validation_pct_of_random": 100 * len(random_pairs & standard_pairs) / len(random_pairs),
        "load_seconds_identifiers_dates_only": random_load_seconds,
    }

    # Plots are summaries of train/validation only.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(daily.date.astype(str), daily.rows, marker="o")
    axes[0].set_ylabel("Rows")
    axes[1].plot(daily.date.astype(str), daily.long_view_rate, marker="o")
    axes[1].set_ylabel("Long-view rate")
    axes[1].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(PLOTS / "daily_standard_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    list_plot = list_metrics.set_index("bucket")
    ax.bar(list_plot.index, list_plot.users)
    ax.set_xlabel("Validation impressions per user")
    ax.set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(PLOTS / "validation_list_length_users.png", dpi=150)
    plt.close(fig)

    profile_seconds = time.perf_counter() - started
    result = {
        "guardrails": {
            "test_labels_accessed": False,
            "test_feature_rows_used_for_analysis": False,
            "official_source_modified": False,
            "validation_end": VALID_END,
            "random_log_columns_loaded": ["user_id", "video_id", "date"],
        },
        "cardinalities": cardinalities,
        "overlap": overlap,
        "repeat_structure": repeat,
        "author_video_structure": author_structure,
        "feature_coverage": feature_coverage,
        "metric_structure": metric_structure,
        "feedback_profile": feedback_df.to_dict("records"),
        "history": history,
        "video_features": video_features,
        "temporal": {"periods": period_rows, "comparison": temporal_comparison},
        "random_exposure": random_audit,
        "engineering": {
            "load_timings_seconds": load_timings,
            "feature_files_load_seconds_approx": feature_load_seconds,
            "profile_total_seconds": profile_seconds,
            "rss_after_log_load_bytes": rss_after_logs,
            "rss_peak_current_bytes": process.memory_info().rss,
            "environment": {
                "os": platform.platform(), "python": sys.version,
                "python_executable": sys.executable, "numpy": np.__version__,
                "pandas": pd.__version__, "cpu_count": os.cpu_count(),
                "physical_memory_bytes": psutil.virtual_memory().total,
            },
        },
    }
    args.output.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({
        "cardinalities": cardinalities, "overlap": overlap,
        "metric_structure": {k: v for k, v in metric_structure.items() if k not in ("activity_buckets", "list_length_buckets")},
        "history": history, "random_exposure": random_audit,
        "profile_total_seconds": profile_seconds,
    }, indent=2, default=json_default))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--predictions", type=Path, default=RESULTS / "baseline_validation_predictions.npz")
    parser.add_argument("--output", type=Path, default=RESULTS / "data_profile_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
