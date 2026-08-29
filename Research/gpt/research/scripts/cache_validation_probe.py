"""Create and verify a deterministic research-only train/validation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd

import profile_train_validation as P


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "source" / "KuaiRand-Pure" / "data"
CACHE_DIR = ROOT / "research" / "experiment_results" / "cache"
OUTPUT = ROOT / "research" / "experiment_results" / "cache_probe.json"
CACHE_VERSION = "train-valid-raw-v1"


def included_source_hash(data_dir: Path):
    """Hash only header + rows belonging to train or validation."""
    digest = hashlib.sha256()
    included_rows = 0
    for name in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        path = data_dir / name
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            header = fh.readline()
            digest.update(name.encode())
            digest.update(header.encode())
            for line in fh:
                first = line.split(",", 3)
                if len(first) < 3:
                    raise RuntimeError(f"Malformed line in {path}")
                date = int(first[2])
                if date <= P.VALID_END:
                    digest.update(line.encode())
                    included_rows += 1
    return digest.hexdigest(), included_rows


def frame_hash(frame: pd.DataFrame):
    digest = hashlib.sha256()
    digest.update("|".join(frame.columns).encode())
    digest.update("|".join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def cache_valid(metadata, source_hash):
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("included_source_sha256") == source_hash
        and metadata.get("test_labels_accessed") is False
    )


def main(args):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    source_hash_start = time.perf_counter()
    source_hash, included_rows = included_source_hash(args.data_dir)
    hash_seconds = time.perf_counter() - source_hash_start
    if included_rows != 1_266_021:
        raise RuntimeError(f"Fingerprint included unexpected row count {included_rows}")

    train, valid, load_timings = P.load_train_valid(args.data_dir)
    original_hashes = {"train": frame_hash(train), "valid": frame_hash(valid)}
    cache_path = args.cache_dir / "train_validation_raw.pkl"
    meta_path = args.cache_dir / "train_validation_raw.meta.json"
    write_start = time.perf_counter()
    with cache_path.open("wb") as fh:
        pickle.dump({"train": train, "valid": valid}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    metadata = {
        "cache_version": CACHE_VERSION,
        "included_source_sha256": source_hash,
        "included_rows": included_rows,
        "test_labels_accessed": False,
        "rows": {"train": len(train), "valid": len(valid)},
        "frame_sha256": original_hashes,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    read_start = time.perf_counter()
    with cache_path.open("rb") as fh:
        cached = pickle.load(fh)
    read_seconds = time.perf_counter() - read_start
    cached_hashes = {key: frame_hash(value) for key, value in cached.items()}
    exact_match = original_hashes == cached_hashes
    current_valid = cache_valid(json.loads(meta_path.read_text(encoding="utf-8")), source_hash)
    altered_metadata = dict(metadata)
    altered_metadata["included_source_sha256"] = "deliberately-wrong"
    invalidation_rejects_changed_fingerprint = not cache_valid(altered_metadata, source_hash)

    result = {
        "purpose": "deterministic cache equivalence and invalidation probe",
        "test_labels_accessed": False,
        "cache_version": CACHE_VERSION,
        "included_source_sha256": source_hash,
        "included_rows": included_rows,
        "row_feature_hashes_original": original_hashes,
        "row_feature_hashes_cached": cached_hashes,
        "identical_rows_and_features": exact_match,
        "current_cache_valid": current_valid,
        "changed_fingerprint_rejected": invalidation_rejects_changed_fingerprint,
        "timings_seconds": {
            "source_content_fingerprint": hash_seconds,
            "raw_csv_load": sum(load_timings.values()),
            "cache_write": write_seconds,
            "cache_read": read_seconds,
        },
        "cache_bytes": cache_path.stat().st_size,
    }
    if not all((exact_match, current_valid, invalidation_rejects_changed_fingerprint)):
        raise RuntimeError(f"Cache verification failed: {result}")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
