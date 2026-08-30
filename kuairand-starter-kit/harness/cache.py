"""Deterministic cache of the parsed splits and the encoded arrays.

The measured cold cost of one node is ~57 s, of which ~8 s is CSV parsing and encoding
that never changes between iterations. Caching those two steps takes a warm node to
~50 s and, on the repo's own probe, beats recompute by ~262x on the load itself.

Two levels:
  raw   -- kit.data.load() output, keyed by the content hash of the source CSVs
  enc   -- pipeline.features.encode() output, keyed additionally by a hash of
           features.py, so any agent edit to the feature builder invalidates it

The encoded cache is the one that matters; it is also the one that MUST invalidate
when the agent edits features.py, or an iteration silently scores its parent's model.
"""
import hashlib
import os
import pickle
import time

import numpy as np

from . import DATA_DIR, PIPELINE, ROOT, SPLIT_SIZES

CACHE_DIR = os.path.join(ROOT, '.cache')
RAW_PATH = os.path.join(CACHE_DIR, 'splits_raw.pkl')
ENC_PATH = os.path.join(CACHE_DIR, 'encoded.npz')
_SOURCE_CSVS = ('log_standard_4_08_to_4_21_pure.csv',
                'log_standard_4_22_to_5_08_pure.csv',
                'video_features_basic_pure.csv')


def _file_sig(path):
    """Cheap, stable signature: size + mtime. Hashing 197 MB per node is not free."""
    st = os.stat(path)
    return '%d:%d' % (st.st_size, int(st.st_mtime))


def _source_hash(data_dir=DATA_DIR):
    h = hashlib.sha256()
    for name in _SOURCE_CSVS:
        h.update(name.encode())
        h.update(_file_sig(os.path.join(data_dir, name)).encode())
    return h.hexdigest()


def _features_hash():
    """Content hash of the agent's feature builder. Changes => encoded cache is stale."""
    with open(os.path.join(PIPELINE, 'features.py'), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _assert_sizes(splits):
    got = {k: len(v) for k, v in splits.items()}
    if got != SPLIT_SIZES:
        raise AssertionError('split sizes %r != official %r' % (got, SPLIT_SIZES))


def load_splits(data_dir=DATA_DIR, use_cache=True):
    """kit.data.load(), cached. Returns {'train': [...], 'valid': [...], 'test': [...]}."""
    sig = _source_hash(data_dir)
    if use_cache and os.path.exists(RAW_PATH):
        try:
            with open(RAW_PATH, 'rb') as fh:
                blob = pickle.load(fh)
            if blob.get('source_hash') == sig:
                _assert_sizes(blob['splits'])
                return blob['splits']
        except Exception:
            pass                                    # corrupt cache is not fatal, rebuild
    from data import load                            # frozen kit loader
    splits = load(data_dir)
    _assert_sizes(splits)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(RAW_PATH, 'wb') as fh:
            pickle.dump({'source_hash': sig, 'splits': splits}, fh, protocol=4)
    return splits


def load_encoded(data_dir=DATA_DIR, use_cache=True):
    """pipeline.features.encode(), cached against features.py's content hash.

    Returns (enc, dim) where enc[split] = (X, y, users) exactly as features.encode does.
    """
    fsig = _features_hash()
    ssig = _source_hash(data_dir)
    if use_cache and os.path.exists(ENC_PATH):
        try:
            z = np.load(ENC_PATH, allow_pickle=False)
            if str(z['features_hash']) == fsig and str(z['source_hash']) == ssig:
                enc = {}
                for sp in SPLIT_SIZES:
                    enc[sp] = (z['%s_X' % sp], z['%s_y' % sp], list(z['%s_u' % sp]))
                return enc, int(z['dim'])
        except Exception:
            pass
    splits = load_splits(data_dir, use_cache=use_cache)
    from features import encode                      # the agent's editable builder
    enc, dim = encode(splits)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        payload = {'dim': np.int64(dim),
                   'features_hash': np.str_(fsig),
                   'source_hash': np.str_(ssig)}
        for sp, (X, y, u) in enc.items():
            payload['%s_X' % sp] = X
            payload['%s_y' % sp] = y
            payload['%s_u' % sp] = np.asarray(u, dtype=np.str_)
        np.savez(ENC_PATH, **payload)
    return enc, dim


def warm(data_dir=DATA_DIR):
    """Build both caches and report timings. Run once before the measured run."""
    t0 = time.time()
    splits = load_splits(data_dir)
    t1 = time.time()
    enc, dim = load_encoded(data_dir)
    t2 = time.time()
    return {'rows': {k: len(v) for k, v in splits.items()}, 'dim': dim,
            'load_s': round(t1 - t0, 2), 'encode_s': round(t2 - t1, 2)}


if __name__ == '__main__':
    import json
    print(json.dumps(warm(), indent=2))
