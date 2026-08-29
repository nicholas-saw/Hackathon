"""Research-only FM training utilities built on TRAIN+VALID ONLY.

Deliberately reimplements the encode step against common.py's loaders
(which never materialize test-range rows) instead of importing data.load(),
so that no code path in the pre-audit can accidentally touch test labels.
The FM model class itself (pure numpy, no data access) is imported from the
official starter-kit baseline.py -- that is model code, not data code.
"""
import os, sys, time
import numpy as np

STARTER_KIT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'starter-kit'))
sys.path.insert(0, STARTER_KIT)
import baseline as B          # noqa: E402  (official FM class + sigmoid, no data access)
from evaluate import evaluate  # noqa: E402  (official metric, called on VALID only here)


def build_rows(df, video_basic_map, extra_cols=None):
    """Return list of column-arrays needed for field construction."""
    extra_cols = extra_cols or []
    cols = {}
    cols['user_id'] = df['user_id'].astype(str).to_numpy()
    cols['video_id'] = df['video_id'].astype(str).to_numpy()
    cols['author_id'] = df['video_id'].map(video_basic_map).astype(str).to_numpy()
    cols['tab'] = df['tab'].astype(str).to_numpy()
    cols['duration_ms'] = df['duration_ms'].to_numpy()
    cols['long_view'] = df['long_view'].to_numpy()
    for c in extra_cols:
        cols[c] = df[c].astype(str).to_numpy()
    return cols


def encode_fields(train_df, valid_df, video_basic, fields, extra_train_maps=None):
    """fields: list of logical field names, each either a plain column name
    present in the log after merge, or 'dur_bucket' (derived). extra_train_maps
    lets you pass per-video/per-user side-table columns (dict: field_name -> series
    indexed by video_id or user_id) to be merged in for both splits."""
    extra_train_maps = extra_train_maps or {}
    vid2author = video_basic.set_index('video_id')['author_id']

    def prep(df):
        out = {}
        out['user_id'] = df['user_id'].astype(str).to_numpy()
        out['video_id'] = df['video_id'].astype(str).to_numpy()
        out['author_id'] = df['video_id'].map(vid2author).astype(str).to_numpy()
        out['tab'] = df['tab'].astype(str).to_numpy()
        for fname, series in extra_train_maps.items():
            if series.index.name == 'video_id' or 'video' in str(series.index.name):
                out[fname] = df['video_id'].map(series).astype(str).to_numpy()
            else:
                out[fname] = df['user_id'].map(series).astype(str).to_numpy()
        return out

    tr_cols = prep(train_df)
    va_cols = prep(valid_df)

    edges = np.quantile(train_df['duration_ms'].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    tr_cols['dur_bucket'] = np.searchsorted(edges, train_df['duration_ms'].to_numpy()).astype(str)
    va_cols['dur_bucket'] = np.searchsorted(edges, valid_df['duration_ms'].to_numpy()).astype(str)

    vocabs = [dict() for _ in fields]
    for i, f in enumerate(fields):
        for v in tr_cols[f]:
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    def to_X(cols_dict, n):
        X = np.empty((n, len(fields)), dtype=np.int32)
        for i, f in enumerate(fields):
            vals = cols_dict[f]
            X[:, i] = [vocabs[i].get(v, unk[i]) + offsets[i] for v in vals]
        return X

    Xtr = to_X(tr_cols, len(train_df))
    Xva = to_X(va_cols, len(valid_df))
    ytr = train_df['long_view'].to_numpy().astype(np.float32)
    yva = valid_df['long_view'].to_numpy().astype(np.float32)
    uva = valid_df['user_id'].to_numpy()
    dim = int(sum(field_dims))
    return Xtr, ytr, Xva, yva, uva, dim


def train_fm(Xtr, ytr, Xva, yva, uva, dim, k=16, lr=0.001, epochs=40, bs=8192,
             patience=4, seed=0, verbose=False):
    m = B.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    history = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        va = evaluate(uva, yva, m.predict(Xva))
        history.append({'epoch': ep, **va, 'time_s': time.time() - t0})
        if verbose:
            print(f"  epoch {ep:2d} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    final_scores = m.predict(Xva)
    final_metrics = evaluate(uva, yva, final_scores)
    return m, final_scores, final_metrics, history
