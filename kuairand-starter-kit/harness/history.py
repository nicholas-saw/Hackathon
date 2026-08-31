"""Causal prior-history aggregates — the one legal route from outcomes to inputs.

`pipeline/features.py` may not read `IDX['label']`, and `same_row()` refuses all eleven
post-impression columns. That is correct for *same-row* use, but it also blocked the
legal case the rules explicitly allow: an aggregate built from **strictly earlier** rows.
Without a helper the agent had no way to comply, so every history hypothesis died at the
static guard.

This module is that helper. It lives on the harness side of the boundary, so the guard in
`harness/guards.py` stays exactly as strict — `features.py` never touches a label, it
calls a function that already did the ordering correctly.

Three properties, enforced in code rather than by discipline:

1. **Strictly earlier.** A row's aggregate is computed from rows with a *smaller*
   `time_ms`. Never its own row.
2. **Ties are not predecessors.** Rows sharing a timestamp do not see each other. The
   walk emits every row in a timestamp group before updating state with any of them.
   This is not hypothetical: 5.60% of validation rows sit in non-unique user/timestamp
   groups (constraints.md C13).
3. **Test outcomes never enter state.** Test rows are excluded from every aggregate.
   They receive the full train+valid state, which is the most history that is legally
   available to them, and contribute nothing back.

The resulting feature is defined identically for train, valid and test rows — an
expanding window over time — so its distribution does not shift across the splits.

Usage from `pipeline/features.py`:

    from harness.history import prior_stats

    rate, count = prior_stats(splits, signal='label', key='user_id')
    # rate[split] and count[split] are float arrays aligned 1:1 with splits[split]

`signal` may be `'label'` or any post-impression column the adapter serves
(`is_click`, `is_like`, `play_time_ms`, …). `key` may be a single row field or a tuple
of them, e.g. `('user_id', 'author_id')` for user-author affinity.

Nothing here decides *which* aggregate is worth building. That is the agent's call.
"""
import numpy as np

from . import adapter

# Fields of the fixed-length row tuple returned by kit.data.load(). Mirrors
# pipeline/features.py's IDX; kit/data.py remains the single source of truth.
IDX = {'date': 0, 'user_id': 1, 'video_id': 2, 'author_id': 3,
       'tab': 4, 'duration_ms': 5, 'label': 6}

# Splits that may contribute to an aggregate. 'test' is absent by construction.
_STATE_SPLITS = ('train', 'valid')


def _smooth(total, n, prior_weight, global_mean):
    """Smoothed rate, with an empty denominator falling back to the global mean.

    `prior_weight=0` asks for the raw unsmoothed rate, which is undefined for a row with
    no history. Returning the global mean there keeps the feature finite; the companion
    `count` array is what tells a model the estimate rested on nothing.
    """
    denom = n + prior_weight
    if denom <= 0.0:
        return global_mean
    return (total + prior_weight * global_mean) / denom


def _key_values(rows, key):
    """Per-row key, as a hashable. `key` is a field name or a tuple of field names."""
    if isinstance(key, str):
        i = IDX[key]
        return [r[i] for r in rows]
    idxs = [IDX[k] for k in key]
    return [tuple(r[i] for i in idxs) for r in rows]


def _signal_values(splits, signal, data_dir=None):
    """Per-row signal for train and valid. Test is never read."""
    if signal == 'label':
        return {s: np.asarray([r[IDX['label']] for r in splits[s]], dtype=np.float64)
                for s in _STATE_SPLITS}
    kw = {'data_dir': data_dir} if data_dir else {}
    aux = adapter.auxiliary_targets((signal,), **kw)
    return {s: aux[s][signal].astype(np.float64) for s in _STATE_SPLITS}


def prior_stats(splits, signal='label', key='user_id', prior_weight=20.0,
                data_dir=None):
    """Smoothed rate and raw count of `signal` over strictly earlier rows sharing `key`.

    Returns (rate, count), each a dict {split_name: np.ndarray float32, aligned to
    splits[split]}. A row with no prior history gets rate = the train-split global mean
    and count = 0.

    Smoothing follows the kit's own item-popularity convention:

        rate = (prior_sum + prior_weight * global) / (prior_count + prior_weight)

    `global` is computed on train only, so no validation outcome influences a validation
    row even through a shared scalar.
    """
    kw = {'data_dir': data_dir} if data_dir else {}
    times = adapter.raw_columns(('time_ms',), dtype=np.int64, **kw)
    sig = _signal_values(splits, signal, data_dir)
    global_mean = float(sig['train'].mean())

    # (time, split, row_index) for every row allowed to carry state.
    order = []
    for s in _STATE_SPLITS:
        t = times[s]['time_ms']
        if len(t) != len(splits[s]):
            raise AssertionError('adapter/kit row-count mismatch for %r: %d vs %d'
                                 % (s, len(t), len(splits[s])))
        order.append(np.stack([t, np.full(len(t), _STATE_SPLITS.index(s)),
                               np.arange(len(t))], axis=1))
    order = np.concatenate(order, axis=0)
    order = order[np.argsort(order[:, 0], kind='stable')]

    keys = {s: _key_values(splits[s], key) for s in splits}

    rate = {s: np.empty(len(splits[s]), dtype=np.float32) for s in splits}
    count = {s: np.empty(len(splits[s]), dtype=np.float32) for s in splits}

    sums, cnts = {}, {}
    n = len(order)
    i = 0
    while i < n:
        j = i
        t0 = order[i, 0]
        while j + 1 < n and order[j + 1, 0] == t0:
            j += 1
        # Emit for the whole timestamp group BEFORE any of it updates state, so rows
        # sharing a timestamp are not predecessors of one another.
        for m in range(i, j + 1):
            s = _STATE_SPLITS[order[m, 1]]
            r = int(order[m, 2])
            k = keys[s][r]
            c = cnts.get(k, 0.0)
            rate[s][r] = _smooth(sums.get(k, 0.0), c, prior_weight, global_mean)
            count[s][r] = c
        for m in range(i, j + 1):
            s = _STATE_SPLITS[order[m, 1]]
            r = int(order[m, 2])
            k = keys[s][r]
            sums[k] = sums.get(k, 0.0) + float(sig[s][r])
            cnts[k] = cnts.get(k, 0.0) + 1.0
        i = j + 1

    # Test rows: the full train+valid state, contributing nothing back.
    if 'test' in splits:
        kt = keys['test']
        for r in range(len(splits['test'])):
            k = kt[r]
            c = cnts.get(k, 0.0)
            rate['test'][r] = _smooth(sums.get(k, 0.0), c, prior_weight, global_mean)
            count['test'][r] = c

    return rate, count


def prior_rate(splits, signal='label', key='user_id', prior_weight=20.0, data_dir=None):
    """`prior_stats(...)[0]` — the smoothed rate alone."""
    return prior_stats(splits, signal=signal, key=key,
                       prior_weight=prior_weight, data_dir=data_dir)[0]


def prior_count(splits, key='user_id', data_dir=None):
    """Number of strictly earlier rows sharing `key`. Uses no outcome column at all."""
    return prior_stats(splits, signal='label', key=key, data_dir=data_dir)[1]


def bucketize(values, edges):
    """Map a float aggregate onto integer bucket ids, for use as a categorical field.

    `edges` must come from the TRAIN split only — fitting them on valid or test would
    leak the scored period's distribution into the encoding.
    """
    return np.searchsorted(np.asarray(edges), np.asarray(values)).astype(np.int32)
