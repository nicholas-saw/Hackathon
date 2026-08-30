"""Cheap, whitelisted analyses the proposer can request BEFORE spending an iteration.

This is the highest-leverage thing in the loop. A node costs ~60 s and, more importantly,
one of the ~10 iterations the convergence rule allows. Most of these answer in under
three seconds, so the agent can test a hypothesis's *premise* before committing an
experiment to it — "does this column carry any within-user signal at all?" is answerable
without training anything.

Every analysis reads train and validation only, through harness.adapter, which cannot
return test rows. The proposer names one by key; it cannot run arbitrary code.
"""
import numpy as np

from . import DATA_DIR
from .adapter import entity_table, raw_columns
from .cache import load_encoded
from .score import evaluate_raw

_CACHE = {}


def _enc():
    if 'enc' not in _CACHE:
        _CACHE['enc'] = load_encoded()
    return _CACHE['enc']


def _valid_users_labels():
    enc, _ = _enc()
    _, y, u = enc['valid']
    return np.asarray(u), np.asarray(y)


def _groups(users):
    order = np.argsort(users, kind='stable')
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(us)])
    return order, starts, counts


# ---------------------------------------------------------------- analyses

def no_op_screen(column=None, split='valid'):
    """Is a candidate feature constant within each user? Then it is provably useless.

    Ranking is within-user, so any term that does not vary inside a user's impression
    list cannot change that list's order. This is the cheapest possible rejection and it
    kills a whole class of plausible-sounding ideas in about a second.
    """
    if column is None:
        return {'error': 'pass column=<log column name>'}
    users, _ = _valid_users_labels()
    vals = raw_columns((column,), splits=('valid',))['valid'][column]
    order, starts, counts = _groups(users)
    v = vals[order]
    varying = 0
    multi = 0
    for s, c in zip(starts, counts):
        if c < 2:
            continue
        multi += 1
        if len(set(v[s:s + c].tolist())) > 1:
            varying += 1
    frac = varying / max(multi, 1)
    return {'column': column, 'users_with_2plus_impressions': int(multi),
            'users_where_it_varies': int(varying),
            'fraction_varying': round(frac, 4),
            'verdict': ('NO-OP: constant within every user, cannot change any ranking'
                        if frac == 0 else
                        'varies within %.1f%% of users, so it can move the metric' % (100 * frac))}


def standalone_rank(column=None, negate=False):
    """Score a raw column directly as a ranker on validation. No training.

    Tells you whether a signal is there at all before you spend a node building a model
    around it. Compare against random 0.4834 and the FM baseline 0.6015.
    """
    if column is None:
        return {'error': 'pass column=<log column name>'}
    users, y = _valid_users_labels()
    raw = raw_columns((column,), splits=('valid',))['valid'][column]
    try:
        vals = raw.astype(float)
    except ValueError:
        _, inv = np.unique(raw, return_inverse=True)
        vals = inv.astype(float)
    if negate:
        vals = -vals
    r = evaluate_raw(users, y, vals)
    return {'column': column, 'negated': bool(negate),
            'GAUC': round(float(r['GAUC']), 5),
            'nDCG@5': round(float(r['nDCG@5']), 5),
            'primary': round(float(r['primary']), 5),
            'reference': {'random': 0.4834, 'item_popularity': 0.5807, 'fm_baseline': 0.6015}}


def label_rate_by(column=None, bins=10):
    """P(long_view | column) on TRAIN. Is there a usable association?"""
    if column is None:
        return {'error': 'pass column=<log column name>'}
    cols = raw_columns((column, 'long_view'), splits=('train',))['train']
    y = (cols['long_view'] != '0').astype(float)
    raw = cols[column]
    try:
        v = raw.astype(float)
        numeric = True
    except ValueError:
        numeric = False
    out = {'column': column, 'overall_rate': round(float(y.mean()), 4)}
    if numeric and len(np.unique(v)) > bins:
        edges = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
        b = np.searchsorted(edges, v)
        out['buckets'] = [{'bucket': int(i), 'n': int((b == i).sum()),
                           'rate': round(float(y[b == i].mean()), 4)}
                          for i in range(bins) if (b == i).any()]
    else:
        vals, inv = np.unique(raw, return_inverse=True)
        out['levels'] = [{'value': str(vals[i]), 'n': int((inv == i).sum()),
                          'rate': round(float(y[inv == i].mean()), 4)}
                         for i in range(min(len(vals), 20))]
    return out


def history_coverage(kind='video'):
    """What fraction of validation rows have this (user, X) pair present in train?

    Answers the premise behind every history / sequence / attention idea. If the
    candidate item never appears in the user's history, target attention has nothing to
    attend to.
    """
    enc, _ = _enc()
    cols_t = raw_columns(('user_id', 'video_id'), splits=('train',))['train']
    cols_v = raw_columns(('user_id', 'video_id'), splits=('valid',))['valid']
    if kind == 'author':
        vb = entity_table('video_basic')
        a_t = np.array([vb.get(v, {}).get('author_id', 'UNK') for v in cols_t['video_id']])
        a_v = np.array([vb.get(v, {}).get('author_id', 'UNK') for v in cols_v['video_id']])
        seen = set(zip(cols_t['user_id'].tolist(), a_t.tolist()))
        hit = sum(1 for p in zip(cols_v['user_id'].tolist(), a_v.tolist()) if p in seen)
    else:
        seen = set(zip(cols_t['user_id'].tolist(), cols_t['video_id'].tolist()))
        hit = sum(1 for p in zip(cols_v['user_id'].tolist(), cols_v['video_id'].tolist())
                  if p in seen)
    n = len(cols_v['user_id'])
    return {'pair': 'user x %s' % kind, 'validation_rows': int(n),
            'rows_with_prior_pair': int(hit),
            'coverage_pct': round(100.0 * hit / n, 3),
            'note': 'a history feature can only act on the covered fraction'}


def headroom_by_list_length():
    """Where the movable metric actually lives, by validation list length.

    Users whose labels are uniform are invariant: nDCG is pinned and GAUC excludes them.
    This says which segments a change could possibly move.
    """
    users, y = _valid_users_labels()
    order, starts, counts = _groups(users)
    ys = y[order]
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 10 ** 6)]
    out = []
    for lo, hi in buckets:
        nu = nrows = inv = 0
        for s, c in zip(starts, counts):
            if lo <= c <= hi:
                nu += 1
                nrows += int(c)
                seg = ys[s:s + c]
                if seg.min() == seg.max():
                    inv += 1
        out.append({'list_length': '%d-%d' % (lo, hi) if hi < 10 ** 6 else '21+',
                    'users': nu, 'rows': nrows,
                    'invariant_users_pct': round(100.0 * inv / max(nu, 1), 1)})
    return {'buckets': out,
            'note': ('invariant = all-0 or all-1 labels; no model change reaches them. '
                     '42.2% of all validation users are invariant.')}


def column_inventory():
    """What is reachable, and how. Answers "what could I even use?"."""
    from .adapter import LOG_COLUMNS
    from . import LEAKY_COLUMNS
    return {
        'in_kit_data_load': ['date', 'user_id', 'video_id', 'author_id', 'tab',
                             'duration_ms', 'label'],
        'via_adapter_raw_columns': [c for c in LOG_COLUMNS],
        'post_impression_never_same_row_input': sorted(LEAKY_COLUMNS),
        'side_tables': {'user': 30, 'video_basic': 12, 'video_stat': 52},
        'access': 'harness.adapter.raw_columns() / auxiliary_targets() / entity_table()',
    }


REGISTRY = {
    'no_op_screen': (no_op_screen,
                     'Is a column constant within each user? If so it is provably a no-op.'),
    'standalone_rank': (standalone_rank,
                        'Score a raw column directly as a ranker on validation, no training.'),
    'label_rate_by': (label_rate_by, 'P(long_view | column) on train, bucketed.'),
    'history_coverage': (history_coverage,
                         'Fraction of validation rows whose (user, video|author) pair is in train.'),
    'headroom_by_list_length': (headroom_by_list_length,
                                'Movable metric by list length, with invariant-user share.'),
    'column_inventory': (column_inventory, 'What columns exist and how to reach them.'),
}


def catalogue():
    return {k: doc for k, (_, doc) in REGISTRY.items()}


def run(name, **params):
    """Execute one whitelisted analysis. Unknown names are refused, not evaluated."""
    if name not in REGISTRY:
        return {'error': 'unknown analysis %r' % name, 'available': sorted(REGISTRY)}
    fn, _ = REGISTRY[name]
    try:
        return fn(**params)
    except TypeError as exc:
        return {'error': 'bad parameters for %r: %s' % (name, exc)}
    except Exception as exc:
        return {'error': '%s: %s' % (type(exc).__name__, exc)}


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) < 2:
        print(json.dumps(catalogue(), indent=2))
    else:
        kw = dict(kv.split('=', 1) for kv in sys.argv[2:])
        print(json.dumps(run(sys.argv[1], **kw), indent=2, default=str))
