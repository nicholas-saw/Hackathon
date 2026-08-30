"""The only sanctioned route to raw columns `kit/data.py` does not expose.

Two rules in AGENT_RULES.md collide. Section 1 says "never open the raw CSVs directly".
Section 5 item 2 says reading them from `features.py` is "within your editable surface"
and is the *only* way to reach `is_click`, `play_time_ms`, `hourmin` and the rest — which
organizer directions #2 (sequence), #3 (multi-task) and #4 (watch-time) all require.

They cannot both stand, because `log_standard_4_22_to_5_08_pure.csv` spans validation AND
test and carries `long_view`. Granting raw reads as written hands over every test label.

This adapter resolves it: raw columns are available, but only for train and validation,
because the test rows are dropped during parsing and never materialise. It also fixes the
join the rules get wrong — section 5 says to join on `(user_id, video_id)`, which
`kit/submit.py` itself documents as non-unique (3.06% of rows repeat, up to 12x). Joining
on it fans out and mis-attributes another impression's outcome onto the current row, which
is a leak wearing a feature's clothes. We align POSITIONALLY instead, by re-reading in
`kit/data.py`'s exact file order and applying its exact date filter.
"""
import csv
import os

import numpy as np

from . import DATA_DIR, LEAKY_COLUMNS, SPLIT_SIZES

# Mirrors kit/data.py: files in this order, then filter by date, preserving file order.
_LOG_FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')
_SPLIT_DATES = {'train': (20220408, 20220421), 'valid': (20220422, 20220428)}
_TEST_DATES = (20220429, 20220508)

LOG_COLUMNS = ('user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click', 'is_like',
               'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view',
               'play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time',
               'is_profile_enter', 'is_rand', 'tab')


class TestRowsRequested(RuntimeError):
    """Raised when someone asks this adapter for test-period data."""


def raw_columns(names, splits=('train', 'valid'), data_dir=DATA_DIR, dtype=None):
    """Return raw log columns, positionally aligned to kit.data.load()[split].

    `splits` may only contain 'train' and 'valid'. Asking for 'test' raises — that is the
    whole point of routing through here.

    Returns {split: {column_name: np.ndarray}} with one entry per row of that split, in
    the same order as `splits[split]`, so it can be zipped with the encoded arrays.
    """
    bad = [s for s in splits if s not in _SPLIT_DATES]
    if bad:
        raise TestRowsRequested(
            'adapter serves train and valid only; refused %r. Test rows carry long_view '
            'for the hidden period — that is why this door is locked.' % bad)
    unknown = [n for n in names if n not in LOG_COLUMNS]
    if unknown:
        raise ValueError('unknown log column(s) %r; available: %s'
                         % (unknown, ', '.join(LOG_COLUMNS)))

    want = list(names)
    out = {s: {n: [] for n in want} for s in splits}
    for fname in _LOG_FILES:
        with open(os.path.join(data_dir, fname), newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                d = int(r['date'])
                if _TEST_DATES[0] <= d <= _TEST_DATES[1]:
                    continue                      # test rows never enter memory
                for s in splits:
                    lo, hi = _SPLIT_DATES[s]
                    if lo <= d <= hi:
                        for n in want:
                            out[s][n].append(r[n])
                        break
    for s in splits:
        got = len(out[s][want[0]]) if want else 0
        if got != SPLIT_SIZES[s]:
            raise AssertionError('adapter produced %d rows for %r, official is %d — '
                                 'alignment with kit.data.load() is broken'
                                 % (got, s, SPLIT_SIZES[s]))
        for n in want:
            out[s][n] = np.asarray(out[s][n], dtype=dtype) if dtype else np.asarray(out[s][n])
    return out


def auxiliary_targets(names=('is_click',), data_dir=DATA_DIR):
    """Post-impression signals as multi-task TARGETS, as float32 vectors.

    Legal use: an auxiliary head. Illegal use: an input feature for the same row. The
    static guard in harness/guards.py catches the illegal one; this function exists so
    the legal one does not require opening a CSV by hand.
    """
    for n in names:
        if n not in LEAKY_COLUMNS:
            raise ValueError('%r is not a post-impression signal; read it with '
                             'raw_columns() as an ordinary feature' % n)
    cols = raw_columns(names, data_dir=data_dir)
    return {s: {n: (v != '0').astype(np.float32) if v.dtype.kind in 'US'
                else v.astype(np.float32)
                for n, v in d.items()} for s, d in cols.items()}


def entity_table(which, data_dir=DATA_DIR):
    """Static per-entity side tables, keyed by id. No temporal component, no leak.

    which: 'user' | 'video_basic' | 'video_stat'
    """
    fname = {'user': 'user_features_pure.csv',
             'video_basic': 'video_features_basic_pure.csv',
             'video_stat': 'video_features_statistic_pure.csv'}[which]
    key = 'user_id' if which == 'user' else 'video_id'
    table = {}
    with open(os.path.join(data_dir, fname), newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            table[r[key]] = r
    return table


def max_date_served(data_dir=DATA_DIR):
    """Verification helper: the newest date this adapter will ever hand back."""
    d = raw_columns(('date',), data_dir=data_dir)
    return max(int(v) for s in d for v in d[s]['date'])


if __name__ == '__main__':
    import json
    cols = raw_columns(('date', 'is_click', 'hourmin'))
    print(json.dumps({s: {'rows': len(v['date']),
                          'max_date': int(max(int(x) for x in v['date']))}
                      for s, v in cols.items()}, indent=2))
