"""Feature engineering -- the only place feature changes are allowed. See AGENT_RULES.md.

kit/data.py is frozen Starter Kit code. It cannot be edited, and this file imports
nothing from it: it only reads the splits returned by kit.data.load(splits). The per-row
schema is restated as IDX below. That restatement is not a second source of truth -- if
anyone changes the row layout in kit/data.py, this has to follow.

Never use ANY test-split data to build features, including its labels and any statistic
derived from it (for example, do not fit bucket edges on the test split).

Same-row input features must go through same_row(x, name). It refuses the
post-impression outcome columns in LEAKY_COLUMNS (is_click / is_like / ... /
play_time_ms / label). Those columns are legal only as multi-task targets for the same
row, or as history aggregated from OTHER rows (the user's past interactions).

For history aggregates, use harness.history -- do not hand-roll them here:

    from harness.history import prior_stats, bucketize
    rate, count = prior_stats(splits, signal='label', key='user_id')

It guarantees strictly-earlier ordering, treats tied timestamps as non-predecessors, and
keeps test outcomes out of every aggregate. Building the same thing by hand in this file
means touching the label directly, which the static guard rejects, and it is the wrong
place to get the ordering right anyway.

kit/data.py currently loads only the label out of those 11 columns. The other 10 are not
in IDX, so same_row() raises KeyError on them, which is the safe failure. Reach them
through harness.adapter, never by opening a raw CSV: log_standard_4_22_to_5_08_pure.csv
spans validation AND test and carries long_view.

Public contract:

    encode(splits) -> (enc, dim)
    enc[name] = (X, y, users)    # X: int32 (N, len(FIELDS)); y: float32 (N,); users: list
    dim                          # total vocab size across fields, fed to model.FM(dim, ...)

FIELDS is not documentation: len(FIELDS) sizes vocabs and the second dimension of X. Add a
column to raw() and you must register its name in FIELDS, or the shapes will not match.
"""
import numpy as np

# Field order of the fixed-length tuple returned by kit/data.py's load(). A restatement;
# kit/data.py itself remains the only source of truth.
IDX = {'date': 0, 'user_id': 1, 'video_id': 2, 'author_id': 3,
       'tab': 4, 'duration_ms': 5, 'label': 6}

# Post-impression outcome/feedback columns. Never use these to predict the label of the
# SAME row. Even computed purely from train data, a same-row use leaks: they are
# concurrent outcomes of the same impression as the label, not information available
# before it. kit/data.py currently loads only the label; the rest are absent from IDX, so
# same_row() raises KeyError on them (the safe failure). The full set of 11 names is
# listed here for when those columns are eventually read in.
LEAKY_COLUMNS = frozenset({
    'label',            # long_view, the main task target itself
    'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
    'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time',
    'is_profile_enter',
})

# The 5 feature fields of the current kit baseline. To add a feature, add a column in
# raw() and register its name here.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def same_row(x, name):
    """Read one column of the current row as an input feature for that row.

    Post-impression outcome columns (see LEAKY_COLUMNS) raise here. They are legal only
    as multi-task targets for the same row, or as history features built from other
    rows -- never as an input for the row being predicted, which would feed the model
    the answer it is meant to produce.
    """
    if name in LEAKY_COLUMNS:
        raise ValueError(f"{name!r} is a post-impression outcome column and cannot be a "
                         f"same-row input feature (same-row use is limited to multi-task "
                         f"targets; history features must come from other rows)")
    return x[IDX[name]]

def encode(splits):
    """Map categorical features to contiguous ids. Unseen values fall into the field's
    UNK slot. Bucket edges may be fitted on splits['train'] only, never valid/test."""
    tr = splits['train']
    edges = _bucket_edges([same_row(x, 'duration_ms') for x in tr])

    def raw(x):
        return [same_row(x, 'user_id'), same_row(x, 'video_id'), same_row(x, 'author_id'),
                same_row(x, 'tab'), str(int(np.searchsorted(edges, same_row(x, 'duration_ms'))))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # one UNK slot at the end of each field
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[IDX['label']]
            users.append(x[IDX['user_id']])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
