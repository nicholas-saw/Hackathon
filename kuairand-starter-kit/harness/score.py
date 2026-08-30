"""Scoring, wrapped around the frozen official evaluator, with the test split sealed.

Two jobs:
  1. Never modify or reimplement kit/evaluate.py. It is the scoreboard; we only call it.
  2. Make a test-split read impossible before the run has designated its submission.

The seal is not a convention. `score_split('test', ...)` raises unless a Journal is
supplied that already contains FINAL_DESIGNATION, and opening it emits TEST_OPEN, so
verify.py can prove the ordering afterwards.
"""
import numpy as np

from . import DATA_DIR
from .journal import FINAL_DESIGNATION, TEST_OPEN


class TestSealError(RuntimeError):
    """Raised when something tries to read test labels before the gate has closed."""


def evaluate_raw(users, labels, scores):
    """The official metric, untouched. Returns {'GAUC','nDCG@5','primary','users','rows'}."""
    from evaluate import evaluate            # frozen kit
    return evaluate(list(users), list(labels), list(scores))


def score_split(split, preds, enc=None, journal=None, data_dir=DATA_DIR):
    """Score predictions for one split.

    'train'/'valid' are always allowed. 'test' requires a journal carrying
    FINAL_DESIGNATION, and emits TEST_OPEN as a side effect.
    """
    if split == 'test':
        if journal is None:
            raise TestSealError(
                'test is sealed: score_split("test", ...) needs a journal, and that '
                'journal must already contain FINAL_DESIGNATION')
        if not journal.has(FINAL_DESIGNATION):
            raise TestSealError(
                'test is sealed: the run has not designated a final submission yet. '
                'Write FINAL_DESIGNATION first — that ordering is what verify.py checks.')
    if enc is None:
        from .cache import load_encoded
        enc, _ = load_encoded(data_dir)
    X, y, u = enc[split]
    preds = np.asarray(preds, dtype=float)
    if len(preds) != len(y):
        raise ValueError('got %d predictions for %d rows in split %r'
                         % (len(preds), len(y), split))
    if not np.all(np.isfinite(preds)):
        raise ValueError('predictions contain NaN or Inf; submit.py would reject this')
    res = evaluate_raw(u, y, preds)
    if split == 'test' and journal is not None:
        journal.append(TEST_OPEN, {'split': 'test', 'rows': int(len(y)),
                                   'metrics': {k: float(v) for k, v in res.items()}})
    return res


def paired_delta(users, labels, preds_a, preds_b, n_boot=0):
    """Primary(a) - Primary(b) over the same users.

    A bare delta is not reportable on this task: the measured paired noise floor is
    sigma ~ 0.0005 on validation, so anything under ~0.002 is inside the resolution
    limit. Returns the delta plus both absolute scores; set n_boot > 0 for a
    user-level bootstrap interval (slow — kit/evaluate.py is pure Python).
    """
    ra = evaluate_raw(users, labels, preds_a)
    rb = evaluate_raw(users, labels, preds_b)
    out = {'primary_a': ra['primary'], 'primary_b': rb['primary'],
           'delta': ra['primary'] - rb['primary'],
           'delta_gauc': ra['GAUC'] - rb['GAUC'],
           'delta_ndcg': ra['nDCG@5'] - rb['nDCG@5']}
    if n_boot:
        users = np.asarray(users)
        labels = np.asarray(labels)
        pa = np.asarray(preds_a, dtype=float)
        pb = np.asarray(preds_b, dtype=float)
        uniq, inv = np.unique(users, return_inverse=True)
        rows_by_user = [np.flatnonzero(inv == j) for j in range(len(uniq))]
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            mask = np.concatenate([rows_by_user[p] for p in pick])
            # A user drawn twice must stay two groups: evaluate() keys on user id, so
            # reusing the original id would merge the copies and corrupt the resample.
            tag = np.concatenate([np.full(len(rows_by_user[p]), c)
                                  for c, p in enumerate(pick)])
            fake_users = np.char.add(users[mask].astype(str),
                                     np.char.add('#', tag.astype(str)))
            da = evaluate_raw(fake_users, labels[mask], pa[mask])['primary']
            db = evaluate_raw(fake_users, labels[mask], pb[mask])['primary']
            deltas.append(da - db)
        deltas = np.asarray(deltas)
        out['ci95'] = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        out['boot_sd'] = float(deltas.std(ddof=1))
    return out


def rank_average(users, score_lists):
    """Within-user rank-average of several score vectors.

    The combination rule for an ensemble checkpoint. Per-user ranks are scale-free, so
    models with different score distributions combine without calibration — and any
    per-user monotone transform is a no-op under both GAUC and nDCG@5 anyway.
    """
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(us)])
    acc = np.zeros(len(users), dtype=float)
    for s in score_lists:
        s = np.asarray(s, dtype=float)
        r = np.empty(len(s), dtype=float)
        for st, c in zip(starts, counts):
            idx = order[st:st + c]
            loc = np.argsort(np.argsort(s[idx]))
            r[idx] = loc / max(c - 1, 1)
        acc += r
    return acc / len(score_lists)
