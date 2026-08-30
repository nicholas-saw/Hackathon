"""Build and validate submissions THROUGH the agent's pipeline.

Why this file exists at all: `kit/submit.py --make` imports `encode` from `kit/data.py`
and `FM` from `kit/baseline.py`. It never touches `pipeline/`. So an agent that spends
fifty iterations improving `pipeline/features.py` and `pipeline/model.py` and then calls
`submit.py --make` ships the untouched official baseline and scores exactly 0.0000 delta.

We therefore build submissions here, from `pipeline.train.fit_predict`, and use the
frozen `kit/submit.py` only for what it is good at: validating the row_id contract. We
import its functions rather than shelling out to its CLI, which crashes on Windows —
`submit.py:102` prints a check mark to a cp1252 console and raises UnicodeEncodeError
*after* validating the file successfully.
"""
import os

import numpy as np

from . import DATA_DIR, SUBMISSIONS
from .cache import load_encoded, load_splits


def _rows(split, data_dir=DATA_DIR):
    return load_splits(data_dir)[split]


def write(path, split, preds, data_dir=DATA_DIR):
    """Write a submission CSV. Returns the path."""
    from submit import write_submission          # frozen kit, imported not shelled
    rows = _rows(split, data_dir)
    preds = np.asarray(preds, dtype=float)
    if len(preds) != len(rows):
        raise ValueError('got %d predictions for %d rows in split %r'
                         % (len(preds), len(rows), split))
    if not np.all(np.isfinite(preds)):
        raise ValueError('predictions contain NaN or Inf; submit.py --check rejects those')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_submission(path, rows, preds)
    return path


def check(path, split, data_dir=DATA_DIR):
    """Validate through the frozen checker. Returns (ok, message, scores)."""
    from submit import read_submission           # frozen kit
    try:
        scores = read_submission(path, _rows(split, data_dir))
    except Exception as exc:                     # the checker raises with a readable reason
        return False, '%s: %s' % (type(exc).__name__, exc), None
    return True, 'row_id contract OK: %d rows, split=%s' % (len(scores), split), scores


def build(split='valid', seeds=(0,), model='fm', data_dir=DATA_DIR, verbose=True, **cfg):
    """Train `seeds` models through pipeline/ and return their per-split predictions.

    Returns (preds_by_seed, enc) where preds_by_seed[seed] = {split: ndarray}.
    """
    from train import fit_predict                # the agent's editable pipeline
    enc, dim = load_encoded(data_dir)
    out = {}
    for s in seeds:
        if verbose:
            print('  training %s seed=%d ...' % (model, s), flush=True)
        out[s] = fit_predict(enc, dim, model=model, seed=s, **cfg)
    return out, enc


def differs_from_baseline(path_a, path_b, tol=1e-12):
    """True when two submissions rank rows differently.

    The assertion that catches the failure this module exists to prevent: if a
    pipeline-built submission is identical to a kit-baseline-built one, the agent's work
    is not in the file.
    """
    import csv
    def col(p):
        with open(p, newline='', encoding='utf-8') as fh:
            r = csv.reader(fh)
            next(r)
            return np.array([float(x[3]) for x in r])
    a, b = col(path_a), col(path_b)
    if len(a) != len(b):
        return True
    return bool(np.max(np.abs(a - b)) > tol)


def floor(seeds=(0, 1, 2, 3, 4), data_dir=DATA_DIR):
    """The banked floor: an N-seed within-user rank-average, built through pipeline/.

    Writes two files. `floor_test.csv` is the bankable artifact — the thing you could
    submit if everything after this fails. Writing scores for test ROWS needs no test
    LABELS, so this does not touch the seal. `floor_valid.csv` is the measurable twin,
    and validation is where the reported number comes from.

    Insurance, not the deliverable. If the autonomous run succeeds you submit the
    checkpoint it designated; falling back to this file is an L5 intervention and must be
    reported as one.
    """
    from .score import rank_average, evaluate_raw
    preds, enc = build(seeds=seeds, data_dir=data_dir)

    _, yv, uv = enc['valid']
    singles = [evaluate_raw(uv, yv, preds[s]['valid'])['primary'] for s in seeds]
    ens_valid = rank_average(uv, [preds[s]['valid'] for s in seeds])
    ens_primary = evaluate_raw(uv, yv, ens_valid)['primary']

    _, _, ut = enc['test']
    ens_test = rank_average(ut, [preds[s]['test'] for s in seeds])

    # float() before round(): kit/evaluate.py hands back numpy scalars, and
    # round(np.float32) stays np.float32, which json.dumps refuses.
    out = {'single_primaries': [round(float(v), 5) for v in singles],
           'single_mean': round(float(np.mean(singles)), 5),
           'ensemble_primary_valid': round(float(ens_primary), 5),
           'gain_over_mean_seed': round(float(ens_primary) - float(np.mean(singles)), 5),
           'files': {}}
    for name, split, vec in (('floor_valid.csv', 'valid', ens_valid),
                             ('floor_test.csv', 'test', ens_test)):
        path = os.path.join(SUBMISSIONS, name)
        write(path, split, vec, data_dir)
        ok, msg, _ = check(path, split, data_dir)
        out['files'][name] = {'path': path, 'check_ok': ok, 'check_msg': msg}
    return out


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser(description='Build a submission through pipeline/.')
    ap.add_argument('--floor', action='store_true', help='build the banked N-seed floor')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument('--check', metavar='PATH', help='validate an existing submission')
    a = ap.parse_args()
    if a.check:
        ok, msg, _ = check(a.check, a.split)
        print(('OK  ' if ok else 'FAIL ') + msg)
        raise SystemExit(0 if ok else 1)
    if a.floor:
        print(json.dumps(floor(seeds=tuple(range(a.seeds))), indent=2))
