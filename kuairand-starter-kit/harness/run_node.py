"""One experiment, in its own process.

The controller shells out to this rather than calling fit_predict in-process, for three
reasons: a syntax error in agent-written code must not kill the controller; a runaway
train loop must be killable as a process tree; and peak RSS must be attributable to the
node rather than to the whole run.

Prints a single JSON line to stdout prefixed with RESULT so the parent can parse it
without being confused by anything the pipeline chose to print.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.cache import load_encoded          # noqa: E402
from harness.score import evaluate_raw          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='npz path for the predictions')
    ap.add_argument('--config', default='{}', help='JSON kwargs for fit_predict')
    ap.add_argument('--seeds', default='0', help='comma-separated seeds')
    a = ap.parse_args()

    cfg = json.loads(a.config)
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    t0 = time.time()
    enc, dim = load_encoded()

    # Blank the hidden-test labels before any agent-written code can reach them.
    # fit_predict needs test FEATURES to produce a submission vector; it never needs
    # test labels. Until this line the whole firewall rested on a regex in
    # harness/guards.py, and a regex is defeated by ordinary spellings --
    # `Xt, yt, ut = enc['test']`, `enc.get('test')[1]`, or a variable key all walk
    # straight past it. This makes the leak structurally impossible instead of
    # textually discouraged: the array the agent could read is zeros, so no test
    # label can influence a development decision even if every guard is bypassed.
    Xt, _yt, ut = enc['test']
    enc['test'] = (Xt, np.zeros(len(ut), dtype=np.float32), ut)

    from train import fit_predict                # the agent's editable pipeline
    from harness.score import rank_average

    _, yv, uv = enc['valid']
    per_seed, preds = {}, {}
    for s in seeds:
        p = fit_predict(enc, dim, seed=s, **cfg)
        per_seed[s] = p
        r = evaluate_raw(uv, yv, p['valid'])
        preds[s] = float(r['primary'])

    if len(seeds) > 1:                            # a checkpoint may be an ensemble
        valid_vec = rank_average(uv, [per_seed[s]['valid'] for s in seeds])
        _, _, ut = enc['test']
        test_vec = rank_average(ut, [per_seed[s]['test'] for s in seeds])
    else:
        valid_vec = per_seed[seeds[0]]['valid']
        test_vec = per_seed[seeds[0]]['test']

    # A diverged model produces NaN scores that evaluate() will happily rank, yielding a
    # plausible-looking primary from garbage — and submit.py would reject the file at the
    # last moment. Fail here instead, loudly, so the node is pruned and logged.
    for name, vec in (('valid', valid_vec), ('test', test_vec)):
        arr = np.asarray(vec, dtype=float)
        if not np.all(np.isfinite(arr)):
            bad = int((~np.isfinite(arr)).sum())
            raise ValueError('non-finite predictions: %d of %d %s scores are NaN/Inf '
                             '(model diverged)' % (bad, arr.size, name))

    res = evaluate_raw(uv, yv, valid_vec)
    np.savez_compressed(a.out, valid=np.asarray(valid_vec, dtype=np.float32),
                        test=np.asarray(test_vec, dtype=np.float32))

    out = {'metrics': {k: float(v) for k, v in res.items()},
           'per_seed_primary': preds,
           'seeds': seeds,
           'config': cfg,
           'seconds': round(time.time() - t0, 2),
           'preds_path': a.out}
    print('RESULT ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
