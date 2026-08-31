"""Controlled objective ablation: pointwise vs pairwise vs listwise, one model.

WHY THIS EXISTS
---------------
Nine independently written "listwise softmax" implementations scored between -0.00318
and +0.00162 on this task (constraints.md C25). Seven of them lost, one won, and the
winner became the banked submission. Reading that as "listwise works" or "listwise does
not work" are both wrong: the label names an INTENT, not a formulation, and the runs
differed in batching, grouping, normalisation and loss composition all at once.

The only way to attribute the winner's gain to its objective is to hold the
implementation fixed and change nothing but the loss. That is what this file does.

WHAT IS HELD FIXED
------------------
Everything except the per-batch update rule:
  - the encoded features and the train/valid split (harness.cache.load_encoded)
  - the FM architecture, dimension, k, and initialisation (pipeline.model.FM.__init__)
  - the Adam optimizer, learning rate and L2 (FM._apply_grad)
  - the batching: `_build_grouped_batches`, verbatim from the winning implementation.
    Every objective sees the same user-grouped batches in the same order for a seed.
  - the epoch budget, the early-stopping metric (validation primary) and its patience
  - the seed, matched across objectives

WHAT VARIES
-----------
Exactly one call inside the inner loop:
  pointwise -> FM.step(Xb, yb)                     pointwise BCE, the official baseline loss
  pairwise  -> step_pairwise(Xb, yb, offs)         within-group BPR over all pos x neg pairs
  listwise  -> step_listwise(Xb, yb, offs)         within-group softmax CE  <- the winner

READING THE RESULT
------------------
If listwise clearly leads, the objective is the active ingredient and the direction is
real. If all three land together, the winner's gain came from something else in its
implementation -- its grouped batching is the obvious suspect, since that is itself a
change from the official baseline -- and seven "failed listwise" runs failed for reasons
that have nothing to do with being listwise.

Run:  python research/objective_ablation/ablate.py --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))

from harness.cache import load_encoded                    # noqa: E402
from harness.score import evaluate_raw                     # noqa: E402
from model import FM, sigmoid                              # noqa: E402


# --------------------------------------------------------------------------------
# Verbatim from the winning implementation (run 20260830T235541Z it1, journal diff).
# Groups rows by user, never splits a user across a batch, reshuffles group order each
# epoch. All three objectives consume these identical batches.
# --------------------------------------------------------------------------------
def _build_grouped_batches(X, y, users, target_bs=8192, rng=None):
    users_arr = np.asarray(users)
    order = np.argsort(users_arr, kind='stable')
    Xs, ys, us = X[order], y[order], users_arr[order]
    change = np.where(us[1:] != us[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(us)]))
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    if rng is not None:
        rng.shuffle(groups)

    batches, batch_idx, batch_offsets, cur = [], [], [0], 0
    for (s, e) in groups:
        batch_idx.extend(range(s, e))
        batch_offsets.append(batch_offsets[-1] + (e - s))
        cur += (e - s)
        if cur >= target_bs:
            batches.append((Xs[batch_idx], ys[batch_idx],
                            np.array(batch_offsets, dtype=np.int64)))
            batch_idx, batch_offsets, cur = [], [0], 0
    if batch_idx:
        batches.append((Xs[batch_idx], ys[batch_idx],
                        np.array(batch_offsets, dtype=np.int64)))
    return batches


class AblationFM(FM):
    """The current FM, plus the two grouped losses. Architecture and optimizer untouched.

    `step_listwise` is the winning implementation's update, transcribed from the journal
    diff. Its own `_adam_update(gV, gW, g.sum())` tail is byte-for-byte equivalent to the
    inherited `_apply_grad(X, g, S, E)` (same L2, same Adam constants, same bias step), so
    calling the inherited one changes no arithmetic and keeps the optimizer shared with
    the other two objectives -- which is the entire point of the experiment.
    """

    def step_listwise(self, X, y, group_offsets):
        """Within-group softmax cross-entropy. Groups with 0 or all positives are
        skipped: they carry no ordering signal. Normalised by total positives."""
        z, E, S = self.logits(X)
        grad = np.zeros(len(z), dtype=np.float32)
        total_pos, loss_sum = 0.0, 0.0
        offs = group_offsets
        for gi in range(len(offs) - 1):
            s, e = int(offs[gi]), int(offs[gi + 1])
            if e - s < 2:
                continue
            zg, yg = z[s:e], y[s:e]
            pos_count, size = float(yg.sum()), e - s
            if pos_count <= 0.0 or pos_count >= size:
                continue
            mx = zg.max()
            expz = np.exp(zg - mx)
            denom = expz.sum()
            grad[s:e] = pos_count * (expz / denom) - yg
            loss_sum += -(float(np.sum(yg * zg)) - pos_count * (mx + np.log(denom)))
            total_pos += pos_count

        if total_pos <= 0.0:
            return 0.0
        g = (grad / total_pos).astype(np.float32)
        self._apply_grad(X, g, S, E)
        return float(loss_sum / total_pos)

    def step_pairwise(self, X, y, group_offsets):
        """Within-group BPR over every positive x negative pair.

        Deliberately mirrors `step_listwise`'s scaffolding rather than the historical BPR
        runs: same eligible groups (mixed-label only), same normalisation discipline
        (divide by the total number of contributing units), same batches. So a difference
        against listwise here is attributable to the loss, not to sampling or scaling.
        """
        z, E, S = self.logits(X)
        grad = np.zeros(len(z), dtype=np.float32)
        total_pairs, loss_sum = 0.0, 0.0
        offs = group_offsets
        for gi in range(len(offs) - 1):
            s, e = int(offs[gi]), int(offs[gi + 1])
            if e - s < 2:
                continue
            yg = y[s:e]
            pos_count, size = float(yg.sum()), e - s
            if pos_count <= 0.0 or pos_count >= size:
                continue
            zg = z[s:e]
            pi = np.flatnonzero(yg > 0)
            ni = np.flatnonzero(yg <= 0)
            d = zg[pi][:, None] - zg[ni][None, :]        # (P, N)
            w = (1.0 - sigmoid(d))                       # -dL/dd for -log sigmoid(d)
            grad[s + pi] -= w.sum(axis=1)
            grad[s + ni] += w.sum(axis=0)
            loss_sum += float(-np.log(sigmoid(d) + 1e-9).sum())
            total_pairs += float(d.size)

        if total_pairs <= 0.0:
            return 0.0
        g = (grad / total_pairs).astype(np.float32)
        self._apply_grad(X, g, S, E)
        return float(loss_sum / total_pairs)


OBJECTIVES = ('pointwise', 'pairwise', 'listwise')


def train_one(enc, dim, objective, seed, k=16, lr=0.001, epochs=40, bs=8192,
              patience=4, verbose=False):
    """One (objective, seed) cell. Identical in every respect except the update call."""
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    m = AblationFM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        # Same generator, same seed => the same batch sequence for every objective.
        for (Xb, yb, offs) in _build_grouped_batches(Xtr, ytr, utr, target_bs=bs, rng=rng):
            if objective == 'pointwise':
                m.step(Xb, yb)
            elif objective == 'pairwise':
                m.step_pairwise(Xb, yb, offs)
            else:
                m.step_listwise(Xb, yb, offs)

        p = float(evaluate_raw(uva, yva, m.predict(Xva))['primary'])
        if verbose:
            print('    %-9s seed %d epoch %2d  valid primary %.5f'
                  % (objective, seed, ep, p), flush=True)
        if p > best + 1e-5:
            best, bad, best_ep = p, 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break

    m.V, m.W, m.b = best_state
    r = evaluate_raw(uva, yva, m.predict(Xva))
    return {'objective': objective, 'seed': seed,
            'primary': float(r['primary']), 'GAUC': float(r['GAUC']),
            'nDCG@5': float(r['nDCG@5']), 'best_epoch': best_ep,
            'seconds': round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--objectives', default=','.join(OBJECTIVES))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'results.json'))
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    objectives = [o.strip() for o in a.objectives.split(',') if o.strip()]

    enc, dim = load_encoded()
    rows = []
    for objective in objectives:
        for seed in seeds:
            r = train_one(enc, dim, objective, seed, epochs=a.epochs, verbose=a.verbose)
            rows.append(r)
            print('%-9s seed %d  primary %.5f  GAUC %.5f  nDCG@5 %.5f  ep%-2d  %5.1fs'
                  % (objective, seed, r['primary'], r['GAUC'], r['nDCG@5'],
                     r['best_epoch'], r['seconds']), flush=True)
            with open(a.out, 'w', encoding='utf-8') as fh:   # checkpoint as we go
                json.dump(rows, fh, indent=1)

    print()
    print('=== matched-seed summary (validation primary) ===')
    base = None
    for objective in objectives:
        v = [r['primary'] for r in rows if r['objective'] == objective]
        if not v:
            continue
        mean = sum(v) / len(v)
        if objective == 'pointwise':
            base = mean
        line = '%-9s n=%d  mean %.5f  min %.5f  max %.5f' % (
            objective, len(v), mean, min(v), max(v))
        if base is not None and objective != 'pointwise':
            per = [r for r in rows if r['objective'] == objective]
            pw = {r['seed']: r['primary'] for r in rows if r['objective'] == 'pointwise'}
            paired = [r['primary'] - pw[r['seed']] for r in per if r['seed'] in pw]
            if paired:
                line += '   vs pointwise: mean %+.5f  worst %+.5f  (%d/%d up)' % (
                    sum(paired) / len(paired), min(paired),
                    sum(1 for d in paired if d > 0), len(paired))
        print(line)
    print()
    print('wrote', a.out)


if __name__ == '__main__':
    main()
