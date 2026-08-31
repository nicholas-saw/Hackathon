"""Training loop + CLI -- the only place to change the optimisation process
(batching, epochs, early stopping, multi-task orchestration). See AGENT_RULES.md.
  --model pop          : item popularity (official baseline; pure statistics, no training)
  --model fm           : Factorization Machine, pointwise BCE (the starting model)
  --model fm_listwise  : same FM, within-user listwise softmax (ListNet top-1) + BCE mix
  --model fm_listwise_pure : same FM, PURE within-user softmax CE, no BCE mix,
                         uncapped groups -- the banked submission's loss
  --model random       : random scores (lower bound; self-check that the evaluator is intact)
numpy only. Usage: see README.md
"""
import argparse, collections, os, sys, time
import numpy as np

# kit/ is frozen and a sibling of pipeline/, so it is not on sys.path by default.
# These two lines only make it importable; they do not make kit/ editable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kit'))

from data import load
from evaluate import evaluate
from features import encode, FIELDS, IDX
from model import FM

# ---------------- item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0, report_test=False):
    """report_test=False (default): score valid only; never touch test while iterating.
    Pass report_test=True only for the single final report (train.py --final)."""
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[IDX['video_id']]] += 1; pos[x[IDX['video_id']]] += x[IDX['label']]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in (('valid', 'test') if report_test else ('valid',)):
        rws = splits[name]
        out[name] = evaluate([x[IDX['user_id']] for x in rws], [x[IDX['label']] for x in rws],
                             [score(x[IDX['video_id']]) for x in rws])
    return out

def run_random(splits, seed=0, report_test=False):
    """report_test=False (default): valid only, for the same reason as run_pop."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in (('valid', 'test') if report_test else ('valid',)):
        rws = splits[name]
        out[name] = evaluate([x[IDX['user_id']] for x in rws], [x[IDX['label']] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           report_test=False):
    """report_test=False (default): training, early stopping and model selection use
    valid only, and test is untouched. This enforces AGENT_RULES.md rule 3 in code,
    not by discipline. Pass True only when reporting the final number."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    out = {'valid': evaluate(uva, yva, m.predict(Xva))}
    if report_test:
        Xte, yte, ute = enc['test']
        out['test'] = evaluate(ute, yte, m.predict(Xte))
    return out

# ---------------- grouped batching for the pure listwise objective ----------------
def _build_grouped_batches(X, y, users, target_bs=8192, rng=None):
    """Pack whole user-groups into batches of roughly target_bs rows, never splitting a
    user. Group order is reshuffled each call so training sees different compositions.

    Transcribed from run 20260830T235541Z iteration 1 along with FM.step_listwise.
    Note this is NOT the batching `fm_listwise` uses: `_make_listwise_batches` below
    subsamples any group larger than `cap`, and this one does not.

    Returns a list of (Xb, yb, group_offsets) tuples.
    """
    users_arr = np.asarray(users)
    order = np.argsort(users_arr, kind='stable')
    Xs, ys, us = X[order], y[order], users_arr[order]
    change = np.where(us[1:] != us[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(us)]))
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    if rng is not None:
        rng.shuffle(groups)

    batches, batch_idx, batch_offsets, cur = [], [], [0], 0
    for (st, en) in groups:
        batch_idx.extend(range(st, en))
        batch_offsets.append(batch_offsets[-1] + (en - st))
        cur += (en - st)
        if cur >= target_bs:
            batches.append((Xs[batch_idx], ys[batch_idx],
                            np.array(batch_offsets, dtype=np.int64)))
            batch_idx, batch_offsets, cur = [], [0], 0
    if batch_idx:
        batches.append((Xs[batch_idx], ys[batch_idx],
                        np.array(batch_offsets, dtype=np.int64)))
    return batches


# ---------------- listwise batching helpers ----------------
def _group_rows_by_user(users):
    """users: list-like aligned to X/y rows -> dict user -> np.array of row indices."""
    d = collections.defaultdict(list)
    for i, u in enumerate(users):
        d[u].append(i)
    return {u: np.array(idxs, dtype=np.int64) for u, idxs in d.items()}

def _make_listwise_batches(user_to_idx, bs, cap, rng):
    """Build epoch batches: shuffle user order, cap each user's group at `cap` rows
    (random subsample without replacement if larger), shuffle within-group order, then
    greedily pack whole user groups into batches of roughly `bs` rows. Returns a list of
    (row_indices, group_ends) pairs; group_ends are exclusive cumulative offsets into the
    concatenated row_indices array, matching FM.step_list's contract."""
    users = list(user_to_idx.keys())
    rng.shuffle(users)
    batches = []
    cur_idx, cur_ends, cur_len = [], [], 0
    for u in users:
        idx = user_to_idx[u]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        else:
            idx = idx.copy()
        rng.shuffle(idx)
        cur_idx.append(idx)
        cur_len += len(idx)
        cur_ends.append(cur_len)
        if cur_len >= bs:
            batches.append((np.concatenate(cur_idx), np.array(cur_ends, dtype=np.int64)))
            cur_idx, cur_ends, cur_len = [], [], 0
    if cur_idx:
        batches.append((np.concatenate(cur_idx), np.array(cur_ends, dtype=np.int64)))
    return batches

def run_fm_listwise(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0,
                     verbose=True, report_test=False, lw_alpha=0.3, cap=64):
    """Same FM capacity as run_fm, trained with a within-user listwise softmax
    (ListNet top-1) mixed with lw_alpha * pointwise BCE. Batches are built by grouping
    train rows by user_id (each user's group capped at `cap` rows), so the softmax
    gradient only ever compares scores within the same user -- exactly the quantity
    GAUC and nDCG@5 depend on. Early stopping/model selection stay on valid primary."""
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    user_to_idx = _group_rows_by_user(utr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        batches = _make_listwise_batches(user_to_idx, bs, cap, rng)
        losses = []
        for idxs, ends in batches:
            losses.append(m.step_list(Xtr[idxs], ytr[idxs], ends, lw_alpha=lw_alpha))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    out = {'valid': evaluate(uva, yva, m.predict(Xva))}
    if report_test:
        Xte, yte, ute = enc['test']
        out['test'] = evaluate(ute, yte, m.predict(Xte))
    return out

# ---------------- harness contract ----------------
def fit_predict(enc, dim, model='fm', k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                seed=0, verbose=False, lw_alpha=0.3, cap=64):
    """Train on `train` and return raw scores for every split.

    THE HARNESS CALLS THIS. `run_fm`/`run_fm_listwise` above return metrics only, which
    is enough for a human reading stdout but not enough to build a submission -- and a
    submission built any other way (e.g. kit/submit.py --make) silently ships the
    untouched baseline instead of whatever this pipeline learned.

    Contract, preserve it when you change the model or the loss:
        fit_predict(enc, dim, ...) -> {'train': ndarray, 'valid': ndarray, 'test': ndarray}
    One float per row, higher = more relevant, aligned to enc[split] row order. Selection
    and early stopping use `valid` only; `test` scores are produced but never read here.
    """
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    if model == 'pop':
        pos, imp = collections.Counter(), collections.Counter()
        for xrow, yrow in zip(Xtr[:, 1], ytr):          # column 1 == video_id slot
            imp[int(xrow)] += 1; pos[int(xrow)] += yrow
        gmean = sum(pos.values()) / max(sum(imp.values()), 1)
        prior = 20.0
        def sc(v):
            v = int(v)
            return (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
        return {sp: np.array([sc(v) for v in enc[sp][0][:, 1]], dtype=float) for sp in enc}

    if model == 'random':
        rng = np.random.default_rng(seed)
        return {sp: rng.random(len(enc[sp][1])) for sp in enc}

    if model == 'fm_listwise_pure':
        # The banked submission's model. Kept as a separate key rather than folded into
        # `fm_listwise`: the two differ in loss composition, group eligibility,
        # normalisation and capping, and conflating them under one name is what made
        # `submissions/verified_listwise_3seed_ensemble.csv` unreproducible from this
        # tree in the first place.
        m = FM(dim, k=k, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            for (Xb, yb, offs) in _build_grouped_batches(Xtr, ytr, utr, target_bs=bs,
                                                         rng=rng):
                m.step_listwise(Xb, yb, offs)
            p = evaluate(uva, yva, m.predict(Xva))['primary']
            if verbose:
                print(f"  epoch {ep:2d} valid primary {p:.5f}", flush=True)
            if p > best + 1e-5:
                best, bad = p, 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b = best_state
        return {sp: m.predict(enc[sp][0]) for sp in enc}

    if model == 'fm_listwise':
        user_to_idx = _group_rows_by_user(utr)
        m = FM(dim, k=k, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            batches = _make_listwise_batches(user_to_idx, bs, cap, rng)
            for idxs, ends in batches:
                m.step_list(Xtr[idxs], ytr[idxs], ends, lw_alpha=lw_alpha)
            p = evaluate(uva, yva, m.predict(Xva))['primary']
            if verbose:
                print(f"  epoch {ep:2d} valid primary {p:.5f}", flush=True)
            if p > best + 1e-5:
                best, bad = p, 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b = best_state
        return {sp: m.predict(enc[sp][0]) for sp in enc}

    # default: model == 'fm', pointwise BCE
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        p = evaluate(uva, yva, m.predict(Xva))['primary']
        if verbose:
            print(f"  epoch {ep:2d} valid primary {p:.5f}", flush=True)
        if p > best + 1e-5:
            best, bad = p, 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return {sp: m.predict(enc[sp][0]) for sp in enc}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='the extracted KuaiRand-Pure data directory')
    ap.add_argument('--model', default='fm',
                    choices=['pop', 'fm', 'fm_listwise', 'fm_listwise_pure', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lw_alpha', type=float, default=0.3,
                    help='weight of the pointwise BCE term mixed into fm_listwise loss')
    ap.add_argument('--cap', type=int, default=64,
                    help='max rows per user group in a listwise batch (subsampled if exceeded)')
    ap.add_argument('--final', action='store_true',
                    help='also score and print test (final report only; never while iterating)')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': lambda s: run_pop(s, report_test=a.final),
           'random': lambda s: run_random(s, a.seed, report_test=a.final),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                   report_test=a.final),
           'fm_listwise': lambda s: run_fm_listwise(s, k=a.k, lr=a.lr, epochs=a.epochs,
                                                     seed=a.seed, report_test=a.final,
                                                     lw_alpha=a.lw_alpha, cap=a.cap)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in (('valid', 'test') if a.final else ('valid',)):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
