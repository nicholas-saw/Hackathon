"""Measurement pass: activity-tier bucket analysis + metric-invariance audit for the
current pipeline/ model, read-only diagnostic script (not one of the 3 editable
pipeline/ files -- see AGENT_RULES.md). Only ever evaluates on `valid`, same
discipline as pipeline/train.py's report_test=False default.

Reports three things:

1. GAUC / nDCG@5 per user-activity tier. Tiers are quartiles of each user's TRAIN
   impression count (their general activity level, computed without touching valid
   labels) plus a cold-start tier for users who appear in valid but never in train.

2. The proportion of valid users whose contribution to the scoreboard is
   mathematically fixed no matter what the model predicts. A user with a uniform
   label across their valid impressions (all long_view=0, or all long_view=1) has:
     - nDCG@5 pinned at a constant (0.0 if all-negative -- kit/evaluate.py returns
       0.0 whenever idcg==0; 1.0 if all-positive -- every position's gain is 1
       regardless of order, so dcg==idcg by construction), and
     - zero weight in GAUC -- kit/evaluate.py only scores users with
       0 < npos < len(labels).
   No amount of re-ranking changes either number for these users, so this is the
   ceiling on how much model quality can move the scoreboard.

3. A histogram of how many logged impressions each user has in the valid split.

Usage:
    python bucket_analysis.py [--data_dir ./KuaiRand-Pure/data] [--k 16]
                               [--epochs 40] [--seed 0] [--n_tiers 4]
"""
import argparse, collections, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'kit'))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))

from data import load
from evaluate import evaluate
from features import encode, IDX
from model import FM


def train_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    """Same early-stopping loop as pipeline/train.py's run_fm, but returns the raw
    valid predictions instead of just the aggregate metrics -- bucket analysis needs
    a per-row score to slice by tier, run_fm only hands back {'GAUC':..., ...}."""
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
    return uva, yva, m.predict(Xva)


def assign_tiers(splits, uva, n_tiers=4):
    """tier label per valid user, from TRAIN impression count (quantile-binned,
    duplicate edges collapsed so heavy ties don't produce empty tiers) plus a
    cold-start tier for users with zero train impressions."""
    train_count = collections.Counter(x[IDX['user_id']] for x in splits['train'])
    uva_set = sorted(set(uva))
    cold = [u for u in uva_set if train_count.get(u, 0) == 0]
    warm = [u for u in uva_set if train_count.get(u, 0) > 0]
    warm_c = np.array([train_count[u] for u in warm])

    edges = np.unique(np.quantile(warm_c, np.linspace(0, 1, n_tiers + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0]])   # all warm users share one train count
    n_bins = len(edges) - 1
    bin_idx = np.digitize(warm_c, edges[1:-1], right=True) if n_bins > 1 else np.zeros(len(warm_c), dtype=int)

    tier_of = {u: 'cold-start (0 train impr.)' for u in cold}
    order = ['cold-start (0 train impr.)'] if cold else []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        order.append(f"T{i+1} [{lo}-{hi} train impr.]")
    for u, b in zip(warm, bin_idx):
        lo, hi = int(edges[b]), int(edges[b + 1])
        tier_of[u] = f"T{b+1} [{lo}-{hi} train impr.]"
    return tier_of, order


def bucket_metrics(uva, yva, scores_va, tier_of, order):
    row_tier = np.array([tier_of[u] for u in uva])
    rows = []
    for t in order:
        mask = row_tier == t
        if not mask.any():
            continue
        sub_u = [u for u, m in zip(uva, mask) if m]
        r = evaluate(sub_u, yva[mask], scores_va[mask])
        rows.append({'tier': t, 'n_users': len(set(sub_u)), 'n_rows': int(mask.sum()),
                     'GAUC': float(r['GAUC']), 'nDCG@5': float(r['nDCG@5']),
                     'primary': float(r['primary'])})
    overall = evaluate(uva, yva, scores_va)
    rows.append({'tier': 'ALL', 'n_users': overall['users'], 'n_rows': overall['rows'],
                 'GAUC': float(overall['GAUC']), 'nDCG@5': float(overall['nDCG@5']),
                 'primary': float(overall['primary'])})
    return rows


def fixed_score_audit(uva, yva, tier_of=None, order=None):
    """proportion of users (and rows) whose GAUC/nDCG@5 contribution cannot change
    regardless of model quality -- see module docstring point 2."""
    byu = collections.defaultdict(lambda: [0, 0])   # user -> [npos, total]
    for u, y in zip(uva, yva):
        s = byu[u]; s[1] += 1; s[0] += int(y)

    def summarize(users):
        tot_u = len(users)
        rows = sum(byu[u][1] for u in users)
        neg = sum(1 for u in users if byu[u][0] == 0)
        pos = sum(1 for u in users if byu[u][0] == byu[u][1])
        fixed_rows = sum(byu[u][1] for u in users if byu[u][0] == 0 or byu[u][0] == byu[u][1])
        single = sum(1 for u in users if byu[u][1] == 1)
        fixed = neg + pos
        return {'n_users': tot_u, 'n_rows': rows,
                'fixed_users': fixed, 'fixed_users_pct': fixed / tot_u if tot_u else 0.0,
                'all_negative_users': neg, 'all_positive_users': pos,
                'single_impression_users': single,
                'fixed_rows': fixed_rows, 'fixed_rows_pct': fixed_rows / rows if rows else 0.0}

    out = {'overall': summarize(list(byu.keys()))}
    if tier_of is not None:
        by_tier = collections.defaultdict(list)
        for u in byu:
            by_tier[tier_of[u]].append(u)
        out['by_tier'] = {t: summarize(by_tier[t]) for t in order if t in by_tier}
    return out


def plot_impressions_hist(uva, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    counts = np.array(list(collections.Counter(uva).values()))
    cap = 40
    capped = np.minimum(counts, cap)
    median, mean = np.median(counts), counts.mean()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bg = '#ffffff'
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    bar_color = '#2f6fed'

    bins = np.arange(1, cap + 2) - 0.5
    ax.hist(capped, bins=bins, color=bar_color, edgecolor=bg, linewidth=0.6)

    ax.axvline(median, color='#44464b', linestyle='--', linewidth=1.2)
    ax.text(median + 0.6, ax.get_ylim()[1] * 0.92, f"median = {median:.0f}",
            color='#44464b', fontsize=9)

    n_over = int((counts > cap).sum())
    fig.text(0.06, 0.97, "Logged impressions per user -- validation split",
              fontsize=13, color='#1a1a1a', ha='left', va='top')
    subtitle = f"n users = {len(counts):,}   mean = {mean:.1f}   median = {median:.0f}   max = {int(counts.max())}"
    if n_over:
        subtitle += f"   ({n_over} users above {cap} folded into the last bin)"
    fig.text(0.06, 0.925, subtitle, fontsize=9.5, color='#6b6d73', ha='left', va='top')
    ax.set_xlabel("impressions logged in valid (per user)", fontsize=10, color='#44464b')
    ax.set_ylabel("number of users", fontsize=10, color='#44464b')

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#c9cbd1')
    ax.tick_params(colors='#44464b', labelsize=9)
    ax.grid(axis='y', color='#e8e9ec', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.12)
    fig.savefig(out_path)
    plt.close(fig)
    return {'n_users': len(counts), 'min': int(counts.min()), 'median': float(median),
            'mean': float(mean), 'max': int(counts.max()),
            'p90': float(np.percentile(counts, 90)), 'p99': float(np.percentile(counts, 99))}


def fmt_table(rows, cols, widths):
    def cell(v, w):
        if isinstance(v, float):
            return f"{v:<{w}.4f}"
        return f"{v!s:<{w}}"

    print("  ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("  ".join('-' * w for w in widths))
    for r in rows:
        print("  ".join(cell(r[c], w) for c, w in zip(cols, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_tiers', type=int, default=4)
    ap.add_argument('--out_dir', default='./analysis_output')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()})

    print("\ntraining pipeline/ FM on train, selecting on valid (test untouched) ...")
    uva, yva, scores_va = train_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                    verbose=not a.quiet)

    os.makedirs(a.out_dir, exist_ok=True)

    # 1. activity-tier bucket metrics
    tier_of, order = assign_tiers(splits, uva, n_tiers=a.n_tiers)
    bucket_rows = bucket_metrics(uva, yva, scores_va, tier_of, order)
    print("\n=== GAUC / nDCG@5 by user-activity tier (tier = quartile of TRAIN impressions) ===")
    fmt_table(bucket_rows, ['tier', 'n_users', 'n_rows', 'GAUC', 'nDCG@5', 'primary'],
              [32, 8, 8, 8, 8, 8])

    # 2. mathematically-fixed-score audit
    audit = fixed_score_audit(uva, yva, tier_of, order)
    ov = audit['overall']
    print("\n=== users whose valid score is fixed regardless of model (uniform label) ===")
    print(f"  {ov['fixed_users']}/{ov['n_users']} users ({ov['fixed_users_pct']*100:.1f}%) have a "
          f"uniform label across their valid impressions")
    print(f"    all-negative (nDCG@5 pinned to 0.0, excluded from GAUC): {ov['all_negative_users']}")
    print(f"    all-positive (nDCG@5 pinned to 1.0, excluded from GAUC): {ov['all_positive_users']}")
    print(f"    of which single-impression users: {ov['single_impression_users']}")
    print(f"  those users account for {ov['fixed_rows']}/{ov['n_rows']} valid rows "
          f"({ov['fixed_rows_pct']*100:.1f}%)")
    print("\n  by tier:")
    audit_rows = [{'tier': t, **v} for t, v in audit['by_tier'].items()]
    for r in audit_rows:
        r['fixed_users_pct'] = round(r['fixed_users_pct'], 4)
    fmt_table(audit_rows, ['tier', 'n_users', 'fixed_users', 'fixed_users_pct'], [32, 8, 12, 12])

    # 3. impressions-per-user histogram (valid split)
    png_path = os.path.join(a.out_dir, 'valid_impressions_per_user.png')
    hist_stats = plot_impressions_hist(uva, png_path)
    print(f"\n=== impressions-per-user distribution, valid split ===")
    print(f"  n_users={hist_stats['n_users']} min={hist_stats['min']} median={hist_stats['median']:.0f} "
          f"mean={hist_stats['mean']:.2f} p90={hist_stats['p90']:.0f} p99={hist_stats['p99']:.0f} "
          f"max={hist_stats['max']}")
    print(f"  saved plot -> {png_path}")

    summary_path = os.path.join(a.out_dir, 'bucket_analysis_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump({'bucket_metrics': bucket_rows, 'fixed_score_audit': audit,
                    'impressions_per_user_valid': hist_stats}, fh, indent=2)
    print(f"  saved summary -> {summary_path}")


if __name__ == '__main__':
    main()
