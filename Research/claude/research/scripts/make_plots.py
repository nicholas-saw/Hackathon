"""Generate summary plots for Phase B (metric headroom) and Phase G (video ratios)
from already-computed JSON results. No new data access."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as fh:
        return json.load(fh)

def plot_b():
    d = load('phase_b_metric.json')
    tiers = ['Cold', 'T1', 'T2', 'T3', 'T4']
    t = d['activity_tier_buckets']
    weight = [t[k]['gauc_weight_share_pct'] for k in tiers]
    gap = [t[k]['movable_nDCG5_gap'] for k in tiers]

    ll = d['list_length_buckets']
    ll_labels = ['1', '2-3', '4-5', '6-10', '11-20', '21+']
    ll_weight = [ll[k]['gauc_weight_share_pct'] for k in ll_labels]
    ll_gap = [ll[k]['movable_nDCG5_gap'] for k in ll_labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(tiers, weight, color='#4C72B0', alpha=0.7, label='GAUC weight share %')
    ax2.plot(tiers, gap, color='#C44E52', marker='o', label='Movable nDCG@5 gap')
    ax.set_ylabel('GAUC weight share (%)'); ax2.set_ylabel('Movable nDCG@5 gap')
    ax.set_title('By train-activity tier'); ax.set_xlabel('Tier')

    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(ll_labels, ll_weight, color='#4C72B0', alpha=0.7)
    ax2.plot(ll_labels, ll_gap, color='#C44E52', marker='o')
    ax.set_ylabel('GAUC weight share (%)'); ax2.set_ylabel('Movable nDCG@5 gap')
    ax.set_title('By validation list length'); ax.set_xlabel('List length')

    fig.suptitle('Where GAUC weight and movable nDCG@5 headroom concentrate (validation)')
    plt.tight_layout()
    path = os.path.join(C.PLOTS_DIR, 'phase_b_headroom.png')
    plt.savefig(path, dpi=120)
    print('saved', path)

def plot_g():
    d = load('phase_g_video_features.json')
    assoc = d['ratio_feature_association_valid']
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, style in [('long_view_ratio', 'o-'), ('play_ratio', 's-'),
                          ('complete_ratio', '^-'), ('like_ratio', 'd-')]:
        q = assoc[name]['long_view_rate_by_quintile']
        vals = list(q.values())
        ax.plot(range(1, len(vals) + 1), vals, style, label=name)
    ax.set_xlabel('Quintile (video-level ratio, low->high)')
    ax.set_ylabel('Validation long_view rate')
    ax.set_title('Video-statistic ratio features vs. long_view (validation)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(C.PLOTS_DIR, 'phase_g_ratio_features.png')
    plt.savefig(path, dpi=120)
    print('saved', path)

if __name__ == '__main__':
    plot_b()
    plot_g()
