"""Phase B -- metric structure: activity/list-length buckets, invariant users,
GAUC weight concentration, oracle/movable gap. Uses the cached official-baseline
(seed 0) validation predictions produced by phase_c_baseline.py."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

STARTER_KIT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'starter-kit'))
sys.path.insert(0, STARTER_KIT)
from evaluate import evaluate, auc, ndcg_at_k

def bucketed_metrics(users, labels, scores, mask):
    """Restrict to rows where mask True, compute evaluate() over that subset."""
    u = np.asarray(users)[mask]; y = np.asarray(labels)[mask]; s = np.asarray(scores)[mask]
    if len(u) == 0:
        return {'GAUC': None, 'nDCG@5': None, 'primary': None, 'users': 0, 'rows': 0}
    return evaluate(u, y, s)

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()

    scores = np.load(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_scores.npy'))
    users = np.load(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_users.npy'), allow_pickle=True)
    labels = np.load(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_labels.npy'))
    # sanity: users/labels order must match valid df row order (encode_fields preserves order)
    assert len(scores) == len(valid), "cached score length mismatch vs valid log"

    valid = valid.reset_index(drop=True).copy()
    valid['baseline_score'] = scores

    # ---- B03 uniform-label / invariant users (validation) ----
    per_user = valid.groupby('user_id')['long_view'].agg(['sum', 'count'])
    per_user['all_neg'] = per_user['sum'] == 0
    per_user['all_pos'] = per_user['sum'] == per_user['count']
    per_user['mixed'] = ~(per_user['all_neg'] | per_user['all_pos'])
    per_user['single_impression'] = per_user['count'] == 1

    def summarize_group(mask_col):
        sub = per_user[per_user[mask_col]]
        return {'users': int(len(sub)), 'pct_users': float(len(sub) / len(per_user) * 100),
                'rows': int(sub['count'].sum()), 'pct_rows': float(sub['count'].sum() / per_user['count'].sum() * 100)}

    out['uniform_label_users'] = {
        'all_negative': summarize_group('all_neg'),
        'all_positive': summarize_group('all_pos'),
        'mixed_movable': summarize_group('mixed'),
        'single_impression': summarize_group('single_impression'),
    }

    # oracle ceiling on our local validation split, using true labels as scores
    oracle = evaluate(valid['user_id'], valid['long_view'], valid['long_view'])
    out['oracle_ceiling_local_valid'] = oracle

    baseline_metrics = evaluate(valid['user_id'], valid['long_view'], valid['baseline_score'])
    out['baseline_seed0_local_valid'] = baseline_metrics

    # ---- B02 list-length buckets ----
    list_len = per_user['count']
    bins = [0, 1, 3, 5, 10, 20, np.inf]
    labels_bins = ['1', '2-3', '4-5', '6-10', '11-20', '21+']
    per_user['len_bucket'] = pd.cut(list_len, bins=bins, labels=labels_bins)
    valid['len_bucket'] = valid['user_id'].map(per_user['len_bucket'])

    list_len_results = {}
    # Official GAUC weights only mixed-label users, by their positive count.
    # Positives from all-positive users must not enter the denominator.
    mixed_users = set(per_user.index[per_user['mixed']])
    valid['gauc_weight'] = np.where(valid['user_id'].isin(mixed_users),
                                    valid['long_view'], 0)
    total_gauc_weight = valid['gauc_weight'].sum()
    for b in labels_bins:
        mask = (valid['len_bucket'] == b).to_numpy()
        m = bucketed_metrics(valid['user_id'].to_numpy(), valid['long_view'].to_numpy(),
                              valid['baseline_score'].to_numpy(), mask)
        m_oracle = bucketed_metrics(valid['user_id'].to_numpy(), valid['long_view'].to_numpy(),
                                     valid['long_view'].to_numpy(), mask)
        n_users_bucket = int((per_user['len_bucket'] == b).sum())
        gauc_weight = float(valid.loc[mask, 'gauc_weight'].sum() / total_gauc_weight * 100) if total_gauc_weight else 0.0
        user_share = n_users_bucket / len(per_user)
        ndcg_gap = (m_oracle['nDCG@5'] - m['nDCG@5']) if m['nDCG@5'] is not None else None
        gauc_gap_contribution = (gauc_weight / 100.0) * (1.0 - m['GAUC'])
        ndcg_gap_contribution = user_share * ndcg_gap
        list_len_results[b] = {'n_users': n_users_bucket, 'n_rows': int(mask.sum()),
                                'baseline_nDCG5': m['nDCG@5'], 'baseline_GAUC': m['GAUC'],
                                'baseline_primary': m['primary'],
                                'oracle_nDCG5': m_oracle['nDCG@5'], 'oracle_GAUC': m_oracle['GAUC'],
                                'gauc_weight_share_pct': gauc_weight,
                                'movable_nDCG5_gap': ndcg_gap,
                                'overall_GAUC_gap_contribution': gauc_gap_contribution,
                                'overall_nDCG5_gap_contribution': ndcg_gap_contribution,
                                'overall_primary_gap_contribution': 0.5 * (gauc_gap_contribution + ndcg_gap_contribution)}
    out['list_length_buckets'] = list_len_results

    # ---- B01/B04/B05 activity buckets (train-derived tiers) ----
    tiers, train_counts, edges = C.train_activity_tiers(train, valid['user_id'].unique())
    valid['tier'] = valid['user_id'].map(tiers)
    per_user['tier'] = per_user.index.map(tiers)

    tier_results = {}
    for t in ['Cold', 'T1', 'T2', 'T3', 'T4']:
        mask = (valid['tier'] == t).to_numpy()
        m = bucketed_metrics(valid['user_id'].to_numpy(), valid['long_view'].to_numpy(),
                              valid['baseline_score'].to_numpy(), mask)
        m_oracle = bucketed_metrics(valid['user_id'].to_numpy(), valid['long_view'].to_numpy(),
                                     valid['long_view'].to_numpy(), mask)
        sub_pu = per_user[per_user['tier'] == t]
        fixed_pct = float((sub_pu['all_neg'] | sub_pu['all_pos']).mean() * 100) if len(sub_pu) else None
        gauc_weight = float(valid.loc[mask, 'gauc_weight'].sum() / total_gauc_weight * 100) if total_gauc_weight else 0.0
        user_share = len(sub_pu) / len(per_user)
        ndcg_gap = (m_oracle['nDCG@5'] - m['nDCG@5']) if m['nDCG@5'] is not None else None
        gauc_gap_contribution = (gauc_weight / 100.0) * (1.0 - m['GAUC'])
        ndcg_gap_contribution = user_share * ndcg_gap
        tier_results[t] = {'n_users': int(len(sub_pu)), 'n_rows': int(mask.sum()),
                            'baseline_GAUC': m['GAUC'], 'baseline_nDCG5': m['nDCG@5'], 'baseline_primary': m['primary'],
                            'oracle_GAUC': m_oracle['GAUC'], 'oracle_nDCG5': m_oracle['nDCG@5'],
                            'fixed_users_pct': fixed_pct, 'gauc_weight_share_pct': gauc_weight,
                            'movable_nDCG5_gap': ndcg_gap,
                            'overall_GAUC_gap_contribution': gauc_gap_contribution,
                            'overall_nDCG5_gap_contribution': ndcg_gap_contribution,
                            'overall_primary_gap_contribution': 0.5 * (gauc_gap_contribution + ndcg_gap_contribution)}
    out['activity_tier_buckets'] = tier_results
    out['activity_tier_edges'] = edges.tolist()
    out['gauc_weight_denominator_positive_rows_from_mixed_users'] = int(total_gauc_weight)

    # ---- validation impressions-per-user / list-length distribution (overall) ----
    out['list_length_distribution'] = {
        'min': int(list_len.min()), 'median': float(list_len.median()), 'mean': float(list_len.mean()),
        'p90': float(list_len.quantile(0.9)), 'p99': float(list_len.quantile(0.99)), 'max': int(list_len.max()),
    }

    C.save_json(out, 'phase_b_metric.json')
    print(json.dumps({k: v for k, v in out.items() if k not in ()}, indent=2, default=str))

if __name__ == '__main__':
    main()
