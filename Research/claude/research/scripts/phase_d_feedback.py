"""Phase E (post-impression feedback profile) -- prevalence, distribution,
relationship with long_view, inter-feedback association, variation by activity.
TRAIN + VALIDATION ONLY. All measurements are same-row DIAGNOSTIC associations,
not proposals to use same-row values as long_view inputs (forbidden by RULES.md)."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

BINARY = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'is_profile_enter']
CONTINUOUS = ['play_time_ms', 'profile_stay_time', 'comment_stay_time']

def summarize_continuous(s):
    return {
        'mean': float(s.mean()), 'median': float(s.median()),
        'p90': float(s.quantile(0.9)), 'p99': float(s.quantile(0.99)),
        'max': float(s.max()), 'pct_zero': float((s == 0).mean() * 100),
    }

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()

    for split_name, df in (('train', train), ('valid', valid)):
        split_out = {}
        # D01 prevalence
        prevalence = {}
        for c in BINARY:
            prevalence[c] = {'mean': float(df[c].mean()), 'pct_positive': float(df[c].mean() * 100)}
        for c in CONTINUOUS:
            prevalence[c] = summarize_continuous(df[c])
        split_out['prevalence'] = prevalence

        # D02 relationship with long_view
        assoc = {}
        for c in BINARY:
            g = df.groupby(c)['long_view'].mean()
            assoc[c] = {
                'long_view_rate_given_0': float(g.get(0, float('nan'))),
                'long_view_rate_given_1': float(g.get(1, float('nan'))),
                'pearson_r': float(df[c].corr(df['long_view'])),
            }
        for c in CONTINUOUS:
            # bucket into deciles among positive values + a zero bucket
            nz = df[df[c] > 0][c]
            assoc_c = {'pearson_r': float(df[c].corr(df['long_view'])),
                       'long_view_rate_when_zero': float(df.loc[df[c] == 0, 'long_view'].mean())}
            if len(nz) > 100:
                try:
                    deciles = pd.qcut(nz, 5, duplicates='drop')
                    rate_by_decile = df.loc[nz.index].groupby(deciles, observed=True)['long_view'].mean()
                    assoc_c['long_view_rate_by_nonzero_quintile'] = {str(k): float(v) for k, v in rate_by_decile.items()}
                except Exception as e:
                    assoc_c['quintile_error'] = str(e)
            assoc[c] = assoc_c
        split_out['long_view_association'] = assoc

        # D03 inter-feedback association (binary x binary phi via pearson corr; binary/continuous pearson)
        all_sig = BINARY + CONTINUOUS
        corr = df[all_sig].corr()
        split_out['inter_feedback_correlation'] = corr.round(4).to_dict()

        out[split_name] = split_out

    # D01/D02/D03 variation by user activity tier (validation side, train-derived tiers)
    tiers, counts, edges = C.train_activity_tiers(train, valid['user_id'].unique())
    valid2 = valid.copy()
    valid2['tier'] = valid2['user_id'].map(tiers)
    by_tier = {}
    for t, g in valid2.groupby('tier', observed=True):
        row = {'n_rows': int(len(g)), 'n_users': int(g['user_id'].nunique())}
        for c in BINARY:
            row[f'{c}_mean'] = float(g[c].mean())
        for c in CONTINUOUS:
            row[f'{c}_mean'] = float(g[c].mean())
            row[f'{c}_pct_zero'] = float((g[c] == 0).mean() * 100)
        row['long_view_rate'] = float(g['long_view'].mean())
        by_tier[t] = row
    out['feedback_by_activity_tier_valid'] = by_tier
    out['activity_tier_edges_train_impr_count'] = edges.tolist()

    C.save_json(out, 'phase_d_feedback.json')
    print(json.dumps({k: v for k, v in out.items() if k not in ('train', 'valid')}, indent=2, default=str))

if __name__ == '__main__':
    main()
