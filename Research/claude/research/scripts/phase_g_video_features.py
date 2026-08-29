"""Phase G -- video basic / statistical feature inventory, redundancy,
aggregation-window inference, and ratio-feature associations (safety unresolved) with
long_view (train+validation only)."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

STAT_COLS = ['show_cnt', 'show_user_num', 'play_cnt', 'play_user_num', 'play_duration',
             'complete_play_cnt', 'valid_play_cnt', 'long_time_play_cnt', 'short_time_play_cnt',
             'play_progress', 'like_cnt', 'comment_cnt', 'follow_cnt', 'share_cnt',
             'download_cnt', 'collect_cnt']

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()
    vbasic = C.load_video_basic()
    vstat = C.load_video_stat()

    # ---- video_basic inventory ----
    basic_inv = {}
    for c in ['video_type', 'upload_type', 'visible_status', 'music_type']:
        vc = vbasic[c].astype(str).value_counts(normalize=True).head(10) * 100
        basic_inv[c] = {'n_unique': int(vbasic[c].nunique(dropna=False)), 'top_pct': vc.round(2).to_dict()}
    for c in ['video_duration', 'server_width', 'server_height']:
        s = vbasic[c].dropna()
        basic_inv[c] = {'min': float(s.min()), 'median': float(s.median()), 'p90': float(s.quantile(0.9)),
                         'max': float(s.max()), 'missing_pct': float(vbasic[c].isna().mean() * 100)}
    basic_inv['tag_cardinality'] = int(vbasic['tag'].nunique(dropna=False))
    basic_inv['music_id_cardinality'] = int(vbasic['music_id'].nunique(dropna=False))
    out['video_basic_inventory'] = basic_inv

    # tab / dur_bucket usefulness (marginal long_view rate spread across categories, train)
    edges = C.dur_bucket_edges(train)
    train_b = C.add_dur_bucket(train, edges)
    tab_rate = train_b.groupby('tab')['long_view'].agg(['mean', 'count'])
    dur_rate = train_b.groupby('dur_bucket')['long_view'].agg(['mean', 'count'])
    out['tab_long_view_rate_train'] = tab_rate.reset_index().to_dict(orient='records')
    out['dur_bucket_long_view_rate_train'] = dur_rate.reset_index().to_dict(orient='records')

    # ---- video_stat inventory ----
    stat_inv = {}
    for c in STAT_COLS + ['counts']:
        s = vstat[c]
        stat_inv[c] = {'missing_pct': float(s.isna().mean() * 100), 'min': float(s.min()),
                        'median': float(s.median()), 'mean': float(s.mean()),
                        'p90': float(s.quantile(0.9)), 'max': float(s.max())}
    out['video_stat_inventory'] = stat_inv

    # redundancy among stats (correlation matrix)
    corr = vstat[STAT_COLS].corr()
    out['video_stat_correlation'] = corr.round(3).to_dict()

    # ---- aggregation-window inference ----
    # hypothesis: show_cnt (and friends) are PER-'counts' averages, i.e.
    # reconstructed_total = show_cnt * counts should be ~integer and should
    # relate to actual observed impressions in our logs.
    recon_total_show = vstat['show_cnt'] * vstat['counts']
    frac = np.abs(recon_total_show - np.round(recon_total_show))
    out['aggregation_window_inference'] = {
        'pct_videos_where_show_cnt_times_counts_is_near_integer': float((frac < 0.05).mean() * 100),
        'counts_field_summary': {'min': float(vstat['counts'].min()), 'median': float(vstat['counts'].median()),
                                  'max': float(vstat['counts'].max())},
    }

    observed_impr = pd.concat([train[['video_id']], valid[['video_id']]]).groupby('video_id').size()
    observed_impr.name = 'observed_train_valid_impressions'
    merged = vstat[['video_id', 'show_cnt', 'counts']].copy()
    merged['recon_total_show'] = merged['show_cnt'] * merged['counts']
    merged = merged.set_index('video_id').join(observed_impr, how='inner')
    merged = merged[merged['observed_train_valid_impressions'] > 0]
    ratio = merged['recon_total_show'] / merged['observed_train_valid_impressions']
    out['aggregation_window_inference'].update({
        'n_videos_compared': int(len(merged)),
        'ratio_recon_over_observed_median': float(ratio.median()),
        'ratio_recon_over_observed_p10': float(ratio.quantile(0.1)),
        'ratio_recon_over_observed_p90': float(ratio.quantile(0.9)),
        'pct_videos_recon_less_than_observed': float((merged['recon_total_show'] < merged['observed_train_valid_impressions']).mean() * 100),
        'interpretation_note': ('If ratio >> 1 broadly, the statistic file aggregation window covers substantially '
                                 'more traffic/time than train+valid standard logs alone (e.g. also includes random-log '
                                 'traffic and/or the evaluation period and/or a longer external window). If ratio ~ 1 the '
                                 'window may correspond closely to train+valid standard logs. This is inferred, not documented.'),
    })

    # ---- ratio features vs long_view (VALID only; causal safety unresolved) ----
    alpha, beta = 1.0, 20.0
    vs = vstat.set_index('video_id')
    ratios = pd.DataFrame(index=vs.index)
    ratios['long_view_ratio'] = (vs['long_time_play_cnt'] + alpha) / (vs['show_cnt'] + beta)
    ratios['play_ratio'] = (vs['play_cnt'] + alpha) / (vs['show_cnt'] + beta)
    ratios['like_ratio'] = (vs['like_cnt'] + alpha) / (vs['show_cnt'] + beta)
    ratios['complete_ratio'] = (vs['complete_play_cnt'] + alpha) / (vs['show_cnt'] + beta)

    valid_m = valid.copy()
    for col in ratios.columns:
        valid_m[col] = valid_m['video_id'].map(ratios[col])

    ratio_assoc = {}
    for col in ratios.columns:
        r = valid_m[col].corr(valid_m['long_view'])
        ratio_assoc[col] = {'pearson_r_with_long_view_valid': float(r)}
        try:
            q = pd.qcut(valid_m[col], 5, duplicates='drop')
            by_q = valid_m.groupby(q, observed=True)['long_view'].mean()
            ratio_assoc[col]['long_view_rate_by_quintile'] = {str(k): float(v) for k, v in by_q.items()}
        except Exception as e:
            ratio_assoc[col]['quintile_error'] = str(e)
    out['ratio_feature_association_valid'] = ratio_assoc

    # missingness of video_id join (videos in logs but absent from stat file)
    valid_video_ids = set(valid['video_id'])
    stat_video_ids = set(vstat['video_id'])
    out['video_stat_coverage'] = {
        'pct_valid_videos_missing_from_stat_file': float(len(valid_video_ids - stat_video_ids) / len(valid_video_ids) * 100),
    }

    C.save_json(out, 'phase_g_video_features.json')
    print(json.dumps(out['aggregation_window_inference'], indent=2, default=str))
    print(json.dumps(out['ratio_feature_association_valid'], indent=2, default=str)[:3000])

if __name__ == '__main__':
    main()
