"""Phase F -- historical information availability.

Because official train dates (2022-04-08..21) strictly precede official
validation dates (2022-04-22..28), EVERY train impression of a user is
strictly prior to EVERY validation impression of that same user. So
"prior train interactions" for a validation row == that user's total train
impression count. This script measures coverage/density of that history,
NOT whether/how to use it as a feature."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

FEEDBACK = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate']

def pct_ge(s, n):
    return float((s >= n).mean() * 100)

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()
    vbasic = C.load_video_basic()

    # attach tag/author to train and valid for repeat-coverage checks
    vid2author = vbasic.set_index('video_id')['author_id']
    vid2tag = vbasic.set_index('video_id')['tag']
    train = train.copy(); valid = valid.copy()
    train['author_id'] = train['video_id'].map(vid2author)
    valid['author_id'] = valid['video_id'].map(vid2author)
    train['tag'] = train['video_id'].map(vid2tag).fillna('__MISSING_TAG__')
    valid['tag'] = valid['video_id'].map(vid2tag).fillna('__MISSING_TAG__')

    valid_users = valid['user_id'].unique()
    tiers, train_counts, edges = C.train_activity_tiers(train, valid_users)

    # ---- overall proportions with >=1/5/10 prior interactions ----
    counts_for_valid_users = train_counts  # indexed by user_id, already restricted to valid users
    out['overall'] = {
        'n_valid_users': int(len(counts_for_valid_users)),
        'pct_ge_1_prior_train_interaction': pct_ge(counts_for_valid_users, 1),
        'pct_ge_5_prior_train_interactions': pct_ge(counts_for_valid_users, 5),
        'pct_ge_10_prior_train_interactions': pct_ge(counts_for_valid_users, 10),
        'median_prior_interactions': float(counts_for_valid_users.median()),
        'mean_prior_interactions': float(counts_for_valid_users.mean()),
        'p90_prior_interactions': float(counts_for_valid_users.quantile(0.9)),
    }

    # per-feedback-signal prior counts (sum of each signal within train, per user)
    per_sig = {}
    train_user_feedback_sums = train.groupby('user_id')[FEEDBACK].sum()
    train_user_playtime_avail = train.groupby('user_id')['play_time_ms'].apply(lambda s: (s > 0).sum())
    for sig in FEEDBACK:
        s = pd.Series(index=counts_for_valid_users.index, dtype=float)
        s[:] = counts_for_valid_users.index.to_series().map(train_user_feedback_sums[sig]).fillna(0).to_numpy()
        per_sig[sig] = {
            'pct_ge_1_prior': pct_ge(s, 1),
            'pct_ge_5_prior': pct_ge(s, 5),
            'pct_ge_10_prior': pct_ge(s, 10),
            'median_prior': float(s.median()),
            'mean_prior': float(s.mean()),
        }
    s_pt = counts_for_valid_users.index.to_series().map(train_user_playtime_avail).fillna(0)
    per_sig['play_time_ms_nonzero_rows'] = {
        'pct_ge_1_prior': pct_ge(s_pt, 1),
        'pct_ge_5_prior': pct_ge(s_pt, 5),
        'pct_ge_10_prior': pct_ge(s_pt, 10),
        'median_prior': float(s_pt.median()),
        'mean_prior': float(s_pt.mean()),
    }
    out['per_feedback_signal_prior'] = per_sig

    # ---- by activity tier ----
    by_tier = {}
    tiers_df = pd.DataFrame({'user_id': counts_for_valid_users.index, 'tier': tiers.values,
                              'count': counts_for_valid_users.values})
    for t, g in tiers_df.groupby('tier', observed=True):
        c = g['count']
        by_tier[t] = {
            'n_users': int(len(g)),
            'median_prior_rows': float(c.median()),
            'pct_ge_1': pct_ge(c, 1), 'pct_ge_5': pct_ge(c, 5), 'pct_ge_10': pct_ge(c, 10),
        }
    out['by_activity_tier'] = by_tier

    # ---- row-level coverage: prior same video / author / tag ----
    train_user_videos = train.groupby('user_id')['video_id'].apply(set)
    train_user_authors = train.groupby('user_id')['author_id'].apply(set)
    train_user_tags = train.groupby('user_id')['tag'].apply(set)

    def row_covered(row, mapping, key):
        s = mapping.get(row['user_id'])
        if s is None:
            return False
        return row[key] in s

    # vectorized-ish via merge for speed
    valid_video_hit = valid.apply(lambda r: row_covered(r, train_user_videos, 'video_id'), axis=1)
    valid_author_hit = valid.apply(lambda r: row_covered(r, train_user_authors, 'author_id'), axis=1)
    valid_tag_hit = valid.apply(lambda r: row_covered(r, train_user_tags, 'tag'), axis=1)

    out['row_level_repeat_coverage_valid'] = {
        'pct_rows_prior_same_video': float(valid_video_hit.mean() * 100),
        'pct_rows_prior_same_author': float(valid_author_hit.mean() * 100),
        'pct_rows_prior_same_tag': float(valid_tag_hit.mean() * 100),
    }

    # by tier as well
    valid_t = valid.copy()
    valid_t['tier'] = valid_t['user_id'].map(tiers)
    valid_t['video_hit'] = valid_video_hit.values
    valid_t['author_hit'] = valid_author_hit.values
    valid_t['tag_hit'] = valid_tag_hit.values
    by_tier_repeat = {}
    for t, g in valid_t.groupby('tier', observed=True):
        by_tier_repeat[t] = {
            'pct_rows_prior_same_video': float(g['video_hit'].mean() * 100),
            'pct_rows_prior_same_author': float(g['author_hit'].mean() * 100),
            'pct_rows_prior_same_tag': float(g['tag_hit'].mean() * 100),
        }
    out['row_level_repeat_coverage_by_tier'] = by_tier_repeat

    # ---- bonus: within-validation chronological history (not requested but relevant) ----
    # Rows sharing an identical timestamp are simultaneous, not prior to one
    # another.  Compute history before each timestamp group, then broadcast it
    # back to all rows in that group.
    time_key = ['user_id', 'date', 'hourmin', 'time_ms']
    time_groups = valid.groupby(time_key, dropna=False).size().rename('rows_at_time').reset_index()
    time_groups = time_groups.sort_values(time_key)
    time_groups['strict_prior_count'] = (
        time_groups.groupby('user_id')['rows_at_time'].cumsum() - time_groups['rows_at_time'])
    within_valid = valid.merge(time_groups[time_key + ['strict_prior_count']], on=time_key, how='left')
    out['within_validation_sequential_note'] = {
        'pct_valid_rows_with_ge1_strictly_earlier_valid_row_same_user': float(
            (within_valid['strict_prior_count'] >= 1).mean() * 100),
        'pct_rows_in_nonunique_user_timestamp_groups': float(
            valid.duplicated(time_key, keep=False).mean() * 100),
        'explanation': 'Strictly earlier means an earlier (date, hourmin, time_ms) tuple; rows tied at the same timestamp do not count one another as history. This is only an availability diagnostic, not a validated feature protocol.',
    }

    out['activity_tier_edges'] = edges.tolist()
    C.save_json(out, 'phase_f_history.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'per_feedback_signal_prior'}, indent=2, default=str)[:4000])

if __name__ == '__main__':
    main()
