"""Phase I -- validation-period random exposure audit.

Only date-only counts are taken for the evaluation period.  No evaluation-row
labels or features are materialized or summarized.
"""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

VALID_LO, VALID_HI = 20220422, 20220428
TEST_LO, TEST_HI = 20220429, 20220508

def main():
    out = {}
    rlog = C.load_random_log()  # validation dates only
    rows_by_period = C.count_random_rows_by_period()
    train = C.load_train_log()
    valid = C.load_valid_log()

    out['row_count_validation_period'] = int(len(rlog))
    out['date_coverage_validation_period'] = {
        'min': int(rlog['date'].min()), 'max': int(rlog['date'].max()),
        'n_unique_dates': int(rlog['date'].nunique())}
    out['rows_by_period'] = {
        'validation_rows': int(rows_by_period['validation']),
        'evaluation_rows_DATE_ONLY': int(rows_by_period['evaluation']),
    }

    out['is_rand_flag_check'] = {
        'pct_rows_is_rand_eq_1': float((rlog['is_rand'] == 1).mean() * 100),
        'standard_train_is_rand_1_pct': float((train['is_rand'] == 1).mean() * 100),
        'standard_valid_is_rand_1_pct': float((valid['is_rand'] == 1).mean() * 100),
    }

    # entity overlap with standard traffic
    std_users = set(train['user_id']) | set(valid['user_id'])
    std_videos = set(train['video_id']) | set(valid['video_id'])
    rlog_users = set(rlog['user_id'])
    rlog_videos = set(rlog['video_id'])
    out['entity_overlap_with_standard'] = {
        'random_users': len(rlog_users),
        'random_videos': len(rlog_videos),
        'random_users_also_in_standard_pct': float(len(rlog_users & std_users) / len(rlog_users) * 100),
        'random_videos_also_in_standard_pct': float(len(rlog_videos & std_videos) / len(rlog_videos) * 100),
        'standard_users_also_in_random_pct': float(len(std_users & rlog_users) / len(std_users) * 100),
        'standard_videos_also_in_random_pct': float(len(std_videos & rlog_videos) / len(std_videos) * 100),
    }

    # user-video pair overlap between random log (validation-period only, safe) and standard valid
    rlog_valid_uv = set(zip(rlog['user_id'], rlog['video_id']))
    std_valid_uv = set(zip(valid['user_id'], valid['video_id']))
    out['uv_pair_overlap_validation_period_only'] = {
        'random_valid_period_uv_pairs': len(rlog_valid_uv),
        'standard_valid_uv_pairs': len(std_valid_uv),
        'shared_uv_pairs': len(rlog_valid_uv & std_valid_uv),
        'pct_of_random_valid_period_shared_with_standard_valid': float(
            len(rlog_valid_uv & std_valid_uv) / len(rlog_valid_uv) * 100) if rlog_valid_uv else 0.0,
    }

    # Descriptive outcome statistics are validation-period only.
    def period_stats(df):
        if len(df) == 0:
            return None
        return {'rows': int(len(df)), 'long_view_rate': float(df['long_view'].mean()),
                'unique_users': int(df['user_id'].nunique()), 'unique_videos': int(df['video_id'].nunique())}

    out['descriptive_by_period'] = {
        'validation_period_random_log': period_stats(rlog),
    }

    # daily breakdown
    daily = rlog.groupby('date').agg(rows=('user_id', 'size'), long_view_rate=('long_view', 'mean'),
                                      uniq_users=('user_id', 'nunique'), uniq_videos=('video_id', 'nunique')).reset_index()
    out['daily_random_log'] = daily.to_dict(orient='records')

    C.save_json(out, 'phase_i_random_log.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'daily_random_log'}, indent=2, default=str))

if __name__ == '__main__':
    main()
