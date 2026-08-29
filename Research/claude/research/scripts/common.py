"""Shared data loading for the KuaiRand-Pure pre-audit.

Hard rule: this module never returns evaluation/test-period rows. The second
standard log file spans both validation and test dates; we read it once and
immediately discard (never return, never inspect) rows outside the
validation date range as a byproduct of the date filter, exactly as
data.py's official splitter does internally. No function here exposes a
'test' split.
"""
import csv
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..',
                                          'source', 'KuaiRand-Pure', 'data'))

TRAIN_LO, TRAIN_HI = 20220408, 20220421
VALID_LO, VALID_HI = 20220422, 20220428
TEST_LO, TEST_HI = 20220429, 20220508  # never used to filter-in; documented only

LOG_TRAIN_FILE = 'log_standard_4_08_to_4_21_pure.csv'
LOG_VALID_TEST_FILE = 'log_standard_4_22_to_5_08_pure.csv'
LOG_RANDOM_FILE = 'log_random_4_22_to_5_08_pure.csv'
USER_FEATURES_FILE = 'user_features_pure.csv'
VIDEO_BASIC_FILE = 'video_features_basic_pure.csv'
VIDEO_STAT_FILE = 'video_features_statistic_pure.csv'

FEEDBACK_COLS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
                  'is_hate', 'play_time_ms', 'profile_stay_time',
                  'comment_stay_time', 'is_profile_enter']


def _path(name):
    return os.path.join(DATA_DIR, name)


def load_train_log():
    df = pd.read_csv(_path(LOG_TRAIN_FILE))
    assert df['date'].min() >= TRAIN_LO and df['date'].max() <= TRAIN_HI, \
        "train log file contains rows outside the official train date range"
    return df.reset_index(drop=True)


def load_valid_log():
    """Return validation rows without materializing evaluation-row fields.

    The shared file must be scanned, but only ``date`` is inspected for rows
    outside the validation range.  Complete evaluation rows (including their
    labels) are never stored in a DataFrame.
    """
    return _load_date_filtered(LOG_VALID_TEST_FILE, VALID_LO, VALID_HI)


def _load_date_filtered(name, lo, hi):
    kept = []
    with open(_path(name), newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_i = header.index('date')
        for row in reader:
            date = int(row[date_i])
            if lo <= date <= hi:
                kept.append(row)
    # Interaction logs are numeric throughout.
    return pd.DataFrame(kept, columns=header).apply(pd.to_numeric).reset_index(drop=True)


def count_test_rows_only():
    """Returns ONLY the integer count of test-range rows, for row-count
    confirmation against the published split sizes. No labels or features
    from these rows are returned or retained."""
    return _count_date_range(LOG_VALID_TEST_FILE, TEST_LO, TEST_HI)


def _count_date_range(name, lo, hi):
    n = 0
    with open(_path(name), newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        date_i = header.index('date')
        for row in reader:
            date = int(row[date_i])
            n += int(lo <= date <= hi)
    return n


def load_user_features():
    return pd.read_csv(_path(USER_FEATURES_FILE))


def load_video_basic():
    return pd.read_csv(_path(VIDEO_BASIC_FILE))


def load_video_stat():
    return pd.read_csv(_path(VIDEO_STAT_FILE))


def load_random_log():
    """Return only the validation-date slice of the random-exposure log."""
    return _load_date_filtered(LOG_RANDOM_FILE, VALID_LO, VALID_HI)


def count_random_rows_by_period():
    """Return date-only row counts; never accesses random-log outcomes."""
    return {
        'validation': _count_date_range(LOG_RANDOM_FILE, VALID_LO, VALID_HI),
        'evaluation': _count_date_range(LOG_RANDOM_FILE, TEST_LO, TEST_HI),
    }


def dur_bucket_edges(train_df, n=10):
    return np.quantile(train_df['duration_ms'].to_numpy(), np.linspace(0, 1, n + 1)[1:-1])


def add_dur_bucket(df, edges):
    df = df.copy()
    df['dur_bucket'] = np.searchsorted(edges, df['duration_ms'].to_numpy())
    return df


def add_author(df, video_basic):
    vid2author = video_basic.set_index('video_id')['author_id']
    df = df.copy()
    df['author_id'] = df['video_id'].map(vid2author)
    df['author_id'] = df['author_id'].fillna(-1).astype(np.int64)
    return df


RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'experiment_results'))
PLOTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


def train_activity_tiers(train_df, valid_user_ids):
    """Assign each validation user a tier based on their TRAIN-side impression
    count only (never valid-side activity). Cold = 0 train impressions.
    T1..T4 = quartiles of train-impression-count among valid users who do
    have >=1 train impression. Returns (tier_series indexed by user_id, edges)."""
    train_counts = train_df.groupby('user_id').size()
    uids = pd.Series(sorted(set(valid_user_ids)))
    counts = uids.map(train_counts).fillna(0).astype(int)
    counts.index = uids.values
    warm_counts = counts[counts > 0]
    edges = np.quantile(warm_counts.to_numpy(), [0.25, 0.5, 0.75]) if len(warm_counts) else np.array([0, 0, 0])

    def tier_of(c):
        if c == 0:
            return 'Cold'
        if c <= edges[0]:
            return 'T1'
        if c <= edges[1]:
            return 'T2'
        if c <= edges[2]:
            return 'T3'
        return 'T4'

    tiers = counts.map(tier_of)
    return tiers, counts, edges


def save_json(obj, name):
    import json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"wrote {path}")
    return path
