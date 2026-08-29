"""Phase A -- dataset structure: cardinalities, missingness, train/valid overlap,
author/video redundancy, repeat-pair frequency, temporal volume."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def main():
    out = {}

    train = C.load_train_log()
    valid = C.load_valid_log()
    vbasic = C.load_video_basic()
    vstat = C.load_video_stat()
    ufeat = C.load_user_features()
    test_rows = C.count_test_rows_only()

    out['row_counts'] = {'train': len(train), 'valid': len(valid), 'test': test_rows}

    # ---- A01 cardinalities ----
    train_a = C.add_author(train, vbasic)
    valid_a = C.add_author(valid, vbasic)

    def card(s):
        return int(s.nunique())

    out['cardinalities'] = {
        'train_users': card(train_a['user_id']),
        'valid_users': card(valid_a['user_id']),
        'train_videos': card(train_a['video_id']),
        'valid_videos': card(valid_a['video_id']),
        'train_authors': card(train_a['author_id']),
        'valid_authors': card(valid_a['author_id']),
        'train_tabs': card(train_a['tab']),
        'valid_tabs': card(valid_a['tab']),
        'total_users_in_user_features': card(ufeat['user_id']),
        'total_videos_in_video_basic': card(vbasic['video_id']),
        'video_basic_tag_unique': card(vbasic['tag'].astype(str)),
        'video_basic_music_id_unique': card(vbasic['music_id'].astype(str)),
        'video_basic_video_type_unique': card(vbasic['video_type'].astype(str)),
        'video_basic_upload_type_unique': card(vbasic['upload_type'].astype(str)),
        'user_active_degree_unique': card(ufeat['user_active_degree'].astype(str)),
    }

    # ---- A02 missingness ----
    def miss_pct(df, cols=None):
        cols = cols if cols is not None else df.columns
        n = len(df)
        return {c: float(df[c].isna().mean() * 100) for c in cols}

    out['missingness'] = {
        'log_train': miss_pct(train),
        'log_valid': miss_pct(valid),
        'video_basic': miss_pct(vbasic),
        'video_stat': miss_pct(vstat),
        'user_features': miss_pct(ufeat),
    }
    # video_id present in basic file but missing author mapping in log (UNK-equivalent)
    out['missingness']['train_video_missing_author_pct'] = float(train_a['author_id'].eq(-1).mean() * 100)
    out['missingness']['valid_video_missing_author_pct'] = float(valid_a['author_id'].eq(-1).mean() * 100)

    # ---- A03 train -> validation entity overlap ----
    tr_users, va_users = set(train_a['user_id']), set(valid_a['user_id'])
    tr_videos, va_videos = set(train_a['video_id']), set(valid_a['video_id'])
    tr_authors, va_authors = set(train_a['author_id']), set(valid_a['author_id'])
    tr_uv = set(zip(train_a['user_id'], train_a['video_id']))
    va_uv = set(zip(valid_a['user_id'], valid_a['video_id']))
    tr_ua = set(zip(train_a['user_id'], train_a['author_id']))
    va_ua = set(zip(valid_a['user_id'], valid_a['author_id']))

    def overlap_pct(va_set, tr_set):
        if len(va_set) == 0:
            return 0.0
        return float(len(va_set & tr_set) / len(va_set) * 100)

    out['overlap'] = {
        'valid_users_seen_in_train_pct': overlap_pct(va_users, tr_users),
        'valid_videos_seen_in_train_pct': overlap_pct(va_videos, tr_videos),
        'valid_authors_seen_in_train_pct': overlap_pct(va_authors, tr_authors),
        'valid_user_video_pairs_seen_in_train_pct': overlap_pct(va_uv, tr_uv),
        'valid_user_author_pairs_seen_in_train_pct': overlap_pct(va_ua, tr_ua),
        'cold_valid_users_count': len(va_users - tr_users),
        'cold_valid_users_pct': float(len(va_users - tr_users) / len(va_users) * 100),
        'cold_valid_videos_count': len(va_videos - tr_videos),
        'cold_valid_videos_pct': float(len(va_videos - tr_videos) / len(va_videos) * 100),
        'cold_valid_authors_count': len(va_authors - tr_authors),
        'cold_valid_authors_pct': float(len(va_authors - tr_authors) / len(va_authors) * 100),
    }

    # tag overlap (user-tag pairs), using video_basic tag
    vid2tag = vbasic.set_index('video_id')['tag']
    # Treat missing tag as an explicit categorical value.  The prior version
    # used astype(str) for overlap but pandas' default dropna=True for repeat
    # counts, producing inconsistent denominators across A03 and A05.
    train_a['tag'] = train_a['video_id'].map(vid2tag).fillna('__MISSING_TAG__')
    valid_a['tag'] = valid_a['video_id'].map(vid2tag).fillna('__MISSING_TAG__')
    tr_ut = set(zip(train_a['user_id'], train_a['tag'].astype(str)))
    va_ut = set(zip(valid_a['user_id'], valid_a['tag'].astype(str)))
    out['overlap']['valid_user_tag_pairs_seen_in_train_pct'] = overlap_pct(va_ut, tr_ut)

    # cold-video / cold-user IMPRESSION share (row-level, not entity-level)
    out['overlap']['valid_rows_with_cold_user_pct'] = float(valid_a['user_id'].isin(va_users - tr_users).mean() * 100)
    out['overlap']['valid_rows_with_cold_video_pct'] = float(valid_a['video_id'].isin(va_videos - tr_videos).mean() * 100)

    # ---- A04 author/video redundancy ----
    vids_per_author = vbasic.groupby('author_id')['video_id'].nunique()
    out['author_video_structure'] = {
        'videos_per_author_median': float(vids_per_author.median()),
        'videos_per_author_mean': float(vids_per_author.mean()),
        'videos_per_author_p90': float(vids_per_author.quantile(0.9)),
        'videos_per_author_max': int(vids_per_author.max()),
        'authors_total': int(vids_per_author.shape[0]),
        'authors_with_exactly_1_video': int((vids_per_author == 1).sum()),
        'authors_with_exactly_1_video_pct': float((vids_per_author == 1).mean() * 100),
    }
    # in TRAIN LOG impressions, how often is author_id fully determined by video_id
    # (always true by construction, so instead measure: given author_id, how many
    # distinct videos from that author actually appear in train impressions)
    vids_per_author_in_train_log = train_a.groupby('author_id')['video_id'].nunique()
    out['author_video_structure']['authors_in_train_log_with_1_distinct_video_impression_pct'] = float(
        (vids_per_author_in_train_log == 1).mean() * 100)

    # ---- A05 repeat-pair / affinity coverage (within-split repeat frequency) ----
    def repeat_stats(df, key_cols, label):
        counts = df.groupby(key_cols).size()
        return {
            f'{label}_unique_pairs': int(counts.shape[0]),
            f'{label}_rows': int(counts.sum()),
            f'{label}_mean_repeat': float(counts.mean()),
            f'{label}_median_repeat': float(counts.median()),
            f'{label}_pct_pairs_seen_more_than_once': float((counts > 1).mean() * 100),
            f'{label}_pct_rows_in_repeated_pairs': float(counts[counts > 1].sum() / counts.sum() * 100) if counts.sum() else 0.0,
        }

    out['repeat_frequency_train'] = {}
    out['repeat_frequency_train'].update(repeat_stats(train_a, ['user_id', 'video_id'], 'user_video'))
    out['repeat_frequency_train'].update(repeat_stats(train_a, ['user_id', 'author_id'], 'user_author'))
    out['repeat_frequency_train'].update(repeat_stats(train_a, ['user_id', 'tag'], 'user_tag'))

    out['repeat_frequency_valid'] = {}
    out['repeat_frequency_valid'].update(repeat_stats(valid_a, ['user_id', 'video_id'], 'user_video'))
    out['repeat_frequency_valid'].update(repeat_stats(valid_a, ['user_id', 'author_id'], 'user_author'))
    out['repeat_frequency_valid'].update(repeat_stats(valid_a, ['user_id', 'tag'], 'user_tag'))

    # duplicate (user_id, video_id) pairs in VALID specifically (README notes 3.06% in test)
    uv_counts_valid = valid_a.groupby(['user_id', 'video_id']).size()
    out['duplicate_uv_pairs_valid'] = {
        'pct_pairs_duplicated': float((uv_counts_valid > 1).mean() * 100),
        'max_repeat': int(uv_counts_valid.max()),
    }

    # ---- A06 temporal interaction volume (daily) ----
    daily_train = train.groupby('date').agg(rows=('user_id', 'size'),
                                             long_view_rate=('long_view', 'mean'),
                                             uniq_users=('user_id', 'nunique'),
                                             uniq_videos=('video_id', 'nunique')).reset_index()
    daily_valid = valid.groupby('date').agg(rows=('user_id', 'size'),
                                             long_view_rate=('long_view', 'mean'),
                                             uniq_users=('user_id', 'nunique'),
                                             uniq_videos=('video_id', 'nunique')).reset_index()
    out['daily_train'] = daily_train.to_dict(orient='records')
    out['daily_valid'] = daily_valid.to_dict(orient='records')

    # long_view overall rate
    out['long_view_rate'] = {'train': float(train['long_view'].mean()),
                              'valid': float(valid['long_view'].mean())}

    C.save_json(out, 'phase_a_structure.json')

    print(json.dumps({k: v for k, v in out.items() if k not in ('daily_train', 'daily_valid')}, indent=2)[:6000])

if __name__ == '__main__':
    main()
