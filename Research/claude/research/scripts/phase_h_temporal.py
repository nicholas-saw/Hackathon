"""Phase H -- temporal structure: daily volume/composition, early vs late train
vs validation, similarity of validation to late-train vs early-train."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def period_summary(df):
    return {
        'rows': int(len(df)),
        'long_view_rate': float(df['long_view'].mean()),
        'unique_users': int(df['user_id'].nunique()),
        'unique_videos': int(df['video_id'].nunique()),
        'mean_duration_ms': float(df['duration_ms'].mean()),
        'mean_play_time_ms': float(df['play_time_ms'].mean()),
        'click_rate': float(df['is_click'].mean()),
        'tab_top3_share_pct': (df['tab'].value_counts(normalize=True).head(3) * 100).round(2).to_dict(),
    }

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()

    dates = sorted(train['date'].unique())
    out['train_date_coverage'] = {
        'official_train_range': [20220408, 20220421],
        'dates_actually_present_in_file': [int(d) for d in dates],
        'n_dates_present': len(dates),
        'n_dates_expected': 14,
        'missing_dates': sorted(set(range(20220408, 20220422)) - set(int(d) for d in dates)),
        'note': 'log_standard_4_08_to_4_21_pure.csv contains zero rows for date 2022-04-08 despite it being the official train start date; the file has 13 distinct dates, not 14.',
    }
    daily_rowcount = train.groupby('date').size()
    out['daily_rowcount_decay'] = {str(int(d)): int(c) for d, c in daily_rowcount.items()}
    n = len(dates)
    early_dates = dates[:n // 2]
    late_dates = dates[n // 2:]

    early = train[train['date'].isin(early_dates)]
    late = train[train['date'].isin(late_dates)]

    out['date_ranges'] = {'early_train': [int(min(early_dates)), int(max(early_dates))],
                           'late_train': [int(min(late_dates)), int(max(late_dates))],
                           'validation': [int(valid['date'].min()), int(valid['date'].max())]}
    out['early_train'] = period_summary(early)
    out['late_train'] = period_summary(late)
    out['validation'] = period_summary(valid)

    # distributional distance: KL-ish comparison via tab distribution, and video-overlap Jaccard
    def jaccard(a, b):
        a, b = set(a), set(b)
        return float(len(a & b) / len(a | b)) if (a or b) else 0.0

    out['video_set_jaccard'] = {
        'early_vs_late_train': jaccard(early['video_id'], late['video_id']),
        'early_vs_valid': jaccard(early['video_id'], valid['video_id']),
        'late_vs_valid': jaccard(late['video_id'], valid['video_id']),
    }
    out['user_set_jaccard'] = {
        'early_vs_late_train': jaccard(early['user_id'], late['user_id']),
        'early_vs_valid': jaccard(early['user_id'], valid['user_id']),
        'late_vs_valid': jaccard(late['user_id'], valid['user_id']),
    }

    # long_view rate distance
    out['long_view_rate_gap'] = {
        'early_train_vs_valid': abs(out['early_train']['long_view_rate'] - out['validation']['long_view_rate']),
        'late_train_vs_valid': abs(out['late_train']['long_view_rate'] - out['validation']['long_view_rate']),
    }

    # per-day full table (train + valid) for a plot / fine-grained look
    both = pd.concat([train.assign(split='train'), valid.assign(split='valid')])
    daily = both.groupby(['split', 'date']).agg(rows=('user_id', 'size'),
                                                  long_view_rate=('long_view', 'mean'),
                                                  uniq_users=('user_id', 'nunique'),
                                                  uniq_videos=('video_id', 'nunique'),
                                                  click_rate=('is_click', 'mean')).reset_index()
    out['daily_full'] = daily.to_dict(orient='records')

    C.save_json(out, 'phase_h_temporal.json')

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for split, marker in [('train', 'o'), ('valid', 's')]:
            d = daily[daily['split'] == split]
            axes[0].plot(d['date'].astype(str), d['rows'], marker=marker, label=split)
            axes[1].plot(d['date'].astype(str), d['long_view_rate'], marker=marker, label=split)
        axes[0].set_ylabel('rows/day'); axes[0].legend(); axes[0].set_title('Daily volume')
        axes[1].set_ylabel('long_view rate'); axes[1].set_title('Daily long_view rate')
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        path = os.path.join(C.PLOTS_DIR, 'phase_h_temporal.png')
        os.makedirs(C.PLOTS_DIR, exist_ok=True)
        plt.savefig(path, dpi=120)
        print('saved plot to', path)
    except Exception as e:
        print('plot failed:', e)

    print(json.dumps({k: v for k, v in out.items() if k != 'daily_full'}, indent=2, default=str))

if __name__ == '__main__':
    main()
