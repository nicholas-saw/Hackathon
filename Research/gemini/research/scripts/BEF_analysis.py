import os
import sys
import pandas as pd
import numpy as np

sys.path.append("../../source/starter-kit")
from data import load, encode
from baseline import run_pop, run_fm, evaluate

DATA_DIR = "../../source/KuaiRand-Pure/data"

def analyze():
    print("Loading standard logs...")
    splits = load(DATA_DIR)
    # splits is a dict: 'train', 'valid', 'test'
    # Each row is (date, user_id, video_id, author_id, tab, duration_ms, long_view_0_or_1)
    
    df_train = pd.DataFrame(splits['train'], columns=['date', 'user_id', 'video_id', 'author_id', 'tab', 'duration_ms', 'long_view'])
    df_valid = pd.DataFrame(splits['valid'], columns=['date', 'user_id', 'video_id', 'author_id', 'tab', 'duration_ms', 'long_view'])
    
    # ------------------------------------------------------------
    # PHASE B — METRIC STRUCTURE
    # ------------------------------------------------------------
    print("\n================== B Metric Structure ==================")
    user_train_counts = df_train.groupby('user_id').size()
    user_train_counts.name = 'train_count'
    
    user_valid_counts = df_valid.groupby('user_id').size()
    user_valid_counts.name = 'valid_count'
    
    # Activity buckets
    def get_bucket(c):
        if pd.isna(c) or c == 0: return 'Cold'
        if c < 10: return 'T1 (<10)'
        if c < 50: return 'T2 (10-49)'
        if c < 150: return 'T3 (50-149)'
        return 'T4 (150+)'
        
    buckets = user_train_counts.apply(get_bucket)
    df_valid['bucket'] = df_valid['user_id'].map(buckets).fillna('Cold')
    
    # List lengths in validation
    print("\nList length distribution in validation:")
    list_lengths = df_valid.groupby('user_id').size()
    def get_len_bucket(l):
        if l == 1: return '1'
        if l <= 3: return '2-3'
        if l <= 5: return '4-5'
        if l <= 10: return '6-10'
        if l <= 20: return '11-20'
        return '21+'
    len_buckets = list_lengths.apply(get_len_bucket)
    print(len_buckets.value_counts().sort_index())
    
    # Uniform-label users in validation
    pos_rates = df_valid.groupby('user_id')['long_view'].mean()
    all_neg = (pos_rates == 0).sum()
    all_pos = (pos_rates == 1).sum()
    mixed = ((pos_rates > 0) & (pos_rates < 1)).sum()
    print(f"\nAll-negative users: {all_neg} ({all_neg/len(pos_rates):.2%})")
    print(f"All-positive users: {all_pos} ({all_pos/len(pos_rates):.2%})")
    print(f"Mixed users: {mixed} ({mixed/len(pos_rates):.2%})")
    print(f"Single-impression users: {(list_lengths == 1).sum()} ({(list_lengths == 1).sum()/len(list_lengths):.2%})")
    
    # GAUC by bucket: run FM on valid to get predictions
    print("\nTraining FM to get predictions for buckets...")
    # we don't want to pollute with full 40 epochs if we just want a rough bucket estimate, 
    # but the instructions say "use baseline predictions" implicitly. We'll run 5 epochs.
    res = run_fm(splits, epochs=12, verbose=False)
    # Actually wait, `run_fm` returns metrics, but doesn't return predictions per user.
    # Let's just do it manually
    print("Done. To get predictions per row, I will use popularity.")
    pop_res = run_pop(splits)
    
    # Let's get actual predictions
    from baseline import FM
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    m = FM(dim, k=16, lr=0.001)
    for _ in range(12):
        for i in range(0, len(ytr), 8192):
            m.step(Xtr[i:i+8192], ytr[i:i+8192])
    preds = m.predict(Xva)
    df_valid['pred'] = preds
    
    # Eval per bucket
    print("\nEval per activity bucket:")
    for b in ['Cold', 'T1 (<10)', 'T2 (10-49)', 'T3 (50-149)', 'T4 (150+)']:
        d = df_valid[df_valid['bucket'] == b]
        if len(d) == 0: continue
        users = d['user_id'].tolist()
        labels = d['long_view'].tolist()
        scores = d['pred'].tolist()
        ev = evaluate(users, labels, scores)
        
        # Invariant users
        u_rates = d.groupby('user_id')['long_view'].mean()
        inv = ((u_rates == 0) | (u_rates == 1)).sum() / len(u_rates)
        
        # Positives
        pos = d['long_view'].sum()
        
        print(f"Bucket {b:10s} | Users: {len(u_rates):5d} | Rows: {len(d):6d} | GAUC: {ev['GAUC']:.4f} | nDCG@5: {ev['nDCG@5']:.4f} | Fixed: {inv:.2%} | Positives (GAUC wgt): {pos}")
        
    print("\nEval per list-length bucket:")
    df_valid['len_bucket'] = df_valid['user_id'].map(len_buckets)
    for b in ['1', '2-3', '4-5', '6-10', '11-20', '21+']:
        d = df_valid[df_valid['len_bucket'] == b]
        if len(d) == 0: continue
        users = d['user_id'].tolist()
        labels = d['long_view'].tolist()
        scores = d['pred'].tolist()
        ev = evaluate(users, labels, scores)
        
        # Oracle nDCG@5
        oracle_ev = evaluate(users, labels, labels)
        
        pos = d['long_view'].sum()
        print(f"Len {b:5s} | Users: {len(d['user_id'].unique()):5d} | Rows: {len(d):6d} | GAUC: {ev['GAUC']:.4f} | nDCG@5: {ev['nDCG@5']:.4f} | Oracle nDCG@5: {oracle_ev['nDCG@5']:.4f} | Positives: {pos}")

    # ------------------------------------------------------------
    # PHASE E — POST-IMPRESSION FEEDBACK PROFILE
    # ------------------------------------------------------------
    print("\n================== E Feedback Profile ==================")
    # Reload raw train and valid to get feedback
    df_train_raw = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval_raw = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    df_valid_raw = df_eval_raw[(df_eval_raw['date'] >= 20220422) & (df_eval_raw['date'] <= 20220428)]
    
    signals = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter']
    
    print("\nPrevalence/Mean in Train & Valid:")
    for sig in signals:
        tm = df_train_raw[sig].mean()
        vm = df_valid_raw[sig].mean()
        # corr with long_view
        if sig == 'play_time_ms':
            c = df_valid_raw[sig].corr(df_valid_raw['long_view'])
        else:
            c = df_valid_raw[sig].astype(float).corr(df_valid_raw['long_view'].astype(float))
        print(f"{sig:20s} | Train: {tm:.4f} | Valid: {vm:.4f} | Corr with long_view: {c:.4f}")

    # ------------------------------------------------------------
    # PHASE F — HISTORICAL INFORMATION AVAILABILITY
    # ------------------------------------------------------------
    # Review note: this block previously duplicated F_history.py's logic with a
    # bug (Cold-bucket users showed a nonzero "Rep Video" rate despite having zero
    # prior train interactions by definition, and the loop crashed with a
    # ZeroDivisionError on the next bucket -- see research/review_artifacts/
    # BEF_rerun_output.txt for the reproduced traceback). F_history.py is the
    # correct, working, standalone implementation of this analysis and is what
    # actually produced the F01 numbers in PRE_AUDIT.md / data_profile.md.
    # Removed here rather than fixed in place, to avoid two drifting copies of
    # the same logic. Run F_history.py for Phase F results.
    print("\n================== F Historical Availability ==================")
    print("See F_history.py for the historical-availability breakdown (this duplicate block was removed on review).")

if __name__ == "__main__":
    analyze()

