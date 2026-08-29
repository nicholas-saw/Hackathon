import os
import pandas as pd
import numpy as np
import time
import sys

DATA_DIR = "../../source/KuaiRand-Pure/data"

def run_cgi():
    # ------------------------------------------------------------
    # PHASE G — VIDEO STATISTIC FEATURES
    # ------------------------------------------------------------
    print("\n================== G Video Statistics ==================")
    df_stat = pd.read_csv(os.path.join(DATA_DIR, "video_features_statistic_pure.csv"))
    print("Columns:", df_stat.columns.tolist())
    
    print("\nMissingness in video statistics:")
    miss = df_stat.isnull().mean()
    print(miss[miss > 0])
    
    print("\nDistribution of some key metrics:")
    metrics = ['show_cnt', 'play_cnt', 'long_time_play_cnt', 'like_cnt', 'comment_cnt', 'follow_cnt', 'share_cnt']
    print(df_stat[metrics].describe(percentiles=[.5, .9, .99]))
    
    # ------------------------------------------------------------
    # PHASE I — RANDOM EXPOSURE LOG
    # ------------------------------------------------------------
    print("\n================== I Random Exposure ==================")
    df_rand = pd.read_csv(os.path.join(DATA_DIR, "log_random_4_22_to_5_08_pure.csv"))
    print(f"Total rows: {len(df_rand)}")
    print(f"Dates covered: {df_rand['date'].min()} to {df_rand['date'].max()}")
    print(f"Unique users: {df_rand['user_id'].nunique()}")
    print(f"Unique videos: {df_rand['video_id'].nunique()}")
    
    # Check overlap with evaluation period (April 29 - May 8)
    eval_mask = (df_rand['date'] >= 20220429) & (df_rand['date'] <= 20220508)
    print(f"Rows in eval period (test leakage risk): {eval_mask.sum()}")
    
    # Load standard
    # Review correction: log_standard_4_22_to_5_08_pure.csv spans BOTH validation
    # (04-22..04-28) AND test (04-29..05-08). The original version of this script
    # used the file unfiltered, which pulled test-period (user,video) identities
    # into std_uv -- a "train+validation only" rule violation (no test LABELS were
    # read, but test rows were touched). Filtered to validation dates only below.
    df_train = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    df_valid = df_eval[(df_eval['date'] >= 20220422) & (df_eval['date'] <= 20220428)]

    rand_uv = set(df_rand['user_id'].astype(str) + "_" + df_rand['video_id'].astype(str))
    std_uv = set(df_train['user_id'].astype(str) + "_" + df_train['video_id'].astype(str)) | \
             set(df_valid['user_id'].astype(str) + "_" + df_valid['video_id'].astype(str))

    overlap = rand_uv.intersection(std_uv)
    print(f"Random UV pairs also in standard logs (train+valid only): {len(overlap)} ({len(overlap)/len(rand_uv):.2%})")
    
if __name__ == "__main__":
    run_cgi()

