import os
import csv
import pandas as pd
import numpy as np

DATA_DIR = "../../source/KuaiRand-Pure/data"

def load_data():
    print("Loading standard logs...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    
    df_valid = df_eval[(df_eval['date'] >= 20220422) & (df_eval['date'] <= 20220428)]
    # We do NOT use test data
    
    print("Loading video basic features...")
    df_video = pd.read_csv(os.path.join(DATA_DIR, "video_features_basic_pure.csv"))
    
    print("Loading user features...")
    df_user = pd.read_csv(os.path.join(DATA_DIR, "user_features_pure.csv"))
    
    return df_train, df_valid, df_video, df_user

def analyze_dataset_structure(df_train, df_valid, df_video, df_user):
    print("Merging author and tag info...")
    vid2author = dict(zip(df_video['video_id'], df_video['author_id']))
    vid2tag = dict(zip(df_video['video_id'], df_video['tag']))
    
    for df in [df_train, df_valid]:
        df['author_id'] = df['video_id'].map(vid2author)
        df['tag'] = df['video_id'].map(vid2tag)

    print("================== A01 Basic dataset cardinalities ==================")
    stats = {}
    for name, df in [('Train', df_train), ('Valid', df_valid)]:
        stats[name] = {
            'Unique users': df['user_id'].nunique(),
            'Unique videos': df['video_id'].nunique(),
            'Unique authors': df['author_id'].nunique(),
            'Unique tags': df['tag'].nunique(),
            'Total rows': len(df)
        }
    print(pd.DataFrame(stats).T)

    print("\n================== A02 Missingness profile ==================")
    print("Missing in video features:")
    print(df_video.isnull().mean()[df_video.isnull().mean() > 0])
    print("Missing in user features:")
    print(df_user.isnull().mean()[df_user.isnull().mean() > 0])

    print("\n================== A03 Train->Validation overlap ==================")
    train_users = set(df_train['user_id'])
    train_videos = set(df_train['video_id'])
    train_authors = set(df_train['author_id'])
    
    valid_users = set(df_valid['user_id'])
    valid_videos = set(df_valid['video_id'])
    valid_authors = set(df_valid['author_id'])
    
    print(f"Validation users in train: {len(valid_users.intersection(train_users)) / len(valid_users):.2%} ({len(valid_users.intersection(train_users))}/{len(valid_users)})")
    print(f"Validation videos in train: {len(valid_videos.intersection(train_videos)) / len(valid_videos):.2%} ({len(valid_videos.intersection(train_videos))}/{len(valid_videos)})")
    print(f"Validation authors in train: {len(valid_authors.intersection(train_authors)) / len(valid_authors):.2%} ({len(valid_authors.intersection(train_authors))}/{len(valid_authors)})")
    
    # Pairs
    df_train['uv_pair'] = df_train['user_id'].astype(str) + "_" + df_train['video_id'].astype(str)
    df_valid['uv_pair'] = df_valid['user_id'].astype(str) + "_" + df_valid['video_id'].astype(str)
    
    df_train['ua_pair'] = df_train['user_id'].astype(str) + "_" + df_train['author_id'].astype(str)
    df_valid['ua_pair'] = df_valid['user_id'].astype(str) + "_" + df_valid['author_id'].astype(str)

    df_train['ut_pair'] = df_train['user_id'].astype(str) + "_" + df_train['tag'].astype(str)
    df_valid['ut_pair'] = df_valid['user_id'].astype(str) + "_" + df_valid['tag'].astype(str)

    train_uv = set(df_train['uv_pair'])
    valid_uv = set(df_valid['uv_pair'])
    print(f"Validation user-video pairs seen in train: {len(valid_uv.intersection(train_uv)) / len(valid_uv):.2%} ({len(valid_uv.intersection(train_uv))}/{len(valid_uv)})")

    train_ua = set(df_train['ua_pair'])
    valid_ua = set(df_valid['ua_pair'])
    print(f"Validation user-author pairs seen in train: {len(valid_ua.intersection(train_ua)) / len(valid_ua):.2%} ({len(valid_ua.intersection(train_ua))}/{len(valid_ua)})")
    
    train_ut = set(df_train['ut_pair'])
    valid_ut = set(df_valid['ut_pair'])
    print(f"Validation user-tag pairs seen in train: {len(valid_ut.intersection(train_ut)) / len(valid_ut):.2%} ({len(valid_ut.intersection(train_ut))}/{len(valid_ut)})")

    print("\n================== A04 Author -> video redundancy ==================")
    vpa = df_video.groupby('author_id')['video_id'].nunique()
    print(f"Videos per author - median: {vpa.median()}")
    print(f"Videos per author - mean: {vpa.mean():.2f}")
    print(f"Authors with exactly 1 video: {sum(vpa == 1)} ({sum(vpa == 1)/len(vpa):.2%})")
    
    print("\n================== A05 Repeat-pair / affinity coverage ==================")
    # measure user-video repeat frequency in train
    uv_counts = df_train.groupby('uv_pair').size()
    print(f"Train user-video pairs appearing > 1 time: {sum(uv_counts > 1) / len(uv_counts):.2%}")
    
    ua_counts = df_train.groupby('ua_pair').size()
    print(f"Train user-author pairs appearing > 1 time: {sum(ua_counts > 1) / len(ua_counts):.2%}")
    
    ut_counts = df_train.groupby('ut_pair').size()
    print(f"Train user-tag pairs appearing > 1 time: {sum(ut_counts > 1) / len(ut_counts):.2%}")

    print("\n================== A06 Temporal interaction volume ==================")
    daily = pd.concat([df_train, df_valid]).groupby('date').agg(
        rows=('user_id', 'count'),
        long_view_rate=('long_view', 'mean'),
        unique_users=('user_id', 'nunique'),
        unique_videos=('video_id', 'nunique')
    )
    print("Daily stats (Train & Valid):")
    print(daily)

if __name__ == "__main__":
    import os
    os.makedirs("../../research/experiment_results", exist_ok=True)
    df_train, df_valid, df_video, df_user = load_data()
    analyze_dataset_structure(df_train, df_valid, df_video, df_user)

