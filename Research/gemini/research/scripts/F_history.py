import os
import pandas as pd

DATA_DIR = "../../source/KuaiRand-Pure/data"

def run_history():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    df_valid = df_eval[(df_eval['date'] >= 20220422) & (df_eval['date'] <= 20220428)]
    
    # Author map
    df_video = pd.read_csv(os.path.join(DATA_DIR, "video_features_basic_pure.csv"))
    vid2author = dict(zip(df_video['video_id'], df_video['author_id']))
    df_train['author_id'] = df_train['video_id'].map(vid2author)
    df_valid['author_id'] = df_valid['video_id'].map(vid2author)
    
    user_train_counts = df_train.groupby('user_id').size()
    
    def get_bucket(c):
        if pd.isna(c) or c == 0: return 'Cold'
        if c < 10: return 'T1 (<10)'
        if c < 50: return 'T2 (10-49)'
        if c < 150: return 'T3 (50-149)'
        return 'T4 (150+)'
        
    df_valid['bucket'] = df_valid['user_id'].map(user_train_counts).apply(get_bucket)
    
    print("\n================== F Historical Availability ==================")
    for b in ['Cold', 'T1 (<10)', 'T2 (10-49)', 'T3 (50-149)', 'T4 (150+)', 'ALL']:
        if b == 'ALL':
            d = df_valid
        else:
            d = df_valid[df_valid['bucket'] == b]
            
        users = d['user_id'].unique()
        if len(users) == 0:
            continue
            
        # Prior interactions count
        train_counts = user_train_counts.reindex(users).fillna(0)
        gt1 = (train_counts >= 1).sum() / len(users)
        gt5 = (train_counts >= 5).sum() / len(users)
        gt10 = (train_counts >= 10).sum() / len(users)
        med = train_counts.median()
        
        # Rep video
        valid_uv = d[['user_id', 'video_id']].drop_duplicates()
        train_uv = df_train[['user_id', 'video_id']].drop_duplicates()
        rep_v = len(valid_uv.merge(train_uv, on=['user_id', 'video_id'])) / len(d)
        
        # Rep author
        valid_ua = d[['user_id', 'author_id']].dropna().drop_duplicates()
        train_ua = df_train[['user_id', 'author_id']].dropna().drop_duplicates()
        rep_a = len(valid_ua.merge(train_ua, on=['user_id', 'author_id'])) / len(d)
        
        print(f"{b:10s} | Users: {len(users):4d} | Med Prior: {med:4.0f} | >=1: {gt1:.2%} | >=5: {gt5:.2%} | >=10: {gt10:.2%} | Rep Video: {rep_v:.2%} | Rep Author: {rep_a:.2%}")

if __name__ == "__main__":
    run_history()

