"""
Review correction for Investigation I01 (Random Exposure Log).

Bug found in research/scripts/GI_analysis.py (Phase I): the "Random UV pairs
also in standard logs" statistic built `std_uv` from the RAW, unfiltered
log_standard_4_22_to_5_08_pure.csv, which spans BOTH the validation window
(2022-04-22..04-28) AND the test window (2022-04-29..05-08). This means the
original computation touched test-period (user_id, video_id) identities,
violating the "train + validation only" development rule -- even though no
test LABELS (long_view) were read or used.

This script recomputes the same statistic using train + validation only, and
separately reports the test-only contribution so the size of the original
violation is auditable.
"""
import os
import pandas as pd

DATA_DIR = "../../source/KuaiRand-Pure/data"

def run():
    df_train = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    df_valid = df_eval[(df_eval['date'] >= 20220422) & (df_eval['date'] <= 20220428)]
    df_test = df_eval[(df_eval['date'] >= 20220429) & (df_eval['date'] <= 20220508)]
    df_rand = pd.read_csv(os.path.join(DATA_DIR, "log_random_4_22_to_5_08_pure.csv"))

    rand_uv = set(df_rand['user_id'].astype(str) + "_" + df_rand['video_id'].astype(str))
    train_uv = set(df_train['user_id'].astype(str) + "_" + df_train['video_id'].astype(str))
    valid_uv = set(df_valid['user_id'].astype(str) + "_" + df_valid['video_id'].astype(str))
    test_uv = set(df_test['user_id'].astype(str) + "_" + df_test['video_id'].astype(str))

    correct = rand_uv.intersection(train_uv | valid_uv)
    original_buggy = rand_uv.intersection(train_uv | valid_uv | test_uv)
    test_only_contribution = rand_uv.intersection(test_uv)

    print("================== I01 (review) — Random log overlap, corrected ==================")
    print(f"Random log unique (user,video) pairs: {len(rand_uv)}")
    print(f"CORRECT overlap (train+valid only):    {len(correct)} ({len(correct)/len(rand_uv):.2%})")
    print(f"ORIGINAL computation (train+valid+test, rule violation): {len(original_buggy)} ({len(original_buggy)/len(rand_uv):.2%})")
    print(f"Of which came only from touching test:  {len(test_only_contribution)} ({len(test_only_contribution)/len(rand_uv):.2%})")
    print()

    rand_valid_rows = int(((df_rand['date'] >= 20220422) & (df_rand['date'] <= 20220428)).sum())
    rand_test_rows = int(((df_rand['date'] >= 20220429) & (df_rand['date'] <= 20220508)).sum())
    print(f"Random log rows in validation window (04-22..04-28): {rand_valid_rows}")
    print(f"Random log rows in test window (04-29..05-08):       {rand_test_rows}")
    print(f"Random log rows in train window (04-08..04-21):      "
          f"{int(((df_rand['date'] >= 20220408) & (df_rand['date'] <= 20220421)).sum())}")
    print(f"Total random log rows: {len(df_rand)}")

if __name__ == "__main__":
    run()
