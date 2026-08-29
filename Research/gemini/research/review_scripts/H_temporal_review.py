"""
Review script for candidate Investigation H01 (Temporal Volume Profile).

Purpose: PRE_AUDIT.md / data_profile.md reduced the daily volume series (from
A_dataset_structure.py's A06 output) to three coarse buckets ("Early train
~120k/day", "Late train ~30k/day", "Validation ~17k/day"). That summary hides
two facts visible in the raw daily series (research/scripts/A_dataset_structure.txt):

1. The official train window is 2022-04-08..2022-04-21, but the raw log file
   log_standard_4_08_to_4_21_pure.csv has ZERO rows on 2022-04-08 (first row
   is 2022-04-09).
2. Row volume is not a stable "~120k/day" during early train -- it ramps up
   sharply then decays: 52,736 (04-09) -> 227,808 (04-10) -> 278,835 peak
   (04-11) -> ... -> ~20,000/day by the end of train. This is a >13x swing
   within the official train window alone.

This script independently reproduces both facts from the raw source files
(read-only) so they can be verified without re-running the full
A_dataset_structure.py pipeline.
"""
import os
import pandas as pd

DATA_DIR = "../../source/KuaiRand-Pure/data"

def run():
    df_train = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv"))
    df_eval = pd.read_csv(os.path.join(DATA_DIR, "log_standard_4_22_to_5_08_pure.csv"))
    df_valid = df_eval[(df_eval['date'] >= 20220422) & (df_eval['date'] <= 20220428)]

    print("================== H01 (review) — Temporal volume anomaly ==================")
    print("Unique dates in train file:", sorted(df_train['date'].unique()))
    print("Rows on 2022-04-08 (official train start date):", int((df_train['date'] == 20220408).sum()))
    print("Min date present:", df_train['date'].min(), "| Max date present:", df_train['date'].max())
    print()

    daily_train = df_train.groupby('date').size()
    daily_valid = df_valid.groupby('date').size()
    print("Daily row counts, train window:")
    print(daily_train)
    print(f"\nPeak day: {daily_train.idxmax()} with {daily_train.max()} rows")
    print(f"Trough day (train window): {daily_train.idxmin()} with {daily_train.min()} rows")
    print(f"Peak/trough ratio (train window): {daily_train.max() / daily_train.min():.1f}x")
    print(f"\nMean rows/day, train days 04-09..04-15 (first 7 days WITH data): {daily_train.loc[20220409:20220415].mean():.0f}")
    print(f"Mean rows/day, train days 04-16..04-21 (last 6 days): {daily_train.loc[20220416:20220421].mean():.0f}")
    print(f"Mean rows/day, validation 04-22..04-28: {daily_valid.mean():.0f}")

if __name__ == "__main__":
    run()
