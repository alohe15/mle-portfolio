"""
Feature engineering pipeline for IEEE-CIS fraud detection.

Reads train_merged.parquet, engineers features, saves enriched dataframe.
All frequency/aggregation stats are computed on TRAIN split only
(time-based split) and mapped onto test to prevent leakage.

Usage:
    python scripts/feature_engineering.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_config import load_config, repo_path

paths = load_config("paths")
fe_config = load_config("feature_engineering")

DATA_PATH = repo_path(paths["train_merged_parquet"])
TRAIN_OUTPUT = repo_path(paths["train_featured_parquet"])
TEST_OUTPUT = repo_path(paths["test_featured_parquet"])
FEATURE_NAMES_PATH = repo_path(paths["engineered_feature_names"])
TRAIN_FRACTION = fe_config["train_fraction"]
HIGH_NULL_THRESHOLD = fe_config["high_null_threshold"]
FREQ_COLS = fe_config["frequency_encode_columns"]


def _fill_uid_component(series: pd.Series) -> pd.Series:
    return series.fillna("nan").astype(str)


def _time_since_last_txn(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values("TransactionDT")
    delta = sorted_df.groupby("card1")["TransactionDT"].diff()
    return delta.reindex(df.index)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    original_cols = set(df.columns)
    print(f"Loaded {DATA_PATH.name}: shape={df.shape}, columns={len(df.columns)}")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    null_rates = train_df.isnull().mean()
    high_null_cols = null_rates[null_rates > HIGH_NULL_THRESHOLD].index.tolist()
    print(f"High-null columns (>{HIGH_NULL_THRESHOLD:.0%}): {len(high_null_cols)}")

    for col in high_null_cols:
        flag_name = f"is_missing_{col}"
        train_df[flag_name] = train_df[col].isnull().astype(int)
        test_df[flag_name] = test_df[col].isnull().astype(int)
    print(f"Missingness flags created: {len(high_null_cols)}")

    v_cols = [c for c in df.columns if c.startswith("V")]
    train_df["v_null_count"] = train_df[v_cols].isnull().sum(axis=1)
    test_df["v_null_count"] = test_df[v_cols].isnull().sum(axis=1)

    id_cols = [c for c in df.columns if c.startswith("id_")]
    train_df["id_null_count"] = train_df[id_cols].isnull().sum(axis=1)
    test_df["id_null_count"] = test_df[id_cols].isnull().sum(axis=1)

    train_df["email_match"] = (
        train_df["P_emaildomain"].notna()
        & train_df["R_emaildomain"].notna()
        & (train_df["P_emaildomain"] == train_df["R_emaildomain"])
    ).astype(int)
    test_df["email_match"] = (
        test_df["P_emaildomain"].notna()
        & test_df["R_emaildomain"].notna()
        & (test_df["P_emaildomain"] == test_df["R_emaildomain"])
    ).astype(int)

    train_df["both_emails_present"] = (
        train_df["P_emaildomain"].notna() & train_df["R_emaildomain"].notna()
    ).astype(int)
    test_df["both_emails_present"] = (
        test_df["P_emaildomain"].notna() & test_df["R_emaildomain"].notna()
    ).astype(int)

    for split in (train_df, test_df):
        split["P_email_suffix"] = split["P_emaildomain"].astype(str).str.split(".").str[-1]
        split["R_email_suffix"] = split["R_emaildomain"].astype(str).str.split(".").str[-1]
        split.loc[split["P_emaildomain"].isnull(), "P_email_suffix"] = np.nan
        split.loc[split["R_emaildomain"].isnull(), "R_email_suffix"] = np.nan

    train_df["TransactionAmt_log1p"] = np.log1p(train_df["TransactionAmt"])
    test_df["TransactionAmt_log1p"] = np.log1p(test_df["TransactionAmt"])

    card1_stats = train_df.groupby("card1")["TransactionAmt"].agg(["mean", "std"])
    global_amt_mean = float(train_df["TransactionAmt"].mean())
    global_amt_std = float(train_df["TransactionAmt"].std())

    for split in (train_df, test_df):
        split["card1_amt_mean"] = split["card1"].map(card1_stats["mean"]).fillna(global_amt_mean)
        split["card1_amt_std"] = split["card1"].map(card1_stats["std"]).fillna(global_amt_std)

    for split in (train_df, test_df):
        z = (split["TransactionAmt"] - split["card1_amt_mean"]) / split["card1_amt_std"]
        split["amt_zscore_card1"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    for split in (train_df, test_df):
        split["transaction_day"] = split["TransactionDT"] // 86400
        split["hour_of_day"] = (split["TransactionDT"] // 3600) % 24
        split["day_of_week"] = split["transaction_day"] % 7

    train_df["time_since_last_txn_card1"] = _time_since_last_txn(train_df)
    test_df["time_since_last_txn_card1"] = _time_since_last_txn(test_df)

    for split in (train_df, test_df):
        split["uid"] = (
            _fill_uid_component(split["card1"])
            + "_"
            + _fill_uid_component(split["addr1"])
            + "_"
            + _fill_uid_component(split["P_emaildomain"])
        )

    uid_stats = train_df.groupby("uid")["TransactionAmt"].agg(["count", "mean", "std"])
    uid_stats.columns = ["uid_txn_count", "uid_amt_mean", "uid_amt_std"]

    for split in (train_df, test_df):
        split["uid_txn_count"] = split["uid"].map(uid_stats["uid_txn_count"]).fillna(1)
        split["uid_amt_mean"] = split["uid"].map(uid_stats["uid_amt_mean"]).fillna(global_amt_mean)
        split["uid_amt_std"] = split["uid"].map(uid_stats["uid_amt_std"]).fillna(0)

    print(f"V-columns: {len(v_cols)}")

    for split in (train_df, test_df):
        v_block = split[v_cols]
        split["v_nonnull_count"] = v_block.notnull().sum(axis=1)
        split["v_mean"] = v_block.mean(axis=1, skipna=True)
        split["v_std"] = v_block.std(axis=1, skipna=True)
        split["v_min"] = v_block.min(axis=1, skipna=True)
        split["v_max"] = v_block.max(axis=1, skipna=True)

    for col in FREQ_COLS:
        freq_map = train_df[col].value_counts(dropna=False)
        freq_name = f"{col}_freq"
        train_df[freq_name] = train_df[col].map(freq_map).fillna(1)
        test_df[freq_name] = test_df[col].map(freq_map).fillna(1)

    new_feature_cols = [c for c in train_df.columns if c not in original_cols]
    print(
        f"Final shapes — train: {train_df.shape}, test: {test_df.shape} | "
        f"new columns: {len(new_feature_cols)}"
    )

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_OUTPUT, index=False)
    test_df.to_parquet(TEST_OUTPUT, index=False)

    train_size_mb = TRAIN_OUTPUT.stat().st_size / (1024 * 1024)
    test_size_mb = TEST_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Saved {TRAIN_OUTPUT} ({train_size_mb:.1f} MB)")
    print(f"Saved {TEST_OUTPUT} ({test_size_mb:.1f} MB)")

    FEATURE_NAMES_PATH.write_text("\n".join(new_feature_cols) + "\n")
    print(f"Saved {len(new_feature_cols)} engineered feature names to {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    main()
