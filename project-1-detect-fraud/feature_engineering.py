"""
Feature engineering pipeline for IEEE-CIS fraud detection.

Reads train_merged.parquet, engineers features, saves enriched dataframe.
All frequency/aggregation stats are computed on TRAIN split only
(time-based split) and mapped onto test to prevent leakage.

Usage:
    python feature_engineering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "train_merged.parquet"
OUTPUT_DIR = REPO_ROOT / "data"
TRAIN_OUTPUT = OUTPUT_DIR / "train_featured.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test_featured.parquet"
FEATURE_NAMES_PATH = OUTPUT_DIR / "engineered_feature_names.txt"
TRAIN_FRACTION = 0.8


def _fill_uid_component(series: pd.Series) -> pd.Series:
    return series.fillna("nan").astype(str)


def _time_since_last_txn(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values("TransactionDT")
    delta = sorted_df.groupby("card1")["TransactionDT"].diff()
    return delta.reindex(df.index)


def main() -> None:
    # === LOAD DATA ===
    df = pd.read_parquet(DATA_PATH)
    original_cols = set(df.columns)
    print(f"Loaded {DATA_PATH.name}: shape={df.shape}, columns={len(df.columns)}")

    # === TIME-BASED TRAIN/TEST SPLIT ===
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # ============================================================
    # 1. MISSINGNESS FEATURES
    # ============================================================

    # 1a.
    null_rates = train_df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.40].index.tolist()
    print(f"High-null columns (>40%): {len(high_null_cols)}")

    # 1b.
    for col in high_null_cols:
        flag_name = f"is_missing_{col}"
        train_df[flag_name] = train_df[col].isnull().astype(int)
        test_df[flag_name] = test_df[col].isnull().astype(int)
    print(f"Missingness flags created: {len(high_null_cols)}")

    # 1c.
    v_cols = [c for c in df.columns if c.startswith("V")]
    train_df["v_null_count"] = train_df[v_cols].isnull().sum(axis=1)
    test_df["v_null_count"] = test_df[v_cols].isnull().sum(axis=1)

    # 1d.
    id_cols = [c for c in df.columns if c.startswith("id_")]
    train_df["id_null_count"] = train_df[id_cols].isnull().sum(axis=1)
    test_df["id_null_count"] = test_df[id_cols].isnull().sum(axis=1)

    # ============================================================
    # 2. EMAIL / IDENTITY FEATURES
    # ============================================================

    # 2a.
    email_match = (
        train_df["P_emaildomain"].notna()
        & train_df["R_emaildomain"].notna()
        & (train_df["P_emaildomain"] == train_df["R_emaildomain"])
    ).astype(int)
    train_df["email_match"] = email_match
    test_df["email_match"] = (
        test_df["P_emaildomain"].notna()
        & test_df["R_emaildomain"].notna()
        & (test_df["P_emaildomain"] == test_df["R_emaildomain"])
    ).astype(int)

    # 2b.
    train_df["both_emails_present"] = (
        train_df["P_emaildomain"].notna() & train_df["R_emaildomain"].notna()
    ).astype(int)
    test_df["both_emails_present"] = (
        test_df["P_emaildomain"].notna() & test_df["R_emaildomain"].notna()
    ).astype(int)

    # 2c.
    for split in (train_df, test_df):
        split["P_email_suffix"] = split["P_emaildomain"].astype(str).str.split(".").str[-1]
        split["R_email_suffix"] = split["R_emaildomain"].astype(str).str.split(".").str[-1]
        split.loc[split["P_emaildomain"].isnull(), "P_email_suffix"] = np.nan
        split.loc[split["R_emaildomain"].isnull(), "R_email_suffix"] = np.nan

    # ============================================================
    # 3. AMOUNT TRANSFORMS
    # ============================================================

    # 3a.
    train_df["TransactionAmt_log1p"] = np.log1p(train_df["TransactionAmt"])
    test_df["TransactionAmt_log1p"] = np.log1p(test_df["TransactionAmt"])

    # 3b.
    card1_stats = train_df.groupby("card1")["TransactionAmt"].agg(["mean", "std"])
    global_amt_mean = float(train_df["TransactionAmt"].mean())
    global_amt_std = float(train_df["TransactionAmt"].std())

    for split in (train_df, test_df):
        split["card1_amt_mean"] = split["card1"].map(card1_stats["mean"]).fillna(global_amt_mean)
        split["card1_amt_std"] = split["card1"].map(card1_stats["std"]).fillna(global_amt_std)

    # 3c.
    for split in (train_df, test_df):
        z = (split["TransactionAmt"] - split["card1_amt_mean"]) / split["card1_amt_std"]
        split["amt_zscore_card1"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ============================================================
    # 4. TIME FEATURES
    # ============================================================

    # 4a–4c.
    for split in (train_df, test_df):
        split["transaction_day"] = split["TransactionDT"] // 86400
        split["hour_of_day"] = (split["TransactionDT"] // 3600) % 24
        split["day_of_week"] = split["transaction_day"] % 7

    # 4d.
    train_df["time_since_last_txn_card1"] = _time_since_last_txn(train_df)
    test_df["time_since_last_txn_card1"] = _time_since_last_txn(test_df)

    # ============================================================
    # 5. UID AGGREGATIONS
    # ============================================================

    # 5a.
    for split in (train_df, test_df):
        split["uid"] = (
            _fill_uid_component(split["card1"])
            + "_"
            + _fill_uid_component(split["addr1"])
            + "_"
            + _fill_uid_component(split["P_emaildomain"])
        )

    # 5b.
    uid_stats = train_df.groupby("uid")["TransactionAmt"].agg(["count", "mean", "std"])
    uid_stats.columns = ["uid_txn_count", "uid_amt_mean", "uid_amt_std"]

    for split in (train_df, test_df):
        split["uid_txn_count"] = split["uid"].map(uid_stats["uid_txn_count"]).fillna(1)
        split["uid_amt_mean"] = split["uid"].map(uid_stats["uid_amt_mean"]).fillna(global_amt_mean)
        split["uid_amt_std"] = split["uid"].map(uid_stats["uid_amt_std"]).fillna(0)

    # ============================================================
    # 6. V-FEATURE SUMMARIES
    # ============================================================

    # 6a.
    print(f"V-columns: {len(v_cols)}")

    # 6b.
    for split in (train_df, test_df):
        v_block = split[v_cols]
        split["v_nonnull_count"] = v_block.notnull().sum(axis=1)
        split["v_mean"] = v_block.mean(axis=1, skipna=True)
        split["v_std"] = v_block.std(axis=1, skipna=True)
        split["v_min"] = v_block.min(axis=1, skipna=True)
        split["v_max"] = v_block.max(axis=1, skipna=True)

    # ============================================================
    # 7. CATEGORICAL FREQUENCY ENCODING
    # ============================================================

    freq_cols = [
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "ProductCD",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
    ]

    for col in freq_cols:
        freq_map = train_df[col].value_counts(dropna=False)
        freq_name = f"{col}_freq"
        train_df[freq_name] = train_df[col].map(freq_map).fillna(1)
        test_df[freq_name] = test_df[col].map(freq_map).fillna(1)

    # ============================================================
    # 8. SAVE OUTPUTS
    # ============================================================

    new_feature_cols = [c for c in train_df.columns if c not in original_cols]
    print(
        f"Final shapes — train: {train_df.shape}, test: {test_df.shape} | "
        f"new columns: {len(new_feature_cols)}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
