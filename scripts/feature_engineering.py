"""
Feature engineering functions for IEEE-CIS fraud detection.

Each transformation is a standalone, importable function with signature
``(df, params) -> df``. Train-fitted transformations also expose a
fit/transform class for use by ``scripts/train.py`` (requires_fit: true).

Usage:
    python scripts/feature_engineering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_HIGH_NULL_THRESHOLD = 0.4
DEFAULT_FREQUENCY_ENCODE_COLUMNS = [
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


def _fill_uid_component(series: pd.Series) -> pd.Series:
    return series.fillna("nan").astype(str)


def _time_since_last_txn(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values("TransactionDT")
    delta = sorted_df.groupby("card1")["TransactionDT"].diff()
    return delta.reindex(df.index)


def create_missingness_flags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add binary missingness indicators for high-null columns.

    Inputs: all columns listed in params['high_null_cols']
    Outputs: is_missing_{col} for each column in params['high_null_cols']
    """
    for col in params["high_null_cols"]:
        df[f"is_missing_{col}"] = df[col].isnull().astype(int)
    return df


def create_v_null_count(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Count non-null values across V-columns per row.

    Inputs: all columns prefixed with 'V'
    Outputs: v_null_count
    """
    v_cols = params.get("v_columns") or [c for c in df.columns if c.startswith("V")]
    df["v_null_count"] = df[v_cols].isnull().sum(axis=1)
    return df


def create_id_null_count(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Count non-null values across id_-columns per row.

    Inputs: all columns prefixed with 'id_'
    Outputs: id_null_count
    """
    id_cols = params.get("id_columns") or [c for c in df.columns if c.startswith("id_")]
    df["id_null_count"] = df[id_cols].isnull().sum(axis=1)
    return df


def create_email_match(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Binary flag when payer and recipient email domains match.

    Week 1 EDA showed email domain mismatch correlates with fraud; P_emaildomain
    is a top SHAP driver in v1. This gives the model a direct split on a pattern
    it otherwise learns indirectly from two high-cardinality categoricals.

    Both null → 0 (no signal, not a match — we cannot confirm domains align).

    Inputs: P_emaildomain, R_emaildomain
    Outputs: email_match
    """
    df["email_match"] = (
        df["P_emaildomain"].notna()
        & df["R_emaildomain"].notna()
        & (df["P_emaildomain"] == df["R_emaildomain"])
    ).astype(int)
    return df


def create_transaction_amt_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Log-scale and fractional-part transforms for TransactionAmt.

    log1p compresses the heavy right skew so LightGBM needs fewer splits to
    separate small vs large amounts. log1p (not log) handles TransactionAmt=0.

    TransactionAmt_cents captures the fractional dollar part; fraud sometimes
    clusters at round amounts or specific cent values (Kaggle competition lore).

    Inputs: TransactionAmt
    Outputs: TransactionAmt_log1p, TransactionAmt_cents
    """
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])
    # Modulo 1 extracts cents; row-level, no cross-row statistics.
    df["TransactionAmt_cents"] = df["TransactionAmt"] % 1
    return df


def create_missingness_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Binary null indicators for columns with fraud-differential missingness.

    EDA Section A3 showed missingness patterns differ between fraud and
    non-fraud. Explicit indicators let LightGBM use missingness as a first-class
    split and capture cross-feature missingness interactions.

    Inputs: columns listed in params["columns"]
    Outputs: {col}_is_null for each column in params["columns"]
    """
    for col in params["columns"]:
        df[f"{col}_is_null"] = df[col].isnull().astype(int)
    return df


class FrequencyEncoder:
    """Frequency-encode high-cardinality categoricals from training-split counts.

    Rare values are often suspicious; frequency 0.0 handles unseen values at
    inference without inventing a pseudo-count (no small-epsilon hack).

    Inputs: columns specified in params["columns"]
    Outputs: {col}_freq for each input column
    """

    def __init__(self) -> None:
        self.freq_maps_: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame, params: dict | None = None) -> FrequencyEncoder:
        params = params or {}
        columns = params["columns"]
        self.freq_maps_ = {
            col: df[col].value_counts(normalize=True).to_dict() for col in columns
        }
        return self

    def transform(self, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        params = params or {}
        df = df.copy()
        for col in params["columns"]:
            freq_map = self.freq_maps_[col]
            # Unseen values at transform time → 0.0 (never seen in training).
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        return df


class UidAggregator:
    """Per-UID aggregate features using (card1, addr1) as a cardholder proxy.

    card1 and addr1 are top SHAP features in v1; together they approximate
    "same cardholder" in Kaggle solutions. Captures velocity and spend patterns
    a single row cannot express. Training-split statistics only — no time windows
    in v2 (v3+ improvement).

    Inputs: card1, addr1, TransactionAmt
    Outputs: uid_tx_count, uid_amt_mean, uid_amt_std
    """

    def __init__(self) -> None:
        self.uid_stats_: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, params: dict | None = None) -> UidAggregator:
        stats = (
            df.groupby(["card1", "addr1"], dropna=False)["TransactionAmt"]
            .agg(["count", "mean", "std"])
            .rename(columns={"count": "uid_tx_count", "mean": "uid_amt_mean", "std": "uid_amt_std"})
        )
        # Single-transaction UIDs have undefined std → 0 (no variance signal).
        stats["uid_amt_std"] = stats["uid_amt_std"].fillna(0.0)
        self.uid_stats_ = stats
        return self

    def transform(self, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        if self.uid_stats_ is None:
            raise RuntimeError("UidAggregator.transform called before fit")

        df = df.copy()
        merged = df.merge(
            self.uid_stats_,
            left_on=["card1", "addr1"],
            right_index=True,
            how="left",
        )
        # Unseen UIDs at inference: count=0, mean=NaN, std=0.
        df["uid_tx_count"] = merged["uid_tx_count"].fillna(0)
        df["uid_amt_mean"] = merged["uid_amt_mean"]
        df["uid_amt_std"] = merged["uid_amt_std"].fillna(0.0)
        return df


def normalize_d_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Normalize D1-D15 timedelta columns by subtracting from transaction day.

    For each D column, computes floor(TransactionDT / 86400 - D_col).
    This converts raw timedeltas into per-client near-constants,
    following Chris Deotte's 1st-place IEEE-CIS methodology.

    Inputs: TransactionDT, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15
    Outputs: D1n, D2n, D3n, D4n, D5n, D6n, D7n, D8n, D9n, D10n, D11n, D12n, D13n, D14n, D15n
    """
    d_cols = [f"D{i}" for i in range(1, 16)]
    day = df["TransactionDT"] / np.float32(86400)
    for d_col in d_cols:
        if d_col in df.columns:
            df[f"{d_col}n"] = np.floor(day - df[d_col])
    return df


def _m_column_to_numeric(series: pd.Series) -> pd.Series:
    """Encode M-column categoricals for numeric aggregation (T/F/M0-M2)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float32")
    return series.map({"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}).astype(
        "float32"
    )


class UidD1Aggregator:
    """Construct D1-normalized UID and compute UID-level aggregation features.

    UID construction: uid = str(card1) + '_' + str(addr1) + '_' + str(floor(day - D1))
    where day = TransactionDT / 86400.

    This follows Chris Deotte's 1st-place IEEE-CIS methodology. The UID itself
    is used only as a grouping key and is NOT included as a model feature.

    Aggregation features computed per UID:
    - TransactionAmt: mean, std
    - D4, D9, D10, D11, D15 (raw values): mean, std
    - C1-C14: mean
    - M1-M9: mean (T/F/M0-M2 encoded to numeric for aggregation only)
    - D15n (normalized): std  (confidence signal: std=0 implies single client)
    - UID frequency (count)
    - P_emaildomain nunique per UID
    - dist1 nunique per UID

    Inputs: card1, addr1, TransactionDT, D1, D4, D9, D10, D11, D15,
            C1-C14, M1-M9, TransactionAmt, P_emaildomain, dist1, D15n
    Outputs: uid_d1_fe,
             TransactionAmt_uid_d1_mean, TransactionAmt_uid_d1_std,
             D4_uid_d1_mean, D4_uid_d1_std,
             D9_uid_d1_mean, D9_uid_d1_std,
             D10_uid_d1_mean, D10_uid_d1_std,
             D11_uid_d1_mean, D11_uid_d1_std,
             D15_uid_d1_mean, D15_uid_d1_std,
             C1_uid_d1_mean, C2_uid_d1_mean, ..., C14_uid_d1_mean,
             M1_uid_d1_mean, M2_uid_d1_mean, ..., M9_uid_d1_mean,
             D15n_uid_d1_std,
             uid_d1_P_emaildomain_nunique,
             uid_d1_dist1_nunique

    requires_fit: true
    """

    def __init__(self) -> None:
        self.params: dict = {}
        self.agg_tables: dict[str, pd.Series] = {}
        self.uid_freq: pd.Series | None = None
        self.nunique_tables: dict[str, pd.Series] = {}

    def _build_uid(self, df: pd.DataFrame) -> pd.Series:
        """Construct the D1-normalized UID string. Internal helper."""
        day = df["TransactionDT"] / np.float32(86400)
        d1n = np.floor(day - df["D1"])
        return (
            df["card1"].astype(str)
            + "_"
            + df["addr1"].astype(str)
            + "_"
            + d1n.astype(str)
        )

    def fit(self, train_df: pd.DataFrame, params: dict | None = None) -> UidD1Aggregator:
        """Compute all aggregation lookup tables from training data only."""
        self.params = params or {}
        uid = self._build_uid(train_df)

        # --- Frequency encoding ---
        self.uid_freq = uid.value_counts()

        # --- Numeric aggregations (mean, std) ---
        agg_cols_mean_std = {
            "TransactionAmt": ["mean", "std"],
            "D4": ["mean", "std"],
            "D9": ["mean", "std"],
            "D10": ["mean", "std"],
            "D11": ["mean", "std"],
            "D15": ["mean", "std"],
        }
        for col, stats in agg_cols_mean_std.items():
            if col not in train_df.columns:
                continue
            grouped = train_df.groupby(uid)[col]
            for stat in stats:
                key = f"{col}_uid_d1_{stat}"
                self.agg_tables[key] = grouped.agg(stat)

        # --- C-feature means ---
        for i in range(1, 15):
            col = f"C{i}"
            if col in train_df.columns:
                key = f"{col}_uid_d1_mean"
                self.agg_tables[key] = train_df.groupby(uid)[col].mean()

        # --- M-feature means (encode categoricals for aggregation only) ---
        for i in range(1, 10):
            col = f"M{i}"
            if col in train_df.columns:
                key = f"{col}_uid_d1_mean"
                m_numeric = _m_column_to_numeric(train_df[col])
                self.agg_tables[key] = m_numeric.groupby(uid).mean()

        # --- D15n std (client confidence signal) ---
        if "D15n" in train_df.columns:
            self.agg_tables["D15n_uid_d1_std"] = train_df.groupby(uid)["D15n"].std()

        # --- Nunique counts ---
        for col in ["P_emaildomain", "dist1"]:
            if col in train_df.columns:
                key = f"uid_d1_{col}_nunique"
                self.nunique_tables[key] = train_df.groupby(uid)[col].nunique()

        return self

    def transform(self, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        """Apply fitted lookup tables to any DataFrame."""
        if self.uid_freq is None:
            raise RuntimeError("UidD1Aggregator.transform called before fit")

        uid = self._build_uid(df)

        # Frequency encoding
        df["uid_d1_fe"] = uid.map(self.uid_freq).fillna(0).astype("float32")

        # Numeric aggregations
        for key, lookup in self.agg_tables.items():
            df[key] = uid.map(lookup).astype("float32")
            # NaN for unseen UIDs — LightGBM handles natively

        # Nunique counts
        for key, lookup in self.nunique_tables.items():
            df[key] = uid.map(lookup).astype("float32")

        # Do NOT add uid column itself — it is a grouping key, not a feature
        return df


def create_both_emails_present(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Flag rows where both payer and recipient email domains are present.

    Inputs: P_emaildomain, R_emaildomain
    Outputs: both_emails_present
    """
    df["both_emails_present"] = (
        df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    ).astype(int)
    return df


def create_email_suffixes(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Extract the domain suffix from payer and recipient email domains.

    Inputs: P_emaildomain, R_emaildomain
    Outputs: P_email_suffix, R_email_suffix
    """
    df["P_email_suffix"] = df["P_emaildomain"].astype(str).str.split(".").str[-1]
    df["R_email_suffix"] = df["R_emaildomain"].astype(str).str.split(".").str[-1]
    df.loc[df["P_emaildomain"].isnull(), "P_email_suffix"] = np.nan
    df.loc[df["R_emaildomain"].isnull(), "R_email_suffix"] = np.nan
    return df


def create_transaction_amt_log1p(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Log-transform transaction amount with log1p.

    Inputs: TransactionAmt
    Outputs: TransactionAmt_log1p
    """
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"])
    return df


def create_card1_amt_stats(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Map card1-level transaction amount mean and std onto each row.

    Inputs: card1, TransactionAmt
    Outputs: card1_amt_mean, card1_amt_std

    params must contain card1_stats (DataFrame with mean/std columns),
    global_amt_mean, and global_amt_std fitted on the training split.
    """
    card1_stats = params["card1_stats"]
    global_amt_mean = params["global_amt_mean"]
    global_amt_std = params["global_amt_std"]
    df["card1_amt_mean"] = df["card1"].map(card1_stats["mean"]).fillna(global_amt_mean)
    df["card1_amt_std"] = df["card1"].map(card1_stats["std"]).fillna(global_amt_std)
    return df


def create_amt_zscore_card1(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Z-score transaction amount relative to card1-level mean and std.

    Inputs: TransactionAmt, card1_amt_mean, card1_amt_std
    Outputs: amt_zscore_card1
    """
    z = (df["TransactionAmt"] - df["card1_amt_mean"]) / df["card1_amt_std"]
    df["amt_zscore_card1"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def create_time_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Derive calendar features from TransactionDT.

    Inputs: TransactionDT
    Outputs: transaction_day, hour_of_day, day_of_week
    """
    df["transaction_day"] = df["TransactionDT"] // 86400
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = df["transaction_day"] % 7
    return df


def create_time_since_last_txn_card1(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Seconds since the previous transaction on the same card1.

    Inputs: card1, TransactionDT
    Outputs: time_since_last_txn_card1
    """
    df["time_since_last_txn_card1"] = _time_since_last_txn(df)
    return df


def create_uid(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Build a composite user identifier from card, address, and email.

    Inputs: card1, addr1, P_emaildomain
    Outputs: uid
    """
    df["uid"] = (
        _fill_uid_component(df["card1"])
        + "_"
        + _fill_uid_component(df["addr1"])
        + "_"
        + _fill_uid_component(df["P_emaildomain"])
    )
    return df


def create_uid_amt_stats(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Map uid-level transaction count, mean, and std onto each row.

    Inputs: uid, TransactionAmt
    Outputs: uid_txn_count, uid_amt_mean, uid_amt_std

    params must contain uid_stats (DataFrame with uid_txn_count, uid_amt_mean,
    uid_amt_std columns) and global_amt_mean fitted on the training split.
    """
    uid_stats = params["uid_stats"]
    global_amt_mean = params["global_amt_mean"]
    df["uid_txn_count"] = df["uid"].map(uid_stats["uid_txn_count"]).fillna(1)
    df["uid_amt_mean"] = df["uid"].map(uid_stats["uid_amt_mean"]).fillna(global_amt_mean)
    df["uid_amt_std"] = df["uid"].map(uid_stats["uid_amt_std"]).fillna(0)
    return df


def create_v_block_aggregates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Row-wise summary statistics across V-columns.

    Inputs: all columns prefixed with 'V'
    Outputs: v_nonnull_count, v_mean, v_std, v_min, v_max
    """
    v_cols = params.get("v_columns") or [c for c in df.columns if c.startswith("V")]
    v_block = df[v_cols]
    df["v_nonnull_count"] = v_block.notnull().sum(axis=1)
    df["v_mean"] = v_block.mean(axis=1, skipna=True)
    df["v_std"] = v_block.std(axis=1, skipna=True)
    df["v_min"] = v_block.min(axis=1, skipna=True)
    df["v_max"] = v_block.max(axis=1, skipna=True)
    return df


def create_frequency_encodings(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Map train-split value counts onto each row as frequency features.

    Inputs: columns listed in params['frequency_encode_columns']
    Outputs: {col}_freq for each column in params['frequency_encode_columns']

    params must contain freq_maps: a dict mapping column name to value counts
    fitted on the training split.
    """
    for col in params["frequency_encode_columns"]:
        freq_name = f"{col}_freq"
        df[freq_name] = df[col].map(params["freq_maps"][col]).fillna(1)
    return df


class HighNullMissingnessFlags:
    """Fit high-null column list on train, then add missingness flags."""

    def __init__(self, high_null_threshold: float = DEFAULT_HIGH_NULL_THRESHOLD, **params: object):
        self.high_null_threshold = float(
            params.get("high_null_threshold", high_null_threshold)
        )
        self.high_null_cols: list[str] = []
        self._fit_params: dict = {}

    def fit(self, df: pd.DataFrame) -> HighNullMissingnessFlags:
        null_rates = df.isnull().mean()
        high_null_cols = null_rates[null_rates > self.high_null_threshold].index.tolist()
        self.high_null_cols = high_null_cols
        self._fit_params = {"high_null_cols": high_null_cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_missingness_flags(df.copy(), self._fit_params)


class Card1AmountStatsEncoder:
    """Fit card1 amount statistics on train, then map onto any split."""

    def __init__(self, **params: object):
        self._fit_params: dict = {}

    def fit(self, df: pd.DataFrame) -> Card1AmountStatsEncoder:
        self._fit_params = {
            "card1_stats": df.groupby("card1")["TransactionAmt"].agg(["mean", "std"]),
            "global_amt_mean": float(df["TransactionAmt"].mean()),
            "global_amt_std": float(df["TransactionAmt"].std()),
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_card1_amt_stats(df.copy(), self._fit_params)


class UidAmountStatsEncoder:
    """Fit uid amount statistics on train, then map onto any split."""

    def __init__(self, **params: object):
        self._fit_params: dict = {}

    def fit(self, df: pd.DataFrame) -> UidAmountStatsEncoder:
        uid_stats = df.groupby("uid")["TransactionAmt"].agg(["count", "mean", "std"])
        uid_stats.columns = ["uid_txn_count", "uid_amt_mean", "uid_amt_std"]
        self._fit_params = {
            "uid_stats": uid_stats,
            "global_amt_mean": float(df["TransactionAmt"].mean()),
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_uid_amt_stats(df.copy(), self._fit_params)


class LegacyFrequencyEncoder:
    """Legacy frequency encoder used by feature_engineering.py main() demo pipeline."""

    def __init__(self, **params: object):
        self.frequency_encode_columns = params.get(
            "frequency_encode_columns", DEFAULT_FREQUENCY_ENCODE_COLUMNS
        )
        self._fit_params: dict = {}

    def fit(self, df: pd.DataFrame) -> LegacyFrequencyEncoder:
        freq_maps = {
            col: df[col].value_counts(dropna=False)
            for col in self.frequency_encode_columns
        }
        self._fit_params = {
            "frequency_encode_columns": self.frequency_encode_columns,
            "freq_maps": freq_maps,
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_frequency_encodings(df.copy(), self._fit_params)


def _engineer_splits(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missingness = HighNullMissingnessFlags().fit(train_df)
    print(f"High-null columns (>{DEFAULT_HIGH_NULL_THRESHOLD:.0%}): {len(missingness.high_null_cols)}")
    train_df = missingness.transform(train_df)
    test_df = missingness.transform(test_df)
    print(f"Missingness flags created: {len(missingness.high_null_cols)}")

    for transform in (
        create_v_null_count,
        create_id_null_count,
        create_email_match,
        create_both_emails_present,
        create_email_suffixes,
        create_transaction_amt_log1p,
    ):
        train_df = transform(train_df, {})
        test_df = transform(test_df, {})

    card1_stats = Card1AmountStatsEncoder().fit(train_df)
    train_df = card1_stats.transform(train_df)
    test_df = card1_stats.transform(test_df)
    train_df = create_amt_zscore_card1(train_df, {})
    test_df = create_amt_zscore_card1(test_df, {})

    for transform in (
        create_time_features,
        create_time_since_last_txn_card1,
        create_uid,
    ):
        train_df = transform(train_df, {})
        test_df = transform(test_df, {})

    uid_stats = UidAmountStatsEncoder().fit(train_df)
    train_df = uid_stats.transform(train_df)
    test_df = uid_stats.transform(test_df)

    train_df = create_v_block_aggregates(train_df, {})
    test_df = create_v_block_aggregates(test_df, {})

    frequency = LegacyFrequencyEncoder().fit(train_df)
    train_df = frequency.transform(train_df)
    test_df = frequency.transform(test_df)
    return train_df, test_df


def main() -> None:
    data_path = REPO_ROOT / "data" / "raw" / "train_merged.parquet"
    processed_dir = REPO_ROOT / "data" / "processed"
    train_output = processed_dir / "train_featured.parquet"
    test_output = processed_dir / "test_featured.parquet"
    feature_names_path = processed_dir / "engineered_feature_names.txt"

    df = pd.read_parquet(data_path)
    original_cols = set(df.columns)
    print(f"Loaded {data_path.name}: shape={df.shape}, columns={len(df.columns)}")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * DEFAULT_TRAIN_FRACTION)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    train_df, test_df = _engineer_splits(train_df, test_df)

    v_cols = [c for c in df.columns if c.startswith("V")]
    print(f"V-columns: {len(v_cols)}")

    new_feature_cols = [c for c in train_df.columns if c not in original_cols]
    print(
        f"Final shapes — train: {train_df.shape}, test: {test_df.shape} | "
        f"new columns: {len(new_feature_cols)}"
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_output, index=False)
    test_df.to_parquet(test_output, index=False)

    train_size_mb = train_output.stat().st_size / (1024 * 1024)
    test_size_mb = test_output.stat().st_size / (1024 * 1024)
    print(f"Saved {train_output} ({train_size_mb:.1f} MB)")
    print(f"Saved {test_output} ({test_size_mb:.1f} MB)")

    feature_names_path.write_text("\n".join(new_feature_cols) + "\n")
    print(f"Saved {len(new_feature_cols)} engineered feature names to {feature_names_path}")


if __name__ == "__main__":
    main()
