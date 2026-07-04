"""
Week 1 EDA — IEEE-CIS Fraud Detection
======================================
Deep exploratory data analysis for the fraud detection project.

Dataset: IEEE-CIS Fraud Detection (Vesta Corporation e-commerce transactions)
Files: train_transaction.csv (~590K rows, 394 cols) + train_identity.csv (~144K rows, 41 cols)
Target: isFraud (1 = fraudulent, 0 = legitimate)

Usage:
    python notebooks/eda_v1_baseline_profiling.py

Data files expected in data/raw/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

plt.switch_backend("Agg")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv")
train_identity = pd.read_csv(DATA_DIR / "train_identity.csv")

print("train_transaction shape:", train_transaction.shape)
print("train_identity shape:", train_identity.shape)

df = train_transaction.merge(train_identity, on="TransactionID", how="left")
print("merged df shape:", df.shape)

# ============================================================================
# SECTION A — DATA PROFILING
# ============================================================================

print("\n" + "=" * 72)
print("SECTION A — DATA PROFILING")
print("=" * 72)

# ----------------------------------------------------------------------------
# A1: Shape and join quality
# ----------------------------------------------------------------------------

print("\n--- A1: Shape and join quality ---")

transaction_ids = set(train_transaction["TransactionID"])
identity_ids = set(train_identity["TransactionID"])
unmatched_identity_rows = (~train_identity["TransactionID"].isin(transaction_ids)).sum()
transactions_with_identity = train_transaction["TransactionID"].isin(identity_ids).sum()
pct_with_identity = 100 * transactions_with_identity / len(train_transaction)

print(f"Total rows (merged): {len(df):,}")
print(f"Total columns: {df.shape[1]}")
print(f"Identity rows NOT matching any transaction: {unmatched_identity_rows:,}")
print(f"Transactions with identity data: {pct_with_identity:.2f}%")

# ----------------------------------------------------------------------------
# A2: Fraud rate and class imbalance
# ----------------------------------------------------------------------------

print("\n--- A2: Fraud rate and class imbalance ---")

fraud_counts = df["isFraud"].value_counts()
fraud_rates = df["isFraud"].value_counts(normalize=True)
print(fraud_rates)
fraud_rate = fraud_rates[1]
print(f"Fraud rate: {fraud_rate:.4%}")
print(f"Non-fraud count: {fraud_counts[0]:,}")
print(f"Fraud count: {fraud_counts[1]:,}")

always_not_fraud_acc = (df["isFraud"] == 0).mean()
print(f"Accuracy of always-predict-not-fraud: {always_not_fraud_acc:.4%}")

n_negative = fraud_counts[0]
n_positive = fraud_counts[1]
scale_pos_weight = n_negative / n_positive
print(f"scale_pos_weight (n_negative / n_positive): {scale_pos_weight:.2f}")

# ----------------------------------------------------------------------------
# A3: Missing values — quantity and pattern
# ----------------------------------------------------------------------------

print("\n--- A3: Missing values ---")

print("\nTop 30 columns by null count:")
print(df.isnull().sum().sort_values(ascending=False).head(30))

null_counts = df.isnull().sum()
n_cols = len(df.columns)
pct_gt_50 = 100 * (null_counts > 0.5 * len(df)).sum() / n_cols
pct_gt_90 = 100 * (null_counts > 0.9 * len(df)).sum() / n_cols
print(f"\nColumns with >50% nulls: {pct_gt_50:.1f}%")
print(f"Columns with >90% nulls: {pct_gt_90:.1f}%")

print("\nTop 20 columns by missingness rate in fraud (isFraud=1):")
missing_by_fraud = df.groupby("isFraud").apply(lambda x: x.isnull().mean()).T
print(missing_by_fraud.sort_values(1, ascending=False).head(20))

missing_gap = (missing_by_fraud[1] - missing_by_fraud[0]).abs()
print("\nTop 20 columns by missingness gap between fraud=1 and fraud=0:")
top_gap_cols = missing_gap.sort_values(ascending=False).head(20)
print(top_gap_cols)

print("\nBinary missingness indicators — correlation with isFraud (top 10 gap columns):")
top_10_gap_cols = missing_gap.sort_values(ascending=False).head(10).index
for col in top_10_gap_cols:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
    corr = df[f"{col}_missing"].corr(df["isFraud"])
    print(f"  {col}_missing: {corr:.4f}")

# ----------------------------------------------------------------------------
# A4: Transaction amount distribution
# ----------------------------------------------------------------------------

print("\n--- A4: Transaction amount distribution ---")

print(df.groupby("isFraud")["TransactionAmt"].describe())
print("\nQuantiles by isFraud:")
print(df.groupby("isFraud")["TransactionAmt"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [(0, "blue"), (1, "red")]:
    subset = df.loc[df["isFraud"] == label, "TransactionAmt"]
    ax.hist(subset, bins=100, alpha=0.5, label=f"isFraud={label}", color=color)
ax.set_yscale("log")
ax.set_xlabel("TransactionAmt")
ax.set_ylabel("Count (log scale)")
ax.set_title("Transaction Amount by Fraud Label")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "amt_distribution_by_fraud.png", dpi=120)
plt.close(fig)

print(f"\nSkewness of TransactionAmt overall: {df['TransactionAmt'].skew():.4f}")
for label in [0, 1]:
    s = df.loc[df["isFraud"] == label, "TransactionAmt"].skew()
    print(f"Skewness of TransactionAmt (isFraud={label}): {s:.4f}")

df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])
print(f"\nSkewness of log1p(TransactionAmt) overall: {df['log_TransactionAmt'].skew():.4f}")
for label in [0, 1]:
    s = df.loc[df["isFraud"] == label, "log_TransactionAmt"].skew()
    print(f"Skewness of log1p(TransactionAmt) (isFraud={label}): {s:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [(0, "blue"), (1, "red")]:
    subset = df.loc[df["isFraud"] == label, "log_TransactionAmt"]
    ax.hist(subset, bins=100, alpha=0.5, label=f"isFraud={label}", color=color)
ax.set_xlabel("log1p(TransactionAmt)")
ax.set_ylabel("Count")
ax.set_title("Log Transaction Amount by Fraud Label")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "log_amt_distribution_by_fraud.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# A5: Temporal structure — TransactionDT
# ----------------------------------------------------------------------------

print("\n--- A5: Temporal structure ---")

print(df["TransactionDT"].describe())
span_days = (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400
print(f"Dataset span: {span_days:.2f} days")

df["time_bin"] = pd.cut(df["TransactionDT"], bins=10, include_lowest=True)
time_bin_stats = df.groupby("time_bin", observed=True).agg(
    fraud_rate=("isFraud", "mean"),
    transaction_count=("TransactionID", "count"),
)
print("\nFraud rate and transaction count by time bin:")
print(time_bin_stats)

fig, ax = plt.subplots(figsize=(10, 6))
time_bin_stats["fraud_rate"].plot(kind="line", marker="o", ax=ax)
ax.set_xlabel("Time bin")
ax.set_ylabel("Fraud rate")
ax.set_title("Fraud Rate by Time Bin")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fraud_rate_by_time.png", dpi=120)
plt.close(fig)

# ============================================================================
# SECTION B — FEATURE EXPLORATION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B — FEATURE EXPLORATION")
print("=" * 72)

# ----------------------------------------------------------------------------
# B1: Categorical features — fraud rates by category
# ----------------------------------------------------------------------------

print("\n--- B1: Categorical fraud rates ---")

for col in ["ProductCD", "card4", "card6", "P_emaildomain"]:
    table = (
        df.groupby(col)["isFraud"]
        .agg(["count", "mean"])
        .rename(columns={"count": "total", "mean": "fraud_rate"})
        .sort_values("fraud_rate", ascending=False)
    )
    table["low_count"] = table["total"] < 100
    if col == "P_emaildomain":
        table = table.head(20)
    print(f"\n{col}:")
    print(table)

# ----------------------------------------------------------------------------
# B2: Email domain mismatch
# ----------------------------------------------------------------------------

print("\n--- B2: Email domain mismatch ---")

both_present = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
df["email_match"] = np.where(
    both_present,
    (df["P_emaildomain"] == df["R_emaildomain"]).astype(int),
    np.nan,
)

email_match_stats = df.groupby("email_match", dropna=False)["isFraud"].agg(
    fraud_rate="mean", count="count"
)
print("Fraud rate and count by email_match (1=match, 0=mismatch, NaN=missing):")
print(email_match_stats)

print("\nComparison: email_match pools all domains into one binary feature:")
for val in [1.0, 0.0]:
    row = email_match_stats.loc[val]
    print(f"  email_match={int(val)}: fraud_rate={row['fraud_rate']:.4%}, count={int(row['count']):,}")

# ----------------------------------------------------------------------------
# B3: V-feature correlations with fraud
# ----------------------------------------------------------------------------

print("\n--- B3: V-feature correlations ---")

v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
v_corr = df[v_cols + ["isFraud"]].corr(numeric_only=True)["isFraud"].drop("isFraud").abs()
v_corr_sorted = v_corr.sort_values(ascending=False)

print("Top 20 V-features by absolute correlation with isFraud:")
print(v_corr_sorted.head(20))

for threshold in [0.1, 0.2, 0.3]:
    n = (v_corr > threshold).sum()
    print(f"V-features with abs_corr > {threshold}: {n}")

top_5_v = v_corr_sorted.head(5).index.tolist()
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, top_5_v):
    for label, color in [(0, "blue"), (1, "red")]:
        subset = df.loc[df["isFraud"] == label, col].dropna()
        ax.hist(subset, bins=100, alpha=0.5, label=f"isFraud={label}", color=color)
    ax.set_title(col)
    ax.legend(fontsize=8)
fig.suptitle("Top 5 V-Features by Fraud Correlation")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "top_v_features_by_fraud.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# B4: Card frequency and fraud rate
# ----------------------------------------------------------------------------

print("\n--- B4: Card frequency and fraud rate ---")

card1_stats = (
    df.groupby("card1")["isFraud"]
    .agg(count="count", fraud_rate="mean")
    .sort_values("fraud_rate", ascending=False)
)
print("Top 20 card1 values by fraud rate:")
print(card1_stats.head(20))

for label, mask in [
    ("<5 transactions", card1_stats["count"] < 5),
    ("5-49 transactions", (card1_stats["count"] >= 5) & (card1_stats["count"] < 50)),
    (">=50 transactions", card1_stats["count"] >= 50),
]:
    median_fr = card1_stats.loc[mask, "fraud_rate"].median()
    print(f"Median fraud rate for card1 values with {label}: {median_fr:.4%}")

card1_freq_map = df["card1"].value_counts()
df["card1_freq"] = df["card1"].map(card1_freq_map)
print(f"\nCorrelation of card1_freq with isFraud: {df['card1_freq'].corr(df['isFraud']):.4f}")

# ============================================================================
# SECTION B-EXT — EXHAUSTIVE FEATURE SCREENING
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B-EXT — EXHAUSTIVE FEATURE SCREENING")
print("=" * 72)

# ----------------------------------------------------------------------------
# B-EXT-1: Numeric features — correlation filter + distribution check
# ----------------------------------------------------------------------------

print("\n--- B-EXT-1: Numeric feature screening ---")


def count_histogram_peaks(series, n_bins=50):
    counts, _ = np.histogram(series.dropna(), bins=n_bins)
    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1] and counts[i] > 0:
            peaks += 1
    return peaks


exclude_cols = {"TransactionID", "TransactionDT", "isFraud"}
numeric_cols = [
    c
    for c in df.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols
]

numeric_summary_rows = []
for col in numeric_cols:
    valid = df[[col, "isFraud"]].dropna()
    if len(valid) == 0:
        abs_corr = np.nan
    else:
        abs_corr = abs(valid[col].corr(valid["isFraud"]))
    numeric_summary_rows.append(
        {
            "feature": col,
            "abs_corr": abs_corr,
            "null_rate": df[col].isnull().mean(),
            "skewness": df[col].skew() if df[col].notna().any() else np.nan,
        }
    )

numeric_summary = pd.DataFrame(numeric_summary_rows).sort_values("abs_corr", ascending=False)
print("Full numeric feature summary (sorted by abs_corr):")
print(numeric_summary.to_string(index=False))

filtered = numeric_summary[numeric_summary["abs_corr"] > 0.1].copy()
filtered["kurtosis"] = filtered["feature"].apply(
    lambda c: df[c].kurtosis() if df[c].notna().any() else np.nan
)
filtered["heavy_tailed"] = filtered["kurtosis"] > 10
filtered["bimodal"] = filtered["feature"].apply(lambda c: count_histogram_peaks(df[c]) >= 2)

print(f"\nFeatures passing correlation filter (abs_corr > 0.1): {len(filtered)}")
print(filtered.to_string(index=False))

n_features_passing_filter = len(filtered)
top_5_features = numeric_summary.head(5)["feature"].tolist()

# ----------------------------------------------------------------------------
# B-EXT-2: Numeric features passing filter — distribution plots
# ----------------------------------------------------------------------------

print("\n--- B-EXT-2: Top numeric feature distributions ---")

v_set = set(v_cols)
top_15_non_v = numeric_summary[~numeric_summary["feature"].isin(v_set)].head(15)["feature"].tolist()

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
for ax, col in zip(axes, top_15_non_v):
    for label, color in [(0, "blue"), (1, "red")]:
        subset = df.loc[df["isFraud"] == label, col].dropna()
        ax.hist(subset, bins=50, alpha=0.5, label=f"isFraud={label}", color=color)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle("Top 15 Non-V Numeric Features by Fraud Correlation")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "top_numeric_features_by_fraud.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# B-EXT-3: Categorical feature screening
# ----------------------------------------------------------------------------

print("\n--- B-EXT-3: Categorical feature screening ---")


def rarest_10pct_fraud_rate(series, target):
    counts = series.value_counts()
    n_rare = max(1, int(len(counts) * 0.1))
    rare_values = counts.tail(n_rare).index
    mask = series.isin(rare_values)
    if mask.sum() == 0:
        return np.nan
    return target[mask].mean()


cat_rows = []
for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    most_common = df[col].mode(dropna=True)
    if len(most_common) == 0:
        mc_fraud_rate = np.nan
    else:
        mc_fraud_rate = df.loc[df[col] == most_common.iloc[0], "isFraud"].mean()
    cat_rows.append(
        {
            "feature": col,
            "unique_count": df[col].nunique(dropna=True),
            "null_rate": df[col].isnull().mean(),
            "most_common_fraud_rate": mc_fraud_rate,
            "rarest_10pct_fraud_rate": rarest_10pct_fraud_rate(df[col], df["isFraud"]),
            "high_cardinality": df[col].nunique(dropna=True) > 1000,
        }
    )

cat_summary = pd.DataFrame(cat_rows).sort_values("unique_count", ascending=False)
print(cat_summary.to_string(index=False))

# ----------------------------------------------------------------------------
# B-EXT-4: Temporal stability of top features
# ----------------------------------------------------------------------------

print("\n--- B-EXT-4: Temporal stability ---")

df["time_bin_5"] = pd.qcut(df["TransactionDT"], q=5, duplicates="drop")
top_10_numeric = numeric_summary.head(10)["feature"].tolist()

stability_rows = []
unstable_features = []
for feat in top_10_numeric:
    row = {"feature": feat}
    for i, tb in enumerate(sorted(df["time_bin_5"].dropna().unique())):
        subset = df[df["time_bin_5"] == tb][[feat, "isFraud"]].dropna()
        if len(subset) < 2:
            corr_val = np.nan
        else:
            corr_val = subset[feat].corr(subset["isFraud"])
        row[f"bin_{i}"] = corr_val
        if not np.isnan(corr_val) and (corr_val < 0.05 or corr_val < 0):
            unstable_features.append((feat, i, corr_val))
    stability_rows.append(row)

stability_df = pd.DataFrame(stability_rows).set_index("feature")
print("Correlation with isFraud by time bin (top 10 features):")
print(stability_df.to_string())
if unstable_features:
    print("\nFeatures with correlation < 0.05 or sign flip in at least one bin:")
    for feat, bin_idx, corr_val in unstable_features:
        print(f"  {feat} in bin_{bin_idx}: corr={corr_val:.4f}")
else:
    print("\nNo features flagged for instability (corr < 0.05 or sign flip).")

# ============================================================================
# SECTION B-UID — UID EXPLORATION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B-UID — UID EXPLORATION")
print("=" * 72)

# ----------------------------------------------------------------------------
# B-UID-1: Construct candidate UIDs
# ----------------------------------------------------------------------------

print("\n--- B-UID-1: Candidate UIDs ---")


def to_uid_str(val):
    if pd.isna(val):
        return "nan"
    return str(val)


df["UID"] = (
    df["card1"].apply(to_uid_str)
    + "_"
    + df["addr1"].apply(to_uid_str)
    + "_"
    + df["P_emaildomain"].apply(to_uid_str)
)

uid_counts = df["UID"].value_counts()
n_unique_uids = df["UID"].nunique()
median_tx_per_uid = uid_counts.median()
max_tx_per_uid = uid_counts.max()
pct_single_tx_uids = 100 * (uid_counts == 1).sum() / n_unique_uids

print(f"Unique UIDs: {n_unique_uids:,}")
print(f"Median transactions per UID: {median_tx_per_uid:.0f}")
print(f"Max transactions per UID: {max_tx_per_uid:,}")
print(f"UIDs with only 1 transaction: {pct_single_tx_uids:.2f}%")

# ----------------------------------------------------------------------------
# B-UID-2: UID-level fraud patterns
# ----------------------------------------------------------------------------

print("\n--- B-UID-2: UID-level fraud patterns ---")

uid_stats = df.groupby("UID").agg(
    transaction_count=("TransactionID", "count"),
    fraud_count=("isFraud", "sum"),
    fraud_rate=("isFraud", "mean"),
    total_spend=("TransactionAmt", "sum"),
    mean_spend=("TransactionAmt", "mean"),
    std_spend=("TransactionAmt", "std"),
)
print("Top 20 UIDs by fraud count:")
print(uid_stats.sort_values("fraud_count", ascending=False).head(20))

for label, mask in [
    ("1 transaction", uid_stats["transaction_count"] == 1),
    ("2-5 transactions", (uid_stats["transaction_count"] >= 2) & (uid_stats["transaction_count"] <= 5)),
    ("6-20 transactions", (uid_stats["transaction_count"] >= 6) & (uid_stats["transaction_count"] <= 20)),
    (">20 transactions", uid_stats["transaction_count"] > 20),
]:
    median_fr = uid_stats.loc[mask, "fraud_rate"].median()
    print(f"Median fraud rate for UIDs with {label}: {median_fr:.4%}")

# ----------------------------------------------------------------------------
# B-UID-3: Behavioral features from UID grouping
# ----------------------------------------------------------------------------

print("\n--- B-UID-3: UID behavioral features ---")

df_sorted = df.sort_values(["UID", "TransactionDT"]).copy()
df_sorted["uid_cumcount"] = df_sorted.groupby("UID").cumcount() + 1
df_sorted["uid_cummean_amt"] = df_sorted.groupby("UID")["TransactionAmt"].transform(
    lambda x: x.expanding().mean()
)
df_sorted["time_since_last_tx"] = df_sorted.groupby("UID")["TransactionDT"].diff()

for col in ["uid_cumcount", "uid_cummean_amt", "time_since_last_tx"]:
    df[col] = df_sorted[col].values
    corr = df[col].corr(df["isFraud"])
    print(f"Correlation of {col} with isFraud: {corr:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [(0, "blue"), (1, "red")]:
    subset = df.loc[df["isFraud"] == label, "time_since_last_tx"].dropna()
    ax.hist(subset, bins=50, alpha=0.5, label=f"isFraud={label}", color=color)
ax.set_xlabel("Time since last transaction (seconds)")
ax.set_ylabel("Count")
ax.set_title("Time Since Last Transaction by Fraud Label")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "uid_time_since_last_by_fraud.png", dpi=120)
plt.close(fig)

# ============================================================================
# SECTION B-FREQ — FREQUENCY ENCODING EXPLORATION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B-FREQ — FREQUENCY ENCODING")
print("=" * 72)

for col in ["card1", "addr1", "P_emaildomain", "card2"]:
    freq_map = df[col].value_counts()
    freq_col = f"{col}_freq"
    df[freq_col] = df[col].map(freq_map)
    corr = df[freq_col].corr(df["isFraud"])
    print(f"Correlation of {freq_col} with isFraud: {corr:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [(0, "blue"), (1, "red")]:
    subset = df.loc[df["isFraud"] == label, "card1_freq"].dropna()
    ax.hist(subset, bins=50, alpha=0.5, label=f"isFraud={label}", color=color)
ax.set_xscale("log")
ax.set_xlabel("card1_freq (log scale)")
ax.set_ylabel("Count")
ax.set_title("card1 Frequency by Fraud Label")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "card1_freq_by_fraud.png", dpi=120)
plt.close(fig)

# ============================================================================
# SECTION C — EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 72)
print("SECTION C — EVALUATION METRICS")
print("=" * 72)

print("\n--- C1: Baseline AUC-PR and AUC-ROC ---")

y_true = df["isFraud"].values
n = len(y_true)

pred_zeros = np.zeros(n)
pred_ones = np.ones(n)
auc_pr_zeros = average_precision_score(y_true, pred_zeros)
auc_pr_ones = average_precision_score(y_true, pred_ones)
print(f"AUC-PR (always predict 0): {auc_pr_zeros:.6f}")
print(f"AUC-PR (always predict 1): {auc_pr_ones:.6f}")

rng = np.random.default_rng(42)
auc_pr_random = []
auc_roc_random = []
for _ in range(100):
    pred_random = rng.uniform(0, 1, size=n)
    auc_pr_random.append(average_precision_score(y_true, pred_random))
    auc_roc_random.append(roc_auc_score(y_true, pred_random))

auc_pr_random_mean = np.mean(auc_pr_random)
auc_pr_random_std = np.std(auc_pr_random)
auc_roc_random_mean = np.mean(auc_roc_random)
auc_roc_random_std = np.std(auc_roc_random)

print(f"AUC-PR (random uniform, 100 runs): {auc_pr_random_mean:.6f} ± {auc_pr_random_std:.6f}")
print(f"AUC-ROC (random uniform, 100 runs): {auc_roc_random_mean:.6f} ± {auc_roc_random_std:.6f}")
print(f'AUC-PR floor (random): ~{fraud_rate:.4f}')
print("AUC-ROC floor (random): ~0.5")
print("AUC-PR target: >0.80")

print("\n--- C2: Precision-Recall tradeoff ---")

rng_scores = np.random.default_rng(123)
fraud_scores = rng_scores.beta(2, 5, size=n_positive)
legit_scores = rng_scores.beta(1, 8, size=n_negative)
sim_scores = np.empty(n)
sim_scores[y_true == 1] = fraud_scores
sim_scores[y_true == 0] = legit_scores

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(legit_scores, bins=50, alpha=0.5, label="Legitimate (Beta(1,8))", color="blue")
ax.hist(fraud_scores, bins=50, alpha=0.5, label="Fraud (Beta(2,5))", color="red")
ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.5")
ax.set_xlabel("Score")
ax.set_ylabel("Count")
ax.set_title("Simulated Score Distribution")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "score_distribution_example.png", dpi=120)
plt.close(fig)

thresholds = np.linspace(0.05, 0.95, 20)
pr_table_rows = []
for t in thresholds:
    pred = (sim_scores >= t).astype(int)
    pr_table_rows.append(
        {
            "threshold": round(t, 4),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        }
    )
pr_table = pd.DataFrame(pr_table_rows)
print("\nPrecision-Recall at thresholds (simulated scores):")
print(pr_table.to_string(index=False))

# ============================================================================
# SECTION D — HYPOTHESES
# ============================================================================

print("\n" + "=" * 72)
print("SECTION D — HYPOTHESES")
print("=" * 72)

top_5_predicted = numeric_summary.head(5)
hypothesis_reasons = {
    "V-features": "Anonymized Vesta features show strongest linear correlation with fraud.",
    "email_match": "P/R email domain mismatch is a high-signal behavioral indicator.",
    "card1_freq": "Card rarity (low frequency) correlates with elevated fraud risk.",
    "TransactionAmt": "Fraud transactions skew toward higher amounts with heavy tails.",
    "uid_cumcount": "Burst activity within a UID (cumulative transaction count) flags fraud.",
}

print("\nTOP 5 PREDICTED IMPORTANT FEATURES:")
for i, row in enumerate(top_5_predicted.itertuples(), 1):
    feat = row.feature
    reason = hypothesis_reasons.get(
        feat,
        f"High abs correlation ({row.abs_corr:.3f}) with isFraud in EDA screening.",
    )
    print(f"  {i}. {feat}: {reason}")

print("\nBIGGEST DATA CHALLENGE:")
print(
    "  Extreme class imbalance (~3.5% fraud) combined with >50% missing values "
    "across many identity and device columns limits supervised signal."
)

print("\nHARDEST FEATURE ENGINEERING DECISION:")
print(
    "  Whether to treat missingness as signal (binary indicators) vs impute/fill, "
    "given missingness rates differ systematically between fraud and non-fraud rows."
)

print("\nAUC-PR TARGET: 0.80")

# ============================================================================
# SECTION E — PROJECT PLAN
# ============================================================================

print("\n" + "=" * 72)
print("SECTION E — PROJECT PLAN")
print("=" * 72)

print("""
Model: LightGBM
Split: Temporal on TransactionDT, first 80% train, last 20% test
Primary metric: AUC-PR
Pipeline: raw JSON → feature engineering → LightGBM predict → fraud probability in <150ms
Demo target: live FastAPI endpoint with documented metrics beating naive baseline

Interviewer summary:
Built an end-to-end fraud detection pipeline on IEEE-CIS e-commerce data using
LightGBM with temporal validation and AUC-PR as the primary metric, achieving
strong lift over random and always-negative baselines via V-feature screening,
missingness indicators, and frequency-encoded identity features.
""")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

summary = pd.DataFrame(
    [
        {"metric": "Total rows", "value": f"{len(df):,}"},
        {"metric": "Total columns", "value": str(df.shape[1])},
        {"metric": "Fraud rate", "value": f"{fraud_rate:.4%}"},
        {"metric": "Fraud count", "value": f"{fraud_counts[1]:,}"},
        {"metric": "Dataset span (days)", "value": f"{span_days:.2f}"},
        {"metric": "Identity match rate", "value": f"{pct_with_identity:.2f}%"},
        {
            "metric": "Top 5 features (abs corr)",
            "value": ", ".join(top_5_features),
        },
        {
            "metric": "Features passing corr filter (>0.1)",
            "value": str(n_features_passing_filter),
        },
        {"metric": "Unique UIDs", "value": f"{n_unique_uids:,}"},
        {"metric": "Median transactions per UID", "value": f"{median_tx_per_uid:.0f}"},
        {
            "metric": "AUC-PR baseline (random)",
            "value": f"{auc_pr_random_mean:.4f} ± {auc_pr_random_std:.4f}",
        },
        {"metric": "AUC-PR target", "value": ">0.80"},
        {"metric": "scale_pos_weight", "value": f"{scale_pos_weight:.2f}"},
    ]
)
print(summary.to_string(index=False))

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("Week 1 EDA Complete")
    print("=" * 72)
