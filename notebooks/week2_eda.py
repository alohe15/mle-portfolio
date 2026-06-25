from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "project-1-detect-fraud" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv")
train_identity = pd.read_csv(DATA_DIR / "train_identity.csv")

df = train_transaction.merge(train_identity, on="TransactionID", how="left")

# 1. NUMERIC FEATURES — Correlation ranking
exclude = {"isFraud", "TransactionID"}
numeric_cols = [
    c
    for c in df.select_dtypes(include="number").columns
    if c not in exclude
]

corr_with_fraud = (
    df[numeric_cols + ["isFraud"]]
    .corr(numeric_only=True)["isFraud"]
    .drop("isFraud")
    .abs()
    .sort_values(ascending=False)
)

print("Top 30 numeric features by absolute correlation with isFraud:")
for feature, corr in corr_with_fraud.head(30).items():
    signed = df[[feature, "isFraud"]].corr(numeric_only=True).loc[feature, "isFraud"]
    print(f"  {feature}: {signed:+.4f} (|corr|={corr:.4f})")

# 2. CATEGORICAL FEATURES — Fraud rate ranking
object_cols = df.select_dtypes(include="object").columns


def max_fraud_rate_spread(col: str) -> float | None:
    level_stats = (
        df.groupby(col, observed=True)["isFraud"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
    )
    level_stats = level_stats[level_stats["count"] >= 100]
    if len(level_stats) < 2:
        return None
    return level_stats["fraud_rate"].max() - level_stats["fraud_rate"].min()


spread_rows = []
for col in object_cols:
    spread = max_fraud_rate_spread(col)
    if spread is not None:
        spread_rows.append({"feature": col, "fraud_rate_spread": spread})

categorical_spreads = (
    pd.DataFrame(spread_rows)
    .sort_values("fraud_rate_spread", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 10 categorical features by max fraud rate spread (levels with >= 100 tx):")
print(categorical_spreads.head(10).to_string(index=False))

# 3. HISTOGRAM PLOTS for top 20 numeric features
top20_features = corr_with_fraud.head(20).index.tolist()
top20_corrs = corr_with_fraud.head(20)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(5, 4, figsize=(20, 25))
axes = axes.flatten()

for ax, feature in zip(axes, top20_features):
    signed_corr = df[[feature, "isFraud"]].corr(numeric_only=True).loc[feature, "isFraud"]
    legit = df.loc[df["isFraud"] == 0, feature].dropna()
    fraud = df.loc[df["isFraud"] == 1, feature].dropna()

    ax.hist(legit, bins=50, density=True, alpha=0.5, label="isFraud=0")
    ax.hist(fraud, bins=50, density=True, alpha=0.5, label="isFraud=1")
    ax.set_title(f"{feature} (corr={signed_corr:+.4f})")
    ax.legend(fontsize=8)

plt.tight_layout()
plot_path = OUTPUT_DIR / "top20_numeric_distributions.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nSaved distribution plot to {plot_path}")


def interpret_feature(feature: str, abs_corr: float) -> tuple[str, list[str]]:
    legit = df.loc[df["isFraud"] == 0, feature].dropna()
    fraud = df.loc[df["isFraud"] == 1, feature].dropna()
    signed_corr = df[[feature, "isFraud"]].corr(numeric_only=True).loc[feature, "isFraud"]
    flags: list[str] = []

    if len(legit) < 10 or len(fraud) < 10:
        return (
            f"Feature {feature} (corr={signed_corr:+.4f}): insufficient non-null data for comparison.",
            flags,
        )

    med0, med1 = legit.median(), fraud.median()
    lo = min(legit.quantile(0.01), fraud.quantile(0.01))
    hi = max(legit.quantile(0.99), fraud.quantile(0.99))
    span = hi - lo

    if span <= 0:
        sentence = (
            f"Feature {feature} (corr={signed_corr:+.4f}): values are nearly constant across both classes."
        )
        flags.append("OVERLAP")
        return sentence, flags

    bins = np.linspace(lo, hi, 40)
    h0, _ = np.histogram(legit, bins=bins, density=True)
    h1, _ = np.histogram(fraud, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.minimum(h0, h1).sum() * bin_width)

    if overlap > 0.85 or abs_corr < 0.05:
        flags.append("OVERLAP")

    fraud_iqr = fraud.quantile(0.75) - fraud.quantile(0.25)
    if span > 0 and fraud_iqr / span < 0.15:
        flags.append("NARROW_BAND")

    if med1 > med0 * 1.1:
        direction = "fraud transactions skew toward higher values"
    elif med1 < med0 * 0.9:
        direction = "fraud transactions skew toward lower values"
    elif overlap > 0.85:
        direction = "distributions overlap almost completely despite modest correlation"
    else:
        direction = "medians are similar but shape or tail behavior differs"

    sentence = (
        f"Feature {feature} (corr={signed_corr:+.4f}): {direction} "
        f"(legit median={med0:.4g}, fraud median={med1:.4g})."
    )
    return sentence, flags


# 4. INTERPRETATION SUMMARY
print("\nInterpretation summary (top 20 numeric features):")
for feature in top20_features:
    sentence, flags = interpret_feature(feature, top20_corrs[feature])
    flag_text = ""
    if "OVERLAP" in flags:
        flag_text += " [FLAG: distributions overlap almost completely — likely low predictive value]"
    if "NARROW_BAND" in flags:
        flag_text += " [FLAG: fraud concentrates in a narrow band — potential threshold signal]"
    print(f"  {sentence}{flag_text}")
