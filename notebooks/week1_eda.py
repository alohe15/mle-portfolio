from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "project-1-detect-fraud" / "data"

train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv")
train_identity = pd.read_csv(DATA_DIR / "train_identity.csv")

merged = train_transaction.merge(train_identity, on="TransactionID", how="left")

transaction_ids = set(train_transaction["TransactionID"])
identity_ids = set(train_identity["TransactionID"])

unmatched_identity_rows = (~train_identity["TransactionID"].isin(transaction_ids)).sum()
transactions_with_identity = train_transaction["TransactionID"].isin(identity_ids).sum()
pct_with_identity = 100 * transactions_with_identity / len(train_transaction)

print(f"Shape: {merged.shape}")
print(f"Unmatched identity rows: {unmatched_identity_rows}")
print(f"Transactions with identity data: {pct_with_identity:.2f}%")

df = merged
fraud_counts = df["isFraud"].value_counts(normalize=True)
print(fraud_counts)
print(f"Fraud rate: {fraud_counts[1]:.4%}")

print("\nTop 30 columns by missing count:")
print(df.isnull().sum().sort_values(ascending=False).head(30))

print("\nTop 20 columns by missingness rate in fraud (isFraud=1):")
print(
    df.groupby("isFraud")
    .apply(lambda x: x.isnull().mean())
    .T.sort_values(1, ascending=False)
    .head(20)
)

print("\nTransactionAmt describe by isFraud:")
print(df.groupby("isFraud")["TransactionAmt"].describe())

print("\nTransactionAmt quantiles by isFraud:")
print(
    df.groupby("isFraud")["TransactionAmt"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
)

print("\nTransactionDT describe:")
print(df["TransactionDT"].describe())

span_days = (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400
print(f"\nDataset span: {span_days:.2f} days")

df["day"] = (df["TransactionDT"] - df["TransactionDT"].min()) // 86400
daily_fraud_rate = df.groupby("day")["isFraud"].mean()
print("\nDaily fraud rate summary:")
print(daily_fraud_rate.describe())
print(f"Min daily fraud rate: {daily_fraud_rate.min():.4%} (day {daily_fraud_rate.idxmin()})")
print(f"Max daily fraud rate: {daily_fraud_rate.max():.4%} (day {daily_fraud_rate.idxmax()})")

for col in ["ProductCD", "card4", "card6", "P_emaildomain"]:
    print(f"\nFraud rate by {col}:")
    print(
        df.groupby([col, "isFraud"])
        .size()
        .unstack()
        .assign(fraud_rate=lambda x: x[1] / (x[0] + x[1]))
        .sort_values("fraud_rate", ascending=False)
    )

df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

print("\nFraud rate by email_match:")
print(
    df.groupby("email_match")["isFraud"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "fraud_rate"})
)

both_emails = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
print("\nFraud rate by email_match (both domains present only):")
print(
    df[both_emails]
    .groupby("email_match")["isFraud"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "fraud_rate"})
)
