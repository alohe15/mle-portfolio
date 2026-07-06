from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

train_transaction = pd.read_csv(RAW_DIR / "train_transaction.csv")
train_identity = pd.read_csv(RAW_DIR / "train_identity.csv")

merged = train_transaction.merge(train_identity, on="TransactionID", how="left")

output_path = RAW_DIR / "train_merged.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(output_path, index=False)

print(f"Saved merged dataset to {output_path}")
print(f"Shape: {merged.shape}")
