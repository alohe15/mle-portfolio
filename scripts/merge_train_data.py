from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "project-1-detect-fraud" / "data"
OUTPUT_DIR = REPO_ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "train_merged.parquet"

train_transaction = pd.read_csv(SOURCE_DIR / "train_transaction.csv")
train_identity = pd.read_csv(SOURCE_DIR / "train_identity.csv")

merged = train_transaction.merge(train_identity, on="TransactionID", how="left")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
merged.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved merged dataset to {OUTPUT_PATH}")
print(f"Shape: {merged.shape}")
