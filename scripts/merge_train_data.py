from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_config import load_config, repo_path

paths = load_config("paths")

train_transaction = pd.read_csv(repo_path(paths["train_transaction_csv"]))
train_identity = pd.read_csv(repo_path(paths["train_identity_csv"]))

merged = train_transaction.merge(train_identity, on="TransactionID", how="left")

output_path = repo_path(paths["train_merged_parquet"])
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(output_path, index=False)

print(f"Saved merged dataset to {output_path}")
print(f"Shape: {merged.shape}")
