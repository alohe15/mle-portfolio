# Artifact handoff — lgbm_v9 / dataset_v5

The serving image (`Dockerfile`) copies a fixed set of files and does not
include `data/`. Gitignored binaries are not in git. This document is the
handoff for every file `models/registry.json` points at for the serving
entry and the v9 Candidate entry, plus the data files the in-container
parity test mounts.

The full handoff checksums live in `artifacts.sha256` at the repo root
(committed). Verify from the repo root before building:

```bash
shasum -a 256 -c artifacts.sha256
```

The serving image uses a smaller startup manifest,
`artifacts.serving.sha256`, containing only files baked into the serving
container. Startup (`services/api/model_loader.py`) requires that file and
checks every listed path **before** it opens the booster or any pickle. A
missing checksum manifest, missing artifact, or digest mismatch raises
`ArtifactChecksumError`. The API lifespan logs that error and does not load
the model, so `/predict` is not served from the wrong files.

## Gitignored files required to build the image

These three files are listed in `.gitignore` (`models/**/*.txt`,
`models/**/*.pkl`). `Dockerfile` copies them explicitly. A fresh clone does
not contain them.

| Path | What it is | How it was produced | Bytes | SHA-256 |
|---|---|---|---|---|
| `models/lgbm_v9_optuna_tuning.txt` | LightGBM booster for lgbm_v9 | `python scripts/train.py --config configs/lgbm_v9.json` (v9 training on dataset_v5) | 250054017 | `a117da49c186abc8721585719c557f12b772a657291b48867e612a890576a01b` |
| `models/lgbm_v9_optuna_tuning_fitted_transforms.pkl` | Frozen `requires_fit` state (frequency encoders and UID aggregations). The API never refits these. | Same `scripts/train.py` run. Written only because the dataset config chain contains `requires_fit` transforms. | 1287519 | `7f7295336ed28553f11207e5998363aefa3347363dea9ebfcd05854f08a9e17c` |
| `models/lgbm_v9_calibrator.pkl` | Isotonic calibrator mapping raw scores to probabilities | `python evals/calibrate.py` after lgbm_v9 exists (fit on the validation split) | 2430 | `1fabce507f509862cd02c1de92bf0141bd5a2f29718031df54ea909cce7213d1` |

Registry fields: `model_path`, `fitted_transforms_path` (Candidate),
`calibrator_path` (Candidate).

## Committed registry files (also checksummed)

These arrive with `git clone`. They are in `artifacts.sha256` so startup
verification covers every path on the serving and Candidate entries, not
only the gitignored binaries.

| Path | What it is | How it was produced | Bytes | SHA-256 |
|---|---|---|---|---|
| `models/lgbm_v9_optuna_tuning_manifest.json` | Tuning manifest / feature contract (`feature_order`, 464 columns). Registry `manifest_path` and `tuning_manifest_path`. | `scripts/train.py --config configs/lgbm_v9.json` | 17131 | `f1deba7a88159637103fd54202000236ca55d5ce71cbbb50f36a3d3400cfbe6f` |
| `models/lgbm_v9_optuna_tuning_metrics.json` | Test-split metrics for v9 | Same training run | 839 | `b01bb7350e9fdb53317a43ab6ea1d3f20982b0e01aef32bd6db60122dcf55b71` |
| `configs/lgbm_v9.json` | Model config (split, hyperparameters) | Authored for the v9 training run; committed | 1081 | `33dd50c8c0f52800de1cadfe549b895902d3b32b0e77b14d0539e6cd160b2df7` |
| `configs/dataset_v5.json` | Dataset config (transform chain the API replays) | Authored for dataset_v5; committed | 3029 | `0020fd6721ada467a497aff15f33e86160a7ea7faa9d07c592e3732ec3838050` |
| `configs/decision_policy_v1.json` | Cost parameters for the argmin action | Phase 3 decision policy; committed | 424 | `db89aefd20c4b83ac55f473ae3670e2299fcfdd9bfa98eef0926b31dfc8e3d5e` |
| `data/processed/dataset_v5_dn_only_manifest.json` | Dataset manifest for dataset_v5 | `python scripts/build_dataset.py --config configs/dataset_v5.json` | 49867 | `68a77e198e64538b1e170d6bf8679bbcc3d2106cb9dcc5e6f13468f2f1af91b9` |

`models/registry.json` itself is committed and is not a checksum target.
Exactly one entry has `is_serving: true` (version 9, dataset_version 5).

## Data files for the parity test (not in the image)

`docker compose run --rm test` mounts `./data` read-only at `/app/data`.
These parquets are gitignored and are **not** in `artifacts.sha256` (they
are not registry artifact paths and are not copied by `Dockerfile`).

| Path | What it is | How to produce it |
|---|---|---|
| `data/processed/dataset_v5_dn_only.parquet` | dataset_v5 feature table | `python scripts/build_dataset.py --config configs/dataset_v5.json` |
| `data/raw/train_merged.parquet` | Raw merged transactions the parity test joins back to | `python scripts/merge_train_data.py` (or the copy you already use for dataset builds) |

## Fresh clone

1. Clone the repo. You get configs, the manifest, metrics, `models/registry.json`, `artifacts.sha256`, and `artifacts.serving.sha256`. You do not get the three gitignored binaries or the parquets.
2. Either run the scripts below, or copy the three binaries from the model owner. Place them at the paths in the table. Do not rename them.
   ```bash
   python scripts/build_dataset.py --config configs/dataset_v5.json
   python scripts/train.py --config configs/lgbm_v9.json
   python evals/calibrate.py
   ```
   Training rewrites the booster, the fitted-transforms pickle, the manifest, and the metrics. After a retrain the digests change; update `artifacts.sha256`, `artifacts.serving.sha256`, and this document from the new files. Do not reuse a version number.
3. Confirm every digest before building:
   ```bash
   shasum -a 256 -c artifacts.sha256
   shasum -a 256 -c artifacts.serving.sha256
   ```
4. Build the serving image (lean: no pytest, no `tests/`, no `data/`):
   ```bash
   docker compose build fraud-api
   ```
5. Run training-vs-serving parity and the endpoint tests inside the test image:
   ```bash
   docker compose run --rm test
   ```
   That service builds `Dockerfile.test`, which extends the serving image, installs `requirements-test.txt` (`pytest`, `httpx`), copies `tests/`, and mounts `data/` read-only. It also copies `configs/lgbm_v9.json`, which the parity test needs to rebuild the temporal split and which the serving image does not carry. The command is `python -m pytest tests/test_training_serving_parity.py tests/test_api_endpoint.py -v`.
