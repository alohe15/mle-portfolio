# Fraud API image — only files needed to serve lgbm_v9 / dataset_v5.
#
# Startup file trace (services/api/model_loader.py + imports):
#   models/registry.json
#   models/lgbm_v9_optuna_tuning_manifest.json   (feature_order)
#   models/lgbm_v9_optuna_tuning.txt             (Booster)
#   models/lgbm_v9_optuna_tuning_fitted_transforms.pkl
#   models/lgbm_v9_calibrator.pkl                (Candidate entry)
#   configs/decision_policy_v1.json              (Candidate entry)
#   configs/dataset_v5.json + inherited v3→v2→v1 (collect_all_transformations)
#   scripts/train.py, scripts/feature_engineering.py
#   evals/cost_model.py                          (predictor → optimal_action)
#   services/api/*

FROM python:3.11-slim

WORKDIR /app

# LightGBM needs libgomp; curl is used by compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Committed source (config-driven serving path)
COPY artifacts.serving.sha256 artifacts.serving.sha256
COPY scripts/ scripts/
COPY services/ services/
COPY evals/ evals/
COPY configs/lgbm_v9.json configs/lgbm_v9.json
COPY configs/dataset_v1.json configs/dataset_v1.json
COPY configs/dataset_v2.json configs/dataset_v2.json
COPY configs/dataset_v3.json configs/dataset_v3.json
COPY configs/dataset_v5.json configs/dataset_v5.json
COPY configs/decision_policy_v1.json configs/decision_policy_v1.json
COPY models/registry.json models/registry.json

# Gitignored artifacts required at startup (explicit — no models/** wildcard)
COPY models/lgbm_v9_optuna_tuning.txt models/lgbm_v9_optuna_tuning.txt
COPY models/lgbm_v9_optuna_tuning_manifest.json models/lgbm_v9_optuna_tuning_manifest.json
COPY models/lgbm_v9_optuna_tuning_fitted_transforms.pkl models/lgbm_v9_optuna_tuning_fitted_transforms.pkl
COPY models/lgbm_v9_calibrator.pkl models/lgbm_v9_calibrator.pkl

EXPOSE 8000

CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
