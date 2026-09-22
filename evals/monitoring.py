"""Pure monitoring checks plus registry-driven scoring helpers.

Check functions have no real-data / model-file dependency so unit tests can
run on synthetic frames. Scoring helpers load the serving stack the same way
the API does (models/registry.json → model_loader.load_serving_artifacts).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
API_DIR = REPO_ROOT / "services" / "api"
DEFAULT_CONFIG_PATH = EVALS_DIR / "monitoring_config.json"

ACTION_NAMES = ("approve", "step_up", "review", "decline")
CATCH_RATE_BY_ACTION = {
    "approve": 0.0,
    "step_up": "step_up_stop_rate",
    "review": "review_catch_rate",
    "decline": 1.0,
}

# TransactionRequest declared fields (services/api/schemas.py) minus metadata.
# Schema checks use this serving contract, not the training DataFrame.
SERVING_REQUEST_FIELDS = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "dist1",
    "dist2",
    "TransactionDT",
]
SERVING_CATEGORICALS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]
# Present on processed splits / logs but not part of the serving feature contract.
NON_SERVING_COLUMNS = {"isFraud", "TransactionID", "metadata"}

MISSING_SENTINEL = "__MISSING__"
OTHER_SENTINEL = "__OTHER__"
HIGH_CARDINALITY_KEEP = 100
N_PSI_BINS = 10
PSI_EPSILON = 1e-4
PERCENTILES = (5, 25, 50, 75, 95)


# ---------------------------------------------------------------------------
# JSON / config helpers
# ---------------------------------------------------------------------------

def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def dump_json(path: Path | str, doc: Any) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def load_monitoring_config(path: Path | str | None = None) -> dict:
    return load_json(path or DEFAULT_CONFIG_PATH)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Check record helpers
# ---------------------------------------------------------------------------

def make_check(
    name: str,
    *,
    tier: str,
    status: str,
    observed_value: Any,
    reference_value: Any,
    threshold: Any,
    severity: str,
    note: str = "",
    detail: Any = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "name": name,
        "tier": tier,
        "status": status,
        "observed_value": observed_value,
        "reference_value": reference_value,
        "threshold": threshold,
        "severity": severity,
    }
    if note:
        rec["note"] = note
    if detail is not None:
        rec["detail"] = detail
    return rec


def _triggered_status(triggered: bool, severity: str) -> str:
    return severity if triggered else "pass"


def overall_status(checks: list[dict]) -> str:
    """healthy / warning / critical from a list of check records."""
    statuses = {c.get("status") for c in checks if c.get("status") not in {"skipped", None}}
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "healthy"


def _cfg(config: dict, tier: str, key: str, default: dict | None = None) -> dict:
    block = config.get(tier) or {}
    if key in block:
        return block[key]
    return default or {"threshold": 0, "severity": "warning"}


# ---------------------------------------------------------------------------
# Dtypes / schema
# ---------------------------------------------------------------------------

def dtype_kind(dtype_like: Any) -> str:
    """Collapse pandas/numpy dtype names to numeric vs string vs bool vs datetime."""
    if isinstance(dtype_like, pd.Series):
        if pd.api.types.is_bool_dtype(dtype_like):
            return "bool"
        if pd.api.types.is_numeric_dtype(dtype_like):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(dtype_like):
            return "datetime"
        return "string"
    text = str(dtype_like).lower()
    if "bool" in text:
        return "bool"
    if any(token in text for token in ("int", "float", "uint", "double", "number")):
        return "numeric"
    if "datetime" in text or "date" in text:
        return "datetime"
    return "string"


def check_schema(
    comparison_df: pd.DataFrame,
    reference_schema: dict,
    config: dict,
    *,
    ignore_extra: set[str] | None = None,
) -> list[dict]:
    """Compare comparison columns/dtypes against the reference schema contract."""
    expected_cols = list(reference_schema["columns"])
    expected_set = set(expected_cols)
    observed_cols = list(comparison_df.columns)
    ignore = set(ignore_extra or ())

    missing = [c for c in expected_cols if c not in comparison_df.columns]
    extra = [
        c
        for c in observed_cols
        if c not in expected_set and c not in ignore
    ]

    expected_dtypes = dict(reference_schema.get("dtypes") or {})
    mismatches: list[dict[str, str]] = []
    for col in expected_cols:
        if col not in comparison_df.columns:
            continue
        if col not in expected_dtypes:
            continue
        exp_kind = dtype_kind(expected_dtypes[col])
        obs_kind = dtype_kind(comparison_df[col])
        if exp_kind != obs_kind:
            mismatches.append(
                {
                    "column": col,
                    "expected": str(expected_dtypes[col]),
                    "observed": str(comparison_df[col].dtype),
                    "expected_kind": exp_kind,
                    "observed_kind": obs_kind,
                }
            )

    t1 = "tier1_immediate"
    miss_cfg = _cfg(config, t1, "schema_missing_columns")
    extra_cfg = _cfg(config, t1, "schema_extra_columns")
    dtype_cfg = _cfg(config, t1, "schema_dtype_mismatch")

    return [
        make_check(
            "schema_missing_columns",
            tier=t1,
            status=_triggered_status(len(missing) > miss_cfg["threshold"], miss_cfg["severity"]),
            observed_value=len(missing),
            reference_value=0,
            threshold=miss_cfg["threshold"],
            severity=miss_cfg["severity"],
            detail=missing,
        ),
        make_check(
            "schema_extra_columns",
            tier=t1,
            status=_triggered_status(len(extra) > extra_cfg["threshold"], extra_cfg["severity"]),
            observed_value=len(extra),
            reference_value=0,
            threshold=extra_cfg["threshold"],
            severity=extra_cfg["severity"],
            detail=extra,
        ),
        make_check(
            "schema_dtype_mismatch",
            tier=t1,
            status=_triggered_status(
                len(mismatches) > dtype_cfg["threshold"], dtype_cfg["severity"]
            ),
            observed_value=len(mismatches),
            reference_value=0,
            threshold=dtype_cfg["threshold"],
            severity=dtype_cfg["severity"],
            detail=mismatches,
        ),
    ]


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

def null_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    n = len(df)
    for col in columns:
        if n == 0:
            rates[col] = 1.0
        elif col not in df.columns:
            rates[col] = 1.0
        else:
            rates[col] = float(df[col].isna().mean())
    return rates


def check_missingness(
    comparison_rates: dict[str, float],
    reference_rates: dict[str, float],
    config: dict,
) -> list[dict]:
    cfg = _cfg(config, "tier1_immediate", "missingness_rate_change")
    threshold = float(cfg["threshold"])
    flagged: list[dict[str, Any]] = []
    max_abs = 0.0
    for col, ref_rate in reference_rates.items():
        obs_rate = float(comparison_rates.get(col, 1.0))
        delta = abs(obs_rate - float(ref_rate))
        max_abs = max(max_abs, delta)
        if delta > threshold:
            flagged.append(
                {
                    "column": col,
                    "observed": obs_rate,
                    "reference": float(ref_rate),
                    "abs_change": delta,
                }
            )
    return [
        make_check(
            "missingness_rate_change",
            tier="tier1_immediate",
            status=_triggered_status(len(flagged) > 0, cfg["severity"]),
            observed_value=max_abs,
            reference_value=0.0,
            threshold=threshold,
            severity=cfg["severity"],
            note=cfg.get("note", ""),
            detail=flagged,
        )
    ]


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(
    expected_frac: np.ndarray | list[float],
    actual_frac: np.ndarray | list[float],
    eps: float = PSI_EPSILON,
) -> float:
    """Population Stability Index between two discrete distributions."""
    expected = np.asarray(expected_frac, dtype=float)
    actual = np.asarray(actual_frac, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError(
            f"PSI shape mismatch: expected {expected.shape} vs actual {actual.shape}"
        )
    if expected.size == 0:
        return 0.0
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def quantile_bin_edges(values: np.ndarray, n_bins: int = N_PSI_BINS) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 1.0]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        lo = float(finite.min())
        return [lo - 0.5, lo + 0.5]
    return [float(x) for x in edges]


def _bin_counts(values: np.ndarray, edges: list[float] | np.ndarray) -> np.ndarray:
    edge_arr = np.asarray(edges, dtype=float)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n_bins = max(len(edge_arr) - 1, 1)
    if finite.size == 0:
        return np.zeros(n_bins, dtype=float)
    idx = np.searchsorted(edge_arr, finite, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return np.bincount(idx.astype(int), minlength=n_bins).astype(float)


def psi_from_values(
    reference: np.ndarray,
    current: np.ndarray,
    bin_edges: list[float] | None = None,
    n_bins: int = N_PSI_BINS,
) -> tuple[float, list[float]]:
    edges = bin_edges if bin_edges is not None else quantile_bin_edges(reference, n_bins)
    ref_counts = _bin_counts(reference, edges)
    cur_counts = _bin_counts(current, edges)
    ref_frac = ref_counts / ref_counts.sum() if ref_counts.sum() else ref_counts
    cur_frac = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts
    return compute_psi(ref_frac, cur_frac), list(edges)


def check_psi_shift(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    config: dict,
    *,
    name: str = "feature_distribution_psi",
) -> dict:
    """PSI check from two raw arrays (used by unit tests and ad-hoc probes)."""
    cfg = _cfg(config, "tier1_immediate", "psi_threshold")
    threshold = float(cfg["threshold"])
    psi, _edges = psi_from_values(reference_values, comparison_values)
    return make_check(
        name,
        tier="tier1_immediate",
        status=_triggered_status(psi > threshold, cfg["severity"]),
        observed_value=psi,
        reference_value=0.0,
        threshold=threshold,
        severity=cfg["severity"],
        note=cfg.get("note", ""),
    )


def check_numeric_psi(
    comparison_df: pd.DataFrame,
    reference_numerics: dict[str, dict],
    feature_names: list[str],
    config: dict,
) -> list[dict]:
    cfg = _cfg(config, "tier1_immediate", "psi_threshold")
    threshold = float(cfg["threshold"])
    flagged: list[dict[str, Any]] = []
    max_psi = 0.0
    per_feature: dict[str, float] = {}
    for name in feature_names:
        stats = reference_numerics.get(name)
        if not stats or name not in comparison_df.columns:
            continue
        edges = stats.get("psi_bin_edges")
        values = pd.to_numeric(comparison_df[name], errors="coerce").to_numpy()
        # Reconstruct a reference sample is not possible; compare via stored
        # bin edges against a uniform-from-counts if present, else recompute
        # PSI using stored reference histogram fractions when available.
        ref_frac = stats.get("psi_bin_fractions")
        if edges and ref_frac:
            cur_counts = _bin_counts(values, edges)
            cur_frac = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts
            psi = compute_psi(np.asarray(ref_frac, dtype=float), cur_frac)
        else:
            # Fall back: no stored histogram — cannot compute a true PSI.
            psi = 0.0
        per_feature[name] = psi
        max_psi = max(max_psi, psi)
        if psi > threshold:
            flagged.append({"column": name, "psi": psi})
    return [
        make_check(
            "feature_distribution_psi",
            tier="tier1_immediate",
            status=_triggered_status(len(flagged) > 0, cfg["severity"]),
            observed_value=max_psi,
            reference_value=0.0,
            threshold=threshold,
            severity=cfg["severity"],
            note=cfg.get("note", ""),
            detail={"flagged": flagged, "per_feature": per_feature},
        )
    ]


def check_score_psi(
    comparison_scores: np.ndarray,
    reference_scores: dict,
    config: dict,
) -> dict:
    cfg = _cfg(config, "tier1_immediate", "score_distribution_psi")
    threshold = float(cfg["threshold"])
    edges = reference_scores.get("psi_bin_edges")
    ref_frac = reference_scores.get("psi_bin_fractions")
    if edges and ref_frac:
        cur_counts = _bin_counts(np.asarray(comparison_scores, dtype=float), edges)
        cur_frac = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts
        psi = compute_psi(np.asarray(ref_frac, dtype=float), cur_frac)
    else:
        psi = 0.0
    return make_check(
        "score_distribution_psi",
        tier="tier1_immediate",
        status=_triggered_status(psi > threshold, cfg["severity"]),
        observed_value=psi,
        reference_value=0.0,
        threshold=threshold,
        severity=cfg["severity"],
    )


def check_amount_psi(
    comparison_amounts: np.ndarray,
    reference_amounts: dict,
    config: dict,
) -> dict:
    cfg = _cfg(config, "tier1_immediate", "amount_distribution_psi")
    threshold = float(cfg["threshold"])
    edges = reference_amounts.get("psi_bin_edges")
    ref_frac = reference_amounts.get("psi_bin_fractions")
    if edges and ref_frac:
        cur_counts = _bin_counts(np.asarray(comparison_amounts, dtype=float), edges)
        cur_frac = cur_counts / cur_counts.sum() if cur_counts.sum() else cur_counts
        psi = compute_psi(np.asarray(ref_frac, dtype=float), cur_frac)
    else:
        psi = 0.0
    return make_check(
        "amount_distribution_psi",
        tier="tier1_immediate",
        status=_triggered_status(psi > threshold, cfg["severity"]),
        observed_value=psi,
        reference_value=0.0,
        threshold=threshold,
        severity=cfg["severity"],
    )


# ---------------------------------------------------------------------------
# Categoricals
# ---------------------------------------------------------------------------

def category_proportions(series: pd.Series, *, max_levels: int | None = None) -> dict[str, float]:
    n = len(series)
    if n == 0:
        return {}
    filled = series.where(series.notna(), MISSING_SENTINEL).astype(str)
    counts = filled.value_counts(normalize=False)
    if max_levels is not None and len(counts) > max_levels:
        top = counts.iloc[:max_levels]
        rest = int(counts.iloc[max_levels:].sum())
        counts = pd.concat([top, pd.Series({OTHER_SENTINEL: rest})])
    return {str(k): float(v) / n for k, v in counts.items()}


def check_categorical_drift(
    comparison_df: pd.DataFrame,
    reference_categoricals: dict[str, dict[str, float]],
    config: dict,
) -> list[dict]:
    cfg = _cfg(
        config,
        "tier1_immediate",
        "categorical_new_levels",
        default={"threshold": 0, "severity": "warning"},
    )
    new_levels: dict[str, list[str]] = {}
    n_new = 0
    for col, ref_props in reference_categoricals.items():
        if col not in comparison_df.columns:
            continue
        obs_props = category_proportions(comparison_df[col])
        known = set(ref_props) | {MISSING_SENTINEL, OTHER_SENTINEL}
        unseen = sorted(k for k in obs_props if k not in known)
        if unseen:
            new_levels[col] = unseen[:20]
            n_new += len(unseen)
    return [
        make_check(
            "categorical_new_levels",
            tier="tier1_immediate",
            status=_triggered_status(n_new > cfg["threshold"], cfg["severity"]),
            observed_value=n_new,
            reference_value=0,
            threshold=cfg["threshold"],
            severity=cfg["severity"],
            detail=new_levels,
        )
    ]


# ---------------------------------------------------------------------------
# Action bands
# ---------------------------------------------------------------------------

def action_proportions(actions: np.ndarray | pd.Series) -> dict[str, float]:
    arr = np.asarray(actions, dtype=object)
    n = len(arr)
    if n == 0:
        return {name: 0.0 for name in ACTION_NAMES}
    return {name: float(np.mean(arr == name)) for name in ACTION_NAMES}


def relative_change(observed: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if observed == 0 else float("inf")
    return abs(observed - reference) / abs(reference)


def check_action_band_volumes(
    comparison_props: dict[str, float],
    reference_props: dict[str, float],
    config: dict,
) -> list[dict]:
    cfg = _cfg(config, "tier1_immediate", "action_band_volume_change")
    threshold = float(cfg["threshold"])
    flagged: list[dict[str, Any]] = []
    max_rel = 0.0
    finite_max = 0.0
    for name in ACTION_NAMES:
        obs = float(comparison_props.get(name, 0.0))
        ref = float(reference_props.get(name, 0.0))
        rel = relative_change(obs, ref)
        if np.isfinite(rel):
            finite_max = max(finite_max, rel)
            max_rel = max(max_rel, rel)
        else:
            max_rel = rel
        if rel > threshold:
            flagged.append(
                {
                    "action": name,
                    "observed": obs,
                    "reference": ref,
                    "relative_change": rel if np.isfinite(rel) else None,
                }
            )
    observed_for_json: Any = finite_max if np.isfinite(max_rel) else None
    return [
        make_check(
            "action_band_volume_change",
            tier="tier1_immediate",
            status=_triggered_status(len(flagged) > 0, cfg["severity"]),
            observed_value=observed_for_json,
            reference_value=0.0,
            threshold=threshold,
            severity=cfg["severity"],
            note=cfg.get("note", ""),
            detail={"flagged": flagged, "observed_proportions": dict(comparison_props)},
        )
    ]


# ---------------------------------------------------------------------------
# Numeric summaries (reference profile)
# ---------------------------------------------------------------------------

def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "n_non_null": 0,
            "psi_bin_edges": [0.0, 1.0],
            "psi_bin_fractions": [1.0],
        }
    edges = quantile_bin_edges(finite)
    counts = _bin_counts(finite, edges)
    fractions = (counts / counts.sum()).tolist() if counts.sum() else [1.0]
    pct = {f"p{p}": float(np.percentile(finite, p)) for p in PERCENTILES}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        **pct,
        "n_non_null": int(finite.size),
        "psi_bin_edges": edges,
        "psi_bin_fractions": fractions,
    }


def operating_threshold_at_flag_rate(scores: np.ndarray, flag_rate: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n_flag = max(int(round(len(scores) * flag_rate)), 1)
    n_flag = min(n_flag, len(scores))
    return float(np.partition(scores, len(scores) - n_flag)[len(scores) - n_flag])


def _catch_rates(actions: np.ndarray, cost_params: dict[str, float]) -> np.ndarray:
    rates = np.zeros(len(actions), dtype=float)
    for action, spec in CATCH_RATE_BY_ACTION.items():
        mask = actions == action
        rates[mask] = float(cost_params[spec]) if isinstance(spec, str) else float(spec)
    return rates


def label_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
    amounts: np.ndarray,
    cost_params: dict[str, float],
    *,
    operating_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    actions = np.asarray(actions, dtype=object)
    amounts = np.asarray(amounts, dtype=float)
    y_true = (labels >= 0.5).astype(int)
    y_pred = (scores >= operating_threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, scores)) if y_true.size else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    neg = y_true == 0
    fpr = float(np.mean(y_pred[neg] == 1)) if neg.any() else 0.0
    catch = _catch_rates(actions, cost_params)
    is_fraud = y_true == 1
    fraud_dollars = amounts * is_fraud
    captured = float(np.sum(is_fraud * catch * amounts))
    total_fraud_dollars = float(np.sum(fraud_dollars))
    capture_rate = (captured / total_fraud_dollars) if total_fraud_dollars else 0.0
    by_action = {
        name: float(np.sum(fraud_dollars[actions == name])) for name in ACTION_NAMES
    }
    friction = actions != "approve"
    customer_friction_rate = float(np.mean(friction[neg])) if neg.any() else 0.0
    return {
        "fraud_rate": float(np.mean(y_true)) if y_true.size else 0.0,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "customer_friction_rate": customer_friction_rate,
        "operating_threshold": float(operating_threshold),
        "fraud_dollars_captured": captured,
        "fraud_dollars_total": total_fraud_dollars,
        "fraud_dollar_capture_rate": capture_rate,
        "fraud_dollars_by_action": by_action,
        "n_positive": int(y_true.sum()),
        "n_rows": int(y_true.size),
    }


def check_tier2(
    comparison_metrics: dict[str, Any],
    reference_labels: dict[str, Any],
    config: dict,
) -> list[dict]:
    t2 = "tier2_delayed"
    checks: list[dict] = []

    pr_cfg = _cfg(config, t2, "pr_auc_drop")
    ref_pr = float(reference_labels["pr_auc"])
    obs_pr = float(comparison_metrics["pr_auc"])
    pr_drop = ref_pr - obs_pr
    checks.append(
        make_check(
            "pr_auc_drop",
            tier=t2,
            status=_triggered_status(pr_drop >= float(pr_cfg["threshold"]), pr_cfg["severity"]),
            observed_value=obs_pr,
            reference_value=ref_pr,
            threshold=pr_cfg["threshold"],
            severity=pr_cfg["severity"],
            note=pr_cfg.get("note", ""),
            detail={"absolute_drop": pr_drop},
        )
    )

    prec_cfg = _cfg(config, t2, "precision_drop")
    ref_p = float(reference_labels["precision"])
    obs_p = float(comparison_metrics["precision"])
    checks.append(
        make_check(
            "precision_drop",
            tier=t2,
            status=_triggered_status(
                (ref_p - obs_p) >= float(prec_cfg["threshold"]), prec_cfg["severity"]
            ),
            observed_value=obs_p,
            reference_value=ref_p,
            threshold=prec_cfg["threshold"],
            severity=prec_cfg["severity"],
            detail={"absolute_drop": ref_p - obs_p},
        )
    )

    rec_cfg = _cfg(config, t2, "recall_drop")
    ref_r = float(reference_labels["recall"])
    obs_r = float(comparison_metrics["recall"])
    checks.append(
        make_check(
            "recall_drop",
            tier=t2,
            status=_triggered_status(
                (ref_r - obs_r) >= float(rec_cfg["threshold"]), rec_cfg["severity"]
            ),
            observed_value=obs_r,
            reference_value=ref_r,
            threshold=rec_cfg["threshold"],
            severity=rec_cfg["severity"],
            detail={"absolute_drop": ref_r - obs_r},
        )
    )

    val_cfg = _cfg(config, t2, "fraud_value_captured_drop")
    # Compare capture *rate* (caught / total fraud dollars) so windows of
    # different size — train 413k vs a daily or test batch — are comparable.
    ref_rate = float(reference_labels.get("fraud_dollar_capture_rate") or 0.0)
    obs_rate = float(comparison_metrics.get("fraud_dollar_capture_rate") or 0.0)
    rel_drop = ((ref_rate - obs_rate) / ref_rate) if ref_rate else 0.0
    checks.append(
        make_check(
            "fraud_value_captured_drop",
            tier=t2,
            status=_triggered_status(
                rel_drop >= float(val_cfg["threshold"]), val_cfg["severity"]
            ),
            observed_value=obs_rate,
            reference_value=ref_rate,
            threshold=val_cfg["threshold"],
            severity=val_cfg["severity"],
            note=val_cfg.get("note", ""),
            detail={
                "relative_drop": rel_drop,
                "observed_dollars_captured": comparison_metrics.get("fraud_dollars_captured"),
                "reference_dollars_captured": reference_labels.get("fraud_dollars_captured"),
                "observed_dollars_total": comparison_metrics.get("fraud_dollars_total"),
                "reference_dollars_total": reference_labels.get("fraud_dollars_total"),
                "observed_by_action": comparison_metrics.get("fraud_dollars_by_action"),
                "reference_by_action": reference_labels.get("fraud_dollars_by_action"),
            },
        )
    )

    fpr_cfg = _cfg(config, t2, "false_positive_rate_increase")
    ref_fpr = float(reference_labels["false_positive_rate"])
    obs_fpr = float(comparison_metrics["false_positive_rate"])
    checks.append(
        make_check(
            "false_positive_rate_increase",
            tier=t2,
            status=_triggered_status(
                (obs_fpr - ref_fpr) >= float(fpr_cfg["threshold"]), fpr_cfg["severity"]
            ),
            observed_value=obs_fpr,
            reference_value=ref_fpr,
            threshold=fpr_cfg["threshold"],
            severity=fpr_cfg["severity"],
            note=fpr_cfg.get("note", ""),
            detail={"absolute_increase": obs_fpr - ref_fpr},
        )
    )

    fr_cfg = _cfg(
        config,
        t2,
        "fraud_rate_shift",
        default={
            "threshold": 0.01,
            "severity": "warning",
            "note": "absolute change in base fraud rate",
        },
    )
    ref_fr = float(reference_labels["fraud_rate"])
    obs_fr = float(comparison_metrics["fraud_rate"])
    checks.append(
        make_check(
            "fraud_rate_shift",
            tier=t2,
            status=_triggered_status(
                abs(obs_fr - ref_fr) >= float(fr_cfg["threshold"]), fr_cfg["severity"]
            ),
            observed_value=obs_fr,
            reference_value=ref_fr,
            threshold=fr_cfg["threshold"],
            severity=fr_cfg["severity"],
            note=fr_cfg.get("note", "absolute change in base fraud rate"),
            detail={"absolute_change": obs_fr - ref_fr},
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def run_tier1_checks(
    *,
    reference: dict,
    comparison_df: pd.DataFrame,
    scores: np.ndarray,
    actions: np.ndarray,
    amounts: np.ndarray,
    config: dict,
    ignore_extra: set[str] | None = None,
    psi_features: list[str] | None = None,
) -> list[dict]:
    schema_checks = check_schema(
        comparison_df,
        reference["schema"],
        config,
        ignore_extra=ignore_extra,
    )
    feature_cols = list(reference["schema"]["columns"])
    # Missingness for model features + TransactionAmt (already in feature_order).
    miss_cols = list(dict.fromkeys(feature_cols + ["TransactionAmt"]))
    miss_checks = check_missingness(
        null_rates(comparison_df, miss_cols),
        reference["missingness"],
        config,
    )
    numeric_names = psi_features or list(reference.get("psi_numeric_features") or [])
    if not numeric_names:
        numeric_names = list(reference.get("numerics") or {})[:20]
    psi_checks = check_numeric_psi(
        comparison_df, reference.get("numerics") or {}, numeric_names, config
    )
    cat_checks = check_categorical_drift(
        comparison_df, reference.get("categoricals") or {}, config
    )
    score_check = check_score_psi(scores, reference.get("scores") or {}, config)
    amount_check = check_amount_psi(amounts, reference.get("amounts") or {}, config)
    action_checks = check_action_band_volumes(
        action_proportions(actions),
        reference.get("actions") or {},
        config,
    )
    return (
        schema_checks
        + miss_checks
        + psi_checks
        + cat_checks
        + [score_check, amount_check]
        + action_checks
    )


def build_report(
    *,
    reference: dict,
    checks: list[dict],
    comparison_n_rows: int,
    labels_present: bool,
    generated_at: str | None = None,
) -> dict[str, Any]:
    status = overall_status(checks)
    return {
        "overall_status": status,
        "checks": checks,
        "metadata": {
            "comparison_n_rows": int(comparison_n_rows),
            "labels_present": bool(labels_present),
            "generated_at": generated_at or utc_now_iso(),
            "reference": reference.get("metadata") or {},
            "status_rule": (
                "healthy = no warnings or criticals; "
                "warning = at least one warning and no criticals; "
                "critical = at least one critical"
            ),
        },
    }


def format_human_summary(report: dict) -> str:
    lines = [
        "=== Drift report ===",
        f"Overall status: {str(report['overall_status']).upper()}",
    ]
    meta = report.get("metadata") or {}
    ref_meta = meta.get("reference") or {}
    lines.append(
        f"Comparison n={meta.get('comparison_n_rows')}  "
        f"labels={'yes' if meta.get('labels_present') else 'no'}  "
        f"model={ref_meta.get('model_version')}  "
        f"dataset=v{ref_meta.get('dataset_version')}  "
        f"reference_split={ref_meta.get('split')}"
    )
    lines.append("")

    def _fmt(check: dict) -> str:
        status = str(check.get("status", "")).upper()
        name = check.get("name")
        obs = check.get("observed_value")
        ref = check.get("reference_value")
        thr = check.get("threshold")
        return (
            f"  [{status:<8}] {name:<32} "
            f"observed={obs}  reference={ref}  threshold={thr}"
        )

    tier1 = [c for c in report["checks"] if c.get("tier") == "tier1_immediate"]
    tier2 = [c for c in report["checks"] if c.get("tier") == "tier2_delayed"]
    lines.append("TIER 1 (immediate)")
    if tier1:
        lines.extend(_fmt(c) for c in tier1)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("TIER 2 (delayed / labels)")
    if tier2:
        lines.extend(_fmt(c) for c in tier2)
    else:
        lines.append("  (skipped — no labels)")
    fired = [
        c
        for c in report["checks"]
        if c.get("status") in {"warning", "critical"}
    ]
    lines.append("")
    if fired:
        lines.append("Fired signals:")
        for check in fired:
            lines.append(
                f"  - {check['name']}: {check['status']} "
                f"(observed={check['observed_value']}, "
                f"reference={check['reference_value']}, "
                f"threshold={check['threshold']})"
            )
    else:
        lines.append("Fired signals: none")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serving-stack scoring (registry-driven, §10)
# ---------------------------------------------------------------------------

def _ensure_import_paths() -> None:
    for directory in (str(API_DIR), str(EVALS_DIR), str(REPO_ROOT / "scripts")):
        if directory not in sys.path:
            sys.path.insert(0, directory)


def load_serving_artifacts() -> dict[str, Any]:
    """Load model, fitted transforms, calibrator, cost params via the API loader."""
    _ensure_import_paths()
    from model_loader import load_serving_artifacts as _load

    return _load()


def apply_monitoring_transforms(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Replay transforms using frozen fitted state — never .fit() (§2g / §10).

    Non-requires_fit steps are skipped when their outputs already exist so a
    processed dataset_v5 parquet is not re-derived (raw D columns were dropped).
    """
    _ensure_import_paths()
    from predictor import apply_transformation_pipeline

    selected: list[dict[str, Any]] = []
    for step in artifacts["transformation_pipeline"]:
        if step["requires_fit"]:
            selected.append(step)
            continue
        outputs = step["config"].get("output_columns") or []
        if not outputs or any(col not in df.columns for col in outputs):
            selected.append(step)
    if not selected:
        return df
    return apply_transformation_pipeline(df, selected)


def build_model_input_batch(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Batch equivalent of services/api/predictor.build_model_input."""
    feature_order: list[str] = artifacts["feature_order"]
    categorical_columns: list[str] = artifacts["categorical_columns"]
    pandas_categorical = artifacts["pandas_categorical"]
    df = df.reset_index(drop=True)
    n = len(df)
    present = set(df.columns)
    data: dict[str, Any] = {}
    for feat in feature_order:
        if feat in present:
            data[feat] = df[feat].to_numpy()
        else:
            data[feat] = np.full(n, np.nan)
    out = pd.DataFrame(data, columns=feature_order)
    cat_set = set(categorical_columns)
    for i, feat in enumerate(categorical_columns):
        cats = list(pandas_categorical[i])
        cat_lookup = set(cats)
        has_missing = MISSING_SENTINEL in cat_lookup
        series = out[feat]
        mask_na = series.isna()
        as_str = np.where(mask_na, None, series.astype(str))
        coded = []
        for value, is_na in zip(as_str, mask_na.to_numpy()):
            if is_na or value is None:
                coded.append(None)
            elif value == MISSING_SENTINEL or value in cat_lookup:
                coded.append(value if value in cat_lookup else None)
            else:
                coded.append(MISSING_SENTINEL if has_missing else None)
        out[feat] = pd.Categorical(coded, categories=cats)
    for feat in feature_order:
        if feat not in cat_set:
            out[feat] = pd.to_numeric(out[feat], errors="coerce")
    return out[feature_order]


def score_and_act(
    feature_df: pd.DataFrame,
    artifacts: dict[str, Any],
    amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict → calibrate → argmin action. Same path as the API, batched."""
    _ensure_import_paths()
    from cost_model import optimal_action

    raw = np.asarray(artifacts["model"].predict(feature_df), dtype=float)
    calibrated = np.asarray(artifacts["calibrator"].predict(raw), dtype=float)
    actions, _costs = optimal_action(calibrated, amounts, artifacts["cost_params"])
    actions = np.asarray(actions, dtype=object)
    return raw, calibrated, actions


def select_psi_numeric_features(
    artifacts: dict[str, Any],
    n: int = 20,
) -> list[str]:
    cat_set = set(artifacts.get("categorical_columns") or [])
    model = artifacts["model"]
    names = list(model.feature_name())
    gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    ranked = sorted(zip(names, gain), key=lambda item: -item[1])
    selected = [name for name, _gain in ranked if name not in cat_set][:n]
    if len(selected) < n:
        for name in artifacts["feature_order"]:
            if name not in cat_set and name not in selected:
                selected.append(name)
            if len(selected) >= n:
                break
    return selected
