"""Print the complete lgbm_v5 lower-learning-rate experiment report.

Usage:
    python evals/print_v5_stats.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
V4_AUC_PR = 0.6309
TREE_CEILING = 15000


def load_json(path: Path) -> dict | list:
    """Load a JSON artifact."""
    return json.loads(path.read_text())


def registry_entry(registry: list[dict], version: int) -> dict:
    """Return exactly one registry entry for a model version."""
    matches = [entry for entry in registry if entry["version"] == version]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry entry for v{version}, found {len(matches)}")
    return matches[0]


def parse_bootstrap(baseline_version: int) -> dict[str, float | bool]:
    """Parse the AUC-PR section of a saved bootstrap run."""
    path = REPO_ROOT / "evals" / f"bootstrap_v{baseline_version}_vs_v5.txt"
    text = path.read_text()
    difference = re.search(r"Mean difference:\s+([+-]\d+\.\d+)", text)
    interval = re.search(
        r"95% CI for difference:\s+\[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]",
        text,
    )
    if not difference or not interval:
        raise ValueError(f"Could not parse AUC-PR bootstrap output: {path}")
    ci_low, ci_high = float(interval.group(1)), float(interval.group(2))
    return {
        "difference": float(difference.group(1)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": ci_low > 0 or ci_high < 0,
    }


def print_performance_table(registry: list[dict]) -> None:
    """Print metrics for every registered model version."""
    print(
        f"{'Version':<9}{'Dataset':<9}{'Description':<35}{'AUC-PR':>8}"
        f"{'AUC-ROC':>10}{'Recall':>9}{'Prec':>9}{'F1':>9}"
        f"{'Thresh':>10}{'Iters':>8}"
    )
    for entry in sorted(registry, key=lambda item: item["version"]):
        metrics = load_json(REPO_ROOT / entry["metrics_path"])["metrics"]
        description = entry["description"][:33]
        print(
            f"v{entry['version']:<8}v{entry['dataset_version']:<8}{description:<35}"
            f"{metrics['auc_pr']:>8.4f}{metrics['auc_roc']:>10.4f}"
            f"{metrics['recall_at_budget']:>9.4f}"
            f"{metrics['precision_at_budget']:>9.4f}"
            f"{metrics['f1_at_budget']:>9.4f}"
            f"{metrics['operating_threshold']:>10.4f}"
            f"{metrics['best_iteration']:>8}"
        )


def print_shap_section(v5_auc_pr: float) -> None:
    """Print conditional v5 SHAP attribution with v4 rank deltas."""
    print("SHAP ATTRIBUTION (Top 20) — if AUC-PR improved")
    print("───────────────────────────────────────────────")
    if v5_auc_pr <= V4_AUC_PR:
        print("Skipped — no AUC-PR improvement")
        return

    v5_path = (
        REPO_ROOT / "evals" / "shap" / "lgbm_v5_lower_learning_rate_shap.json"
    )
    if not v5_path.exists():
        raise FileNotFoundError(f"v5 improved but SHAP artifact is missing: {v5_path}")
    v4_path = (
        REPO_ROOT / "evals" / "shap" / "lgbm_v4_tuned_hyperparams_shap.json"
    )
    v5_shap = load_json(v5_path)
    v4_shap = load_json(v4_path)
    v4_ranks = {
        item["feature"]: item["rank"] for item in v4_shap["top_features"]
    }
    print(
        f"{'Rank':<6}{'Feature':<24}{'Mean |SHAP|':>13}"
        f"{'v4 Rank':>11}{'Δ Rank':>10}"
    )
    for item in v5_shap["top_features"]:
        old_rank = v4_ranks.get(item["feature"])
        old_text = str(old_rank) if old_rank is not None else "—"
        delta = f"{old_rank - item['rank']:+d}" if old_rank is not None else "—"
        print(
            f"{item['rank']:<6}{item['feature']:<24}"
            f"{item['mean_abs_shap']:>13.6f}{old_text:>11}{delta:>10}"
        )


def main() -> None:
    """Load versioned artifacts and print the requested structured report."""
    registry = load_json(REGISTRY_PATH)
    v5_entry = registry_entry(registry, 5)
    v5_metrics = load_json(REPO_ROOT / v5_entry["metrics_path"])
    metrics = v5_metrics["metrics"]
    best_iteration = int(metrics["best_iteration"])
    headroom = TREE_CEILING - best_iteration
    converged = best_iteration <= TREE_CEILING - 100
    bootstrap = {version: parse_bootstrap(version) for version in range(1, 5)}
    v4_result = bootstrap[4]

    if v4_result["ci_low"] > 0:
        decision = "FLIP"
        rationale = "The paired 95% bootstrap CI for v5 − v4 clears zero."
    elif v4_result["ci_high"] < 0:
        decision = "DO NOT FLIP"
        rationale = "The paired 95% bootstrap CI shows a significant regression."
    else:
        decision = "INCONCLUSIVE"
        rationale = "The paired 95% bootstrap CI for v5 − v4 includes zero."

    line = "═" * 64
    print(line)
    print("         lgbm_v5 — Lower Learning Rate / Capacity Fix")
    print(line)
    print()
    print("HEADER")
    print("──────")
    print("Branch:           v5-lower-learning-rate")
    print("Dataset:          v2 (unchanged from v4)")
    print("Parent model:     v4 (AUC-PR 0.6309, best_iteration 2999/3000)")
    print("Hypothesis:       Lower learning rate (0.01) + raised ceiling (15000)")
    print("                  resolves v4's tree capacity saturation")
    print()
    print("METHODOLOGY")
    print("───────────")
    print("• Same dataset_v2, same 461 features")
    print("• Changed: learning_rate 0.05 → 0.01, n_estimators 3000 → 15000")
    print("• All other hyperparameters identical to v4")
    print("• Temporal 80/20 split on TransactionDT")
    print()
    print("CAPACITY DIAGNOSTIC")
    print("───────────────────")
    print("v4 best_iteration: 2999 / 3000 (saturated)")
    print(
        f"v5 best_iteration: {best_iteration} / {TREE_CEILING} "
        f"(headroom: {headroom} trees unused)"
    )
    print(f"Converged naturally: {'Yes' if converged else 'No'}")
    print()
    print("PERFORMANCE COMPARISON (v1 → v5)")
    print("────────────────────────────────")
    print_performance_table(registry)
    print()
    print("BOOTSTRAP SIGNIFICANCE (500 iterations)")
    print("───────────────────────────────────────")
    for version, result in bootstrap.items():
        significant = "Yes" if result["significant"] else "No"
        print(
            f"v5 vs v{version}:  Δ AUC-PR = {result['difference']:+.4f} "
            f"[95% CI: ({result['ci_low']:+.4f}, {result['ci_high']:+.4f})] "
            f"— Significant: {significant}"
        )
    print()
    print("⚠️  5 models against same test set. Bonferroni α = 0.0125 for 4 comparisons.")
    print()
    print_shap_section(float(metrics["auc_pr"]))
    print()
    print("SERVING DECISION")
    print("────────────────")
    print(f"Decision:  {decision}")
    print(f"Rationale: {rationale}")
    print()
    print("FLAGS")
    print("─────")
    print(f"• best_iteration: {best_iteration} / {TREE_CEILING}")
    print("• Multiple comparisons: 5th model against same holdout")
    wall_minutes = v5_metrics["training"]["wall_time_seconds"] / 60
    print(f"• Training time: ~{wall_minutes:.1f}m (expected ~5× v4)")
    print()
    print("NEXT STEPS")
    print("──────────")
    if converged:
        print("• Capacity was the problem: retry velocity features on the v5 model")
        print("  (v6 = dataset_v3 velocity features + v5 hyperparameters)")
    else:
        print("• Capacity was not resolved: investigate other bottlenecks")
    print()
    print("ARTIFACTS")
    print("─────────")
    artifacts = [
        ("configs/lgbm_v5.json", "committed"),
        ("models/lgbm_v5_lower_learning_rate.txt", "gitignored"),
        ("models/lgbm_v5_lower_learning_rate_metrics.json", "committed"),
        ("models/lgbm_v5_lower_learning_rate_manifest.json", "committed"),
        ("models/lgbm_v5_lower_learning_rate_fitted_transforms.pkl", "gitignored"),
        ("models/registry.json", "committed"),
    ]
    shap_path = "evals/shap/lgbm_v5_lower_learning_rate_shap.json"
    shap_status = "committed" if float(metrics["auc_pr"]) > V4_AUC_PR else "not run"
    artifacts.append((shap_path, shap_status))
    print(f"{'File':<67}{'Status':<12}Exists?")
    for relative, status in artifacts:
        exists = "✓" if (REPO_ROOT / relative).exists() else "✗"
        print(f"{relative:<67}{status:<12}{exists}")
    print(line)


if __name__ == "__main__":
    main()
