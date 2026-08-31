"""Register a model version as a Candidate in the registry.

Validates all artifact paths exist before writing. Never silently replaces
the registry — updates the existing entry for the specified version or
appends a new one.

Usage:
    python scripts/register_model.py --version 9 --stage Candidate
    python scripts/register_model.py --version 9 --stage Production --promoted-by "Deepa"

§6 compliance: registry is the single source of truth.
§12 compliance: registry.json is committed to git.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_git_commit():
    """Get current git commit SHA."""
    try:
        full = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT
        ).decode().strip()
        short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_ROOT
        ).decode().strip()
        return full, short
    except subprocess.CalledProcessError:
        return None, None

def validate_path(path_str, field_name, errors):
    """Check that a file exists. Append to errors if not."""
    full_path = PROJECT_ROOT / path_str
    if not full_path.exists():
        errors.append(f"  {field_name}: {path_str} → NOT FOUND")
    else:
        return True
    return False

def load_registry():
    """Load the existing registry."""
    registry_path = PROJECT_ROOT / 'models' / 'registry.json'
    if registry_path.exists():
        return json.load(open(registry_path))
    return []

def save_registry(registry):
    """Save registry, sorted by version."""
    registry_path = PROJECT_ROOT / 'models' / 'registry.json'
    registry = sorted(registry, key=lambda e: e['version'])
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"Saved registry ({len(registry)} entries) to {registry_path}")

def main():
    parser = argparse.ArgumentParser(description='Register a model as Candidate')
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--stage', type=str, default='Candidate',
                        choices=['Experimental', 'Candidate', 'Production'])
    parser.add_argument('--promoted-by', type=str, default=None,
                        help='Name of the person promoting to Production')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and print without writing')
    args = parser.parse_args()

    v = args.version
    errors = []
    warnings = []

    print(f"Registering v{v} as {args.stage}...")
    print()

    # --- Load existing registry ---
    registry = load_registry()
    existing = [e for e in registry if e['version'] == v]

    if existing:
        entry = existing[0]
        print(f"Existing entry found for v{v}. Will update lifecycle_stage.")
    else:
        entry = {'version': v}
        print(f"No existing entry for v{v}. Will create new entry.")

    # --- Resolve artifact paths ---
    import glob

    model_files = glob.glob(str(PROJECT_ROOT / f'models/lgbm_v{v}_*.txt'))
    metrics_files = glob.glob(str(PROJECT_ROOT / f'models/lgbm_v{v}_*_metrics.json'))
    manifest_files = glob.glob(str(PROJECT_ROOT / f'models/lgbm_v{v}_*_manifest.json'))

    if not model_files:
        errors.append(f"  model_path: no models/lgbm_v{v}_*.txt found")
    if not metrics_files:
        errors.append(f"  metrics_path: no models/lgbm_v{v}_*_metrics.json found")
    if not manifest_files:
        errors.append(f"  manifest_path: no models/lgbm_v{v}_*_manifest.json found")

    # Use paths relative to project root
    model_path = str(Path(model_files[0]).relative_to(PROJECT_ROOT)) if model_files else None
    metrics_path = str(Path(metrics_files[0]).relative_to(PROJECT_ROOT)) if metrics_files else None
    manifest_path = str(Path(manifest_files[0]).relative_to(PROJECT_ROOT)) if manifest_files else None

    config_path = f'configs/lgbm_v{v}.json'
    validate_path(config_path, 'config_path', errors)

    # Load config to find dataset version and config path
    dataset_version = None
    dataset_config_path = None
    if (PROJECT_ROOT / config_path).exists():
        config = json.load(open(PROJECT_ROOT / config_path))
        dataset_version = config.get('dataset_version')
        dataset_config_path = config.get('dataset_config_path')
        if dataset_config_path:
            validate_path(dataset_config_path, 'dataset_config_path', errors)

    # Decision policy (optional — only for Phase 3+)
    decision_policy_path = None
    for candidate in [f'configs/decision_policy_v1.json']:
        if (PROJECT_ROOT / candidate).exists():
            policy = json.load(open(PROJECT_ROOT / candidate))
            if policy.get('model_version') == v:
                decision_policy_path = candidate
                break
    if not decision_policy_path:
        warnings.append("  No decision policy config found for this version")

    # Calibrator (optional — only for Phase 3+)
    calibrator_path = f'models/lgbm_v{v}_calibrator.pkl'
    if not (PROJECT_ROOT / calibrator_path).exists():
        calibrator_path = None
        warnings.append("  No calibrator found — decision policy may not work")

    # --- Load metrics for test_metrics block ---
    test_metrics = {}
    if metrics_path and (PROJECT_ROOT / metrics_path).exists():
        metrics = json.load(open(PROJECT_ROOT / metrics_path))
        m = metrics.get('metrics', metrics)
        training = metrics.get('training', {})
        test_metrics = {
            'test_auc_pr': m.get('test_auc_pr', m.get('auc_pr')),
            'val_auc_pr': m.get('val_auc_pr'),
            'best_iteration': m.get('best_iteration'),
            'n_features': training.get('n_features'),
        }
        created_at = metrics.get('timestamp', metrics.get('created_at'))
    else:
        created_at = None

    # --- Evaluation evidence ---
    evidence = {}
    evidence_files = {
        'model_lock_report': 'docs/model_lock_report.md',
        'final_evaluation': 'docs/final_evaluation.md',
        'model_card': 'docs/model_card.md',
        'decision_policy': 'docs/decision_policy.md',
        'shap_analysis': f'evals/figures/shap_top20_v{v}.json',
        'ablation_results': f'evals/figures/ablation_results_v{v}.json',
        'calibration_before': f'evals/figures/calibration_before_v{v}.png',
        'calibration_after': f'evals/figures/calibration_after_v{v}.png',
        'threshold_policy_results': f'evals/figures/threshold_policy_results_v{v}.json',
        'sensitivity_analysis': f'evals/figures/sensitivity_analysis_v{v}.json',
        'comparison_table': 'evals/model_comparison_results.txt',
    }
    for key, path in evidence_files.items():
        if (PROJECT_ROOT / path).exists():
            evidence[key] = path
        else:
            warnings.append(f"  Evidence missing: {key} → {path}")

    # --- Git commit ---
    git_full, git_short = get_git_commit()
    if not git_full:
        errors.append("  Cannot determine git commit")

    # Check for uncommitted changes
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=PROJECT_ROOT
        ).decode().strip()
        if status:
            warnings.append("  Uncommitted changes detected — commit hash may not fully represent the state")
    except subprocess.CalledProcessError:
        pass

    # --- Build the entry ---
    now = datetime.now(timezone.utc).isoformat()

    entry.update({
        'version': v,
        'dataset_version': dataset_version,
        'description': entry.get('description', ''),
        'lifecycle_stage': args.stage,
        'model_path': model_path,
        'manifest_path': manifest_path,
        'metrics_path': metrics_path,
        'config_path': config_path,
        'dataset_config_path': dataset_config_path,
    })

    if decision_policy_path:
        entry['decision_policy_path'] = decision_policy_path
    if calibrator_path:
        entry['calibrator_path'] = calibrator_path

    entry.update({
        'git_commit': git_full,
        'git_commit_short': git_short,
        'created_at': created_at or entry.get('created_at'),
        'registered_at': now,
        'evaluation_evidence': evidence,
        'test_metrics': test_metrics,
        'promotion': {
            'promoted_by': args.promoted_by,
            'promoted_at': now if args.stage == 'Production' else None,
            'promotion_checklist': 'docs/model_promotion_checklist.md',
        },
        'is_serving': entry.get('is_serving', False),
    })

    # --- Report ---
    print("VALIDATION RESULTS")
    print("=" * 50)

    if errors:
        print(f"\nERRORS ({len(errors)}) — must fix before registering:")
        for e in errors:
            print(e)
    else:
        print("\n  No errors ✓")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}) — non-blocking:")
        for w in warnings:
            print(w)
    else:
        print("\n  No warnings ✓")

    print(f"\nProposed entry:")
    print(json.dumps(entry, indent=2))

    if errors:
        print(f"\n✗ REGISTRATION BLOCKED — {len(errors)} error(s) must be resolved")
        sys.exit(1)

    if args.dry_run:
        print("\n  Dry run — no changes written.")
        return

    # --- Write ---
    if existing:
        for i, e in enumerate(registry):
            if e['version'] == v:
                registry[i] = entry
                break
    else:
        registry.append(entry)

    save_registry(registry)
    print(f"\n✓ v{v} registered as {args.stage}")

if __name__ == '__main__':
    main()
