# Decision Policy for lgbm_v9

Per-transaction expected-cost minimization on calibrated fraud probabilities.
Status: **proposed**. Serving model: `lgbm_v9` (`is_serving: true`). Evaluation
uses the **validation split only**.

Config: `configs/decision_policy_v1.json` (`policy_type: per_transaction_argmin`).

## Calibration

Raw LightGBM scores for v9 are inflated by `scale_pos_weight ≈ 32.2`, so mean
raw score is far above the empirical fraud rate. Before any cost model is
applied, scores are mapped to calibrated probabilities with
`sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")`.

Procedure (`evals/calibrate.py`):

1. Load dataset_v5, apply the three-way temporal split from `configs/lgbm_v9.json`
   (70 / 15 / 15), fit `requires_fit` transforms on train via
   `apply_requires_fit_transforms`, and score **validation** with the frozen
   booster.
2. Fit 5-fold out-of-fold isotonic models on validation
   (`KFold(n_splits=5, shuffle=True, random_state=42)`) so no row is calibrated
   by a model that saw its own label. OOF probabilities are used for the printed
   calibration gap.
3. Refit a single deployment calibrator on the full validation split and save
   it to `models/lgbm_v9_calibrator.pkl`.

Summary metrics printed by the script: raw score mean, calibrated score mean,
actual fraud rate, calibration gap (`calibrated_mean − fraud_rate`).

## Cost model

Given calibrated probability \(p\), amount \(a\), and cost parameters:

| Action | Expected cost |
|--------|----------------|
| approve | \(p \cdot a \cdot F\) |
| step_up | \(p \cdot a \cdot F \cdot (1-s) + (1-p) \cdot a \cdot f\) |
| review | \(p \cdot a \cdot F \cdot (1-c) + R\) |
| decline | \((1-p) \cdot a \cdot d\) |

where \(F\) = `fraud_cost_multiplier`, \(s\) = `step_up_stop_rate`,
\(f\) = `step_up_friction_rate`, \(c\) = `review_catch_rate`,
\(R\) = `review_cost_per_txn`, \(d\) = `decline_revenue_loss`.

Implementation: `evals/cost_model.py` (`compute_all_costs`, `optimal_action`).
All functions accept scalars or NumPy arrays.

Default assumptions (`configs/decision_policy_v1.json`):

| Parameter | Default | Status |
|-----------|---------|--------|
| `fraud_cost_multiplier` | 2.0 | Unconfirmed — chargeback / ops multiple |
| `step_up_stop_rate` | 0.70 | Unconfirmed — fraction of fraud stopped by step-up |
| `step_up_friction_rate` | 0.05 | Unconfirmed — legit abandonment under step-up |
| `review_catch_rate` | 0.95 | Unconfirmed — analyst catch rate |
| `review_cost_per_txn` | 10.0 | Unconfirmed — loaded cost per review |
| `review_capacity_per_day` | 200 | Unconfirmed — ops capacity (used by threshold search + reporting) |
| `decline_revenue_loss` | 1.0 | Unconfirmed — lost margin multiple on false declines |

None of these are measured on IEEE-CIS; treat them as editable business knobs.

## Analytical decision boundaries

Set pairs of costs equal and solve in \((p, a)\) space (with defaults
\(F=2, s=0.7, f=0.05, c=0.95, R=10, d=1\)):

**Approve vs step-up** (independent of \(a\) — a vertical line in \(p\)):

\[
p^\* = \frac{f}{F s + f} = \frac{0.05}{2\cdot0.7 + 0.05} \approx 0.0345
\]

**Approve vs decline** (also vertical):

\[
p^\* = \frac{d}{F + d} = \frac{1}{3} \approx 0.333
\]

**Approve vs review** (hyperbola):

\[
p^\* = \frac{R}{a F c} = \frac{10}{a \cdot 2 \cdot 0.95} = \frac{10}{1.9 a}
\]

**Step-up vs decline**, **step-up vs review**, **review vs decline** yield
additional curves; the argmin policy evaluates all four costs per row and takes
the minimum, so the realized regions are the lower envelope of these surfaces.

Intuition: review has a fixed \$10 fee, so it is never optimal for tiny amounts
(at \(a=5\), \(p=0.5\), review cost exceeds the other options). For large
amounts the residual fraud term dominates and review / decline become
attractive relative to blind approval.

## Fixed-threshold baseline

`evals/threshold_policy.py` ignores amount at decision time and assigns:

- \(p < t_1\) → approve
- \(t_1 \le p < t_2\) → step_up
- \(t_2 \le p < t_3\) → review
- \(p \ge t_3\) → decline

Thresholds are chosen by two-pass grid search on validation, minimizing

\[
\sum_i \mathrm{cost}(a_i, p_i, \mathrm{action}_i) + 10^6 \cdot \max(0, \mathrm{reviews/day} - \mathrm{capacity})
\]

- Coarse pass: step \(0.01\) over \(0 \le t_1 \le t_2 \le t_3 \le 1\)
- Fine pass: step \(0.005\) within \(\pm 0.05\) of the coarse optimum

The capacity penalty forces the threshold policy to respect
`review_capacity_per_day`; the argmin policy does **not** enforce a global cap
(see below).

## Why argmin dominates

For any fixed \((p, a)\), argmin selects \(\arg\min_k \mathrm{cost}_k(p, a)\).
A threshold rule is a special case that assigns the same action to all rows
with the same \(p\), ignoring \(a\). Therefore, on any finite set of rows, the
sum of chosen expected costs under argmin is ≤ the sum under any threshold
triplet. If this invariant fails in code, it is a bug — not an empirical
finding.

Empirically, disagreements concentrate in low-amount and high-amount quartiles:
thresholds cannot approve cheap low-\(p\) edge cases selectively while
reviewing expensive mid-\(p\) transactions with the same score.

## Capacity

Argmin has no global review budget. In production, queue pressure should be
handled by ranking review candidates on `cost_gap` (how much worse the
second-best action is) and filling the daily review slots from the top of that
list — converting unconstrained argmin into a capacity-feasible policy without
collapsing back to amount-blind thresholds.

The threshold baseline *does* bake capacity into the search objective via the
\(10^6\) excess-review penalty so the comparison is operationally meaningful.

## Sensitivity

`evals/sensitivity_analysis.py` sweeps each of the six tunable cost parameters
one-at-a-time (5–8 values), holding others at defaults, and reruns argmin on
validation.

Observed leverage on the v9 validation split (defaults as in
`decision_policy_v1.json`):

| Parameter | Cost range | Review-count range |
|-----------|------------|--------------------|
| `fraud_cost_multiplier` | largest (~196k) | high |
| `review_cost_per_txn` | large (~179k) | **largest** (~24k) |
| `review_catch_rate` | large (~119k) | moderate |
| `step_up_friction_rate` | moderate | moderate |
| `decline_revenue_loss` | smaller | small |
| `step_up_stop_rate` | smaller | moderate |

Most leverage on total expected cost: `fraud_cost_multiplier`.
Most leverage on review volume: `review_cost_per_txn`.
Re-run the script after any business update to the defaults.

## Evaluation entrypoints

| Script | Role |
|--------|------|
| `evals/calibrate.py` | Fit / save calibrator; print calibration gap |
| `evals/compare_policies.py` | Side-by-side argmin vs thresholds + bootstrap CI |
| `evals/sensitivity_analysis.py` | Parameter sweeps |
| `tests/test_decision_policy.py` | Analytical unit tests |

Holdout **test** split is intentionally unused in Phase 3.
