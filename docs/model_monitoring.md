# Model monitoring — synthetic batch report

This is a local batch over the 24 invented payloads in `tests/load/payloads.json`. It is not a labeled evaluation.

## How to run

Port-forward the Service, then score the file. The second command joins the invented labels. Both writes stay outside Git, under `~/fraud-evidence/`.

```bash
kubectl -n fraud-local port-forward svc/fraud-api 8000:8000
python3 scripts/prediction_batch_report.py
python3 scripts/prediction_batch_report.py --labels tests/fixtures/simulated_labels.json --out-dir ~/fraud-evidence/batch-simulated
```

The first command uses the default output directory `~/fraud-evidence/batch`. Each run writes `predictions.jsonl` (example id, the returned `fraud_probability`, `recommended_action`, and `request_id`) and `report.json`. The probability is the value the API returned. It is not replaced by the action. The script does not write payloads or scores to application logs.

## Results

Both runs scored all 24 rows. Invalid scores: 0. The score distribution is the same in both reports because the labels are not an input to `/predict`. Exact values are in [docs/evidence/phase10b/batch/report.json](evidence/phase10b/batch/report.json) and [docs/evidence/phase10b/batch/report_simulated.json](evidence/phase10b/batch/report_simulated.json).

| Quantile | Score |
| --- | ---: |
| min | 0.009896 |
| p10 | 0.019066 |
| p25 | 0.098751 |
| p50 | 0.234146 |
| p75 | 0.351852 |
| p90 | 0.439675 |
| max | 0.516779 |

Action counts: approve 3, step_up 0, review 21, decline 0, other 0.

Without `--labels`, `label_status` is `unavailable`, coverage is null, and performance is null. With the simulated file, `label_status` is `simulated`, 10 of 24 rows match (`label_coverage` 10/24), and performance is `not computed (simulated labels, n too small)`.

## Label assumptions

There are no real labels. `tests/fixtures/simulated_labels.json` holds invented 0/1 values for 10 of the 24 `example_id`s. The join is on that stable synthetic id. An id with no label is left unmatched. It is not treated as 0.

## Limits

These payloads are synthetic and out of distribution. Most of them land on `review`, so the action mix says nothing about real traffic. A drift alert or an action-mix change on its own does not prove that accuracy has fallen. Accuracy needs real delayed labels. That is the Phase 7 Tier 2 plan in [docs/monitoring_plan.md](monitoring_plan.md): chargebacks typically arrive 30–90 days later, and precision, recall, and fraud value captured are computed only on a lag-mature window.
