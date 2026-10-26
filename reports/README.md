# Aura Farming — reports/

This directory stores experiment outputs generated during Phases 4–7.
Most generated files are excluded from Git via `.gitignore`.

## Subdirectories

| Path | Contents |
|---|---|
| `reports/figures/` | Confusion matrices, calibration curves, coverage plots (PNG) |
| `reports/metrics/` | Metric CSVs from each experiment |

## Files generated here

- `baseline_a_metrics.csv` — Baseline A (no quality gate or calibration)
- `full_pipeline_metrics.csv` — Full reliability-aware pipeline
- `calibration_ece.json` — ECE before and after calibration
- `coverage_vs_accuracy.csv` — Threshold sweep results
- `confusion_matrix_baseline.png`
- `confusion_matrix_pipeline.png`
- `calibration_reliability_diagram.png`
- `gradcam_examples/` — Selected Grad-CAM overlay images

## Reporting rule

**Only include results from experiments that have actually been run.**
Do not add fabricated metric values, charts, or accuracy claims.
Every file in this directory must trace back to a real experiment.
