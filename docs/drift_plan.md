# Drift Plan

## Purpose
Simulate and detect data drift between a reference snapshot and a current snapshot of the preprocessed (silver) dataset. This supports monitoring model robustness (especially recall on the `Fail` class) without changing labels.

We do not modify the target; only feature distributions are compared.

---

## Implementation

Detection is implemented in `src/drift_detection.py` using Evidently:

- **Inputs**

  - `reference_data_path`: CSV path to the reference dataset (e.g., `data/silver/preprocessed_ml2_student_performance.csv`)

  - `current_data_path`: CSV path to the current/drifted dataset (e.g., `data/silver/drifted_test.csv`)


- **Target handling**

  - If a column named `target` exists, it is **dropped** from both reference and current before drift analysis:
    ```python
    if "target" in reference_data.columns:
        reference_features = reference_data.drop(columns=["target"])
        current_features = current_data.drop(columns=["target"])
    else:
        reference_features = reference_data
        current_features = current_data
    ```

(In our silver dataset, the label column is named `Pass/Fail`, so nothing is dropped.)

### Drift Simulation
Drifted CSVs are created using controlled changes to features only:

1. **Numerical** (`Age`, `Hours of Sleep`, `Hours Reviewing`): scaling ×1.2 and/or Gaussian noise (σ = 10% of feature std).  

2. **Categorical** (`Course`, `Relationship status`): random 10–15% category flips among valid values.  

3. **Labels** (`Pass/Fail`): unchanged.  


### Detector
Uses `Report(metrics=[DataDriftPreset()])` to compute dataset-level and feature-level drift.

### Outputs
A JSON summary is written to `reports/drift_report.json` with:

1. `drift_detected`: `true` if any feature drift is detected (i.e., `drifted_columns > 0`).  

2. `feature_drifts`: per-feature drift scores (Evidently’s `ValueDrift` metric values).  

3. `overall_drift_score`: dataset-level share of drifted features reported by Evidently.  


**Example structure:**

```json
{
  "drift_detected": true,
  "feature_drifts": {
    "Age": 0.32,
    "Hours of Sleep": 0.18,
    "Hours Reviewing": 0.27
  },
  "overall_drift_score": 0.40
}
```

### Running the Detector
From the project root, run:

```
python src/drift_detection.py
```

### Artifacts and Logging

A JSON summary is written to `reports/drift_report.json`.