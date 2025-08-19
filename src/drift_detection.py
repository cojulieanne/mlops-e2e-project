import pandas as pd
import numpy as np
import mlflow
import json
import os
from scipy.stats import ks_2samp, chi2_contingency


def calculate_drift(original, drifted, numerical_cols, categorical_cols):
    drift_results = {}

    for col in numerical_cols:
        stat, pval = ks_2samp(original[col], drifted[col])
        drift_results[col] = {
            "drift_statistic": float(stat),
            "p_value": float(pval),
            "drift_detected": bool(pval < 0.05),
            "type": "numerical",
        }

    for col in categorical_cols:
        contingency = pd.crosstab(original[col], drifted[col])
        chi2, pval, _, _ = chi2_contingency(contingency)
        drift_results[col] = {
            "chi2_statistic": float(chi2),
            "p_value": float(pval),
            "drift_detected": bool(pval < 0.05),
            "type": "categorical",
        }

    return drift_results


def log_drift_to_mlflow(drift_results, dataset_name="train"):
    with mlflow.start_run(run_name=f"drift_detection_{dataset_name}", nested=True):
        for col, metrics in drift_results.items():
            for k, v in metrics.items():
                if k != "type":
                    mlflow.log_metric(f"{dataset_name}_{col}_{k}", v)


if __name__ == "__main__":
    train = pd.read_csv("data/silver/preprocessed_ml2_student_performance.csv")
    test = pd.read_csv("data/silver/preprocessed_ml2_student_performance.csv")
    drifted_train = pd.read_csv("data/silver/drifted_train.csv")
    drifted_test = pd.read_csv("data/silver/drifted_test.csv")

    numerical_cols = ["Age", "Hours of Sleep", "Hours Reviewing"]
    categorical_cols = [
        "Course (STEM=1, Non-STEM=0)",
        "In a Relationship (1=Yes, 0=No, 0.5=It's complicated)",
    ]

    mlflow.set_experiment("student_performance_drift_detection")

    train_drift = calculate_drift(train, drifted_train, numerical_cols, categorical_cols)
    test_drift = calculate_drift(test, drifted_test, numerical_cols, categorical_cols)

    log_drift_to_mlflow(train_drift, "train")
    log_drift_to_mlflow(test_drift, "test")

    report = {"train_drift": train_drift, "test_drift": test_drift}
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(report, f, indent=4)