import pandas as pd
import json
import os
from typing import Dict, Any
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)


    if "target" in reference_data.columns:
        reference_features = reference_data.drop(columns=["target"])
        current_features = current_data.drop(columns=["target"])
    else:
        reference_features = reference_data
        current_features = current_data

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    report_dict = report.as_dict()
    drift_result = report_dict["metrics"][0]["result"]

    drift_detected = drift_result["dataset_drift"]
    overall_drift_score = drift_result["drift_share"]

    feature_drifts = {
        item["column_name"]: item["drift_score"]
        for item in drift_result["drift_by_columns"]
    }

    result = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall_drift_score,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    output = detect_drift(
        "data/silver/preprocessed_ml2_student_performance.csv",
        "data/silver/drifted_test.csv",
    )