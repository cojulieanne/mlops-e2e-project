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
    report.run(reference_features, current_features)

    report_dict = report.as_dict()

    drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]


    feature_drift_info = report_dict["metrics"][1]["result"]["drift_by_columns"]

    selected_features = list(feature_drift_info.keys())[:3]

    feature_drifts = {
        f: feature_drift_info[f]["drift_score"] for f in selected_features
    }

    overall_drift_score = (
        sum(feature_drifts.values()) / len(feature_drifts)
        if feature_drifts
        else 0.0
    )

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