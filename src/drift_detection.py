import pandas as pd
import json
import os
from typing import Dict, Any
from evidently import Report
from evidently.presets import DataDriftPreset


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
    
    eval = report.run(
        reference_data=reference_features,
        current_data=current_features
    )
    report_dict = eval.dict()

    
    overall_metric = report_dict["metrics"][0]
    overall_drift_score = overall_metric["value"]["share"]
    drifted_columns = overall_metric["value"]["count"]
    drift_detected = drifted_columns > 0  

    feature_drifts = {
        m["metric_id"].replace("ValueDrift(column=", "").rstrip(")"): float(m["value"])
        for m in report_dict["metrics"]
        if m["metric_id"].startswith("ValueDrift")
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