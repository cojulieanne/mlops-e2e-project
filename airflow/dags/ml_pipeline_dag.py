from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import mlflow
import json
import os

from src.load_data.load_data import load
from src.preprocessing.preprocess_data import preprocess_data
# from src.feature_engineering.split_data import split_data
from src.training.training_mlflow import get_default_binary_models
from src.evaluation.evaluation import evaluation
from src.drift_detection import detect_drift


def run_drift_detection():
    report_path = "reports/drift_report.json"
    results = detect_drift(
        "data/silver/preprocessed_ml2_student_performance.csv",
        "data/silver/drifted_test.csv",
    )
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    return results


def branch_on_drift():
    report_path = "reports/drift_report.json"
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"{report_path} not found.")
    with open(report_path, "r") as f:
        results = json.load(f)
    if results.get("drift_detected", False):
        return "retrain_model"
    else:
        return "pipeline_complete"


with DAG(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2022, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    mlflow.set_tracking_uri("http://mlflow:5000")

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    # feature_eng_task = PythonOperator(
    #     task_id="feature_engineering",
    #     python_callable=split_data,
    # )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=get_default_binary_models(cv=5),
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluation,
    )

    drift_task = PythonOperator(
        task_id="drift_detection",
        python_callable=run_drift_detection,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=get_default_binary_models(cv=5),
    )

    pipeline_complete = EmptyOperator(task_id="pipeline_complete")

    # Dependencies
    load_task >> preprocess_task >> train_task >> evaluate_task
    evaluate_task >> drift_task >> branch_task
    branch_task >> [retrain_task, pipeline_complete]