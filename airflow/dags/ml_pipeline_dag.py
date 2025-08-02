from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.load_data import load_data
from src.feature_engineering import split_data
from src.preprocessing import preprocess_data
from src.training import training
from src.evaluation import evaluation


with DAG(
    dag_id='ml_pipeline_dag',
    start_date=datetime(2022, 1, 1),
    schedule_interval=None,
) as dag:

    loading_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    
    feat_data = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=training,
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluation,
    )

   # Set dependencies
    loading_data >> feat_data >> preprocess_task >> train_task >> evaluate_task