from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.preprocessing import preprocess_data
from src.training import training
from src.evaluation import evaluation

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 3),
    'retries': 1,
}

with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['dag1', 'dag2'],
) as dag:

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
    preprocess_task >> train_task >> evaluate_task