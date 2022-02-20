import yaml
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable

from airflow.operators.python_operator import PythonOperator

from data_prep import prepare_data
from hyperparameter_tune import tune_hyperparams
from train_model import train_and_save_model
from predict import predict_and_evaluate

# Basic config -------------------------------------------------------------------------------------------
with open('python/configs.yaml', 'r') as file:
    configs = yaml.safe_load(file)

default_args = {
    'owner': 'mwyss',
    'depends_on_past': False,
    'start_date': datetime(2022, 2, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def get_best_params(ti):
    """Gets the best params from the tuning step"""
    return ti.xcom_pull(key='return_value', task_ids='tune_hyperparams')

# DAG config --------------------------------------------------------------------------------------------
with DAG(
    dag_id='student_performance_dag',
    description='DAG prepares data, tunes model, trains model, predicts on \
        test data, and saves model and performance metrics.',
    schedule_interval='@once',
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
) as dag:

    download_and_prep_data = PythonOperator(
        task_id='download_and_prep_data',
        python_callable=prepare_data,
        op_kwargs={'configs': configs}
    )

    tune_hyperparams = PythonOperator(
        task_id='tune_hyperparams',
        pyton_callable=tune_hyperparams,
        op_kwargs={'configs': configs}
    )

    train_and_save_model = PythonOperator(
        task_id='train_and_save_model',
        pyton_callable=train_and_save_model,
        op_kwargs={
            'configs': configs,
            'params': get_best_params()
        }
    )

    predict_and_evaluate = PythonOperator(
        task_id='predict_and_evaluate',
        pyton_callable=predict_and_evaluate,
        op_kwargs={'configs': configs}
    )

# Task config ------------------------------------------------------------------------------------------
    download_and_prep_data.set_downstream(tune_hyperparams)
    tune_hyperparams.set_downstream(train_and_save_model)
    train_and_save_model.set_downstream(predict_and_evaluate)
