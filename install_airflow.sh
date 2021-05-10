#!/bin/bash

AIRFLOW_VERSION=2.0.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow[async,postgres,google]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

pip install papermill
pip install networkx
pip install matplotlib
pip install Werkzeug

# partially taken from on https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/setup_demo.sh
airflow db init

sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/load_examples = True/load_examples = False/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/max_threads = 2/max_threads = 1/g' ~/airflow/airflow.cfg

airflow db reset --yes
airflow db init


# Copy Dags to ~/airflow/dags
mkdir -p ~/airflow/dags
cp pipeline/pipelines.py ~/airflow/dags/
cp pipeline/pipeline_args.py ~/airflow/dags/
cp pipeline/model.py ~/airflow/dags/
cp pipeline/features.py ~/airflow/dags/
cp pipeline/preprocessing.py ~/airflow/dags/
cp pipeline/config.py ~/airflow/dags/
cp pipeline/airflow_runner.py ~/airflow/dags/

# Copy data to ~/airflow/data
cp -R data ~/airflow

