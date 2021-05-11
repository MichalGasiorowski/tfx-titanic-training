#!/bin/bash
# see more: https://github.com/tensorflow/tfx/blob/master/tfx/examples/airflow_workshop/setup/setup_demo.sh

AIRFLOW_HOME=$HOME/"airflow"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

AIRFLOW_VERSION=2.0.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow[async,postgres,google]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

pip install papermill
pip install networkx
pip install matplotlib
pip install Werkzeug

airflow db init

sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 10/g' $AIRFLOW_HOME/airflow.cfg
sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' $AIRFLOW_HOME/airflow.cfg
sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' $AIRFLOW_HOME/airflow.cfg
sed -i'.orig' 's/load_examples = True/load_examples = False/g' $AIRFLOW_HOME/airflow.cfg
sed -i'.orig' 's/max_threads = 2/max_threads = 1/g' $AIRFLOW_HOME/airflow.cfg

airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin

airflow db reset --yes
airflow db init


