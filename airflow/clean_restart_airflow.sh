#!/bin/bash

AIRFLOW_HOME=$HOME/"airflow"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

./stop_airflow.sh

# delete dags folder in airflow
rm -rf $AIRFLOW_HOME/dags

# clean artifact store
rm -rf $AIRFLOW_HOME/artifact-store

# clean logs
rm -rf $AIRFLOW_HOME/logs/scheduler
rm -rf $AIRFLOW_HOME/logs/dag_processor_manager
rm -rf $AIRFLOW_HOME/logs/tfx-titanic-training

#clean db
# airflow db reset --yes
# airflow db init

echo 'Copy pipeline files to airflow dags directory.'
./copy_airflow_dags.sh

./start_airflow.sh

