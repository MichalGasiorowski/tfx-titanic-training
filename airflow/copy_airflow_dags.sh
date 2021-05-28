#!/bin/bash

AIRFLOW_HOME=$HOME/"airflow"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

# Copy code to ~/airflow/dags
mkdir -p $AIRFLOW_HOME/dags
cp ../pipeline/pipelines.py $AIRFLOW_HOME/dags/
cp ../pipeline/pipeline_args.py $AIRFLOW_HOME/dags/
cp ../pipeline/model.py $AIRFLOW_HOME/dags/
cp ../pipeline/features.py $AIRFLOW_HOME/dags/
cp ../pipeline/preprocessing.py $AIRFLOW_HOME/dags/
cp ../pipeline/config.py $AIRFLOW_HOME/dags/
cp ../pipeline/airflow_runner.py $AIRFLOW_HOME/dags/

# Copy data and other resources needed during pipeline execution
cp -R ../data $AIRFLOW_HOME/dags/
cp -R ../pipeline/hyperparameters $AIRFLOW_HOME/dags/
cp -R ../pipeline/schema $AIRFLOW_HOME/dags/
cp -R ../pipeline/client $AIRFLOW_HOME/dags/
cp -R ../pipeline/lib $AIRFLOW_HOME/dags/


