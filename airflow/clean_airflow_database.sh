#!/bin/bash

AIRFLOW_HOME=$HOME/"airflow"
echo "Clean airflow database"

airflow db reset --yes
airflow db init
