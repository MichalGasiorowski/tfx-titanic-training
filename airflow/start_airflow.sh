#!/bin/bash

AIRFLOW_HOME=$HOME/"airflow"
echo 'Start webserver in the background'
airflow webserver -p 8080 -D
sleep 3s
echo 'Start scheduler in the background'
airflow scheduler -D

