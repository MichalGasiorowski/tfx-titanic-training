#!/bin/bash

AIRFLOW_HOME=$HOME/"airflow"
echo "AIRFLOW_HOME: $AIRFLOW_HOME"

AIRFLOW_WEBSERVER_PID=`cat $AIRFLOW_HOME/airflow-webserver.pid`
echo "AIRFLOW_WEBSERVER_PID: $AIRFLOW_WEBSERVER_PID"

echo "Kill airflow webserver"
kill -9 $AIRFLOW_WEBSERVER_PID
kill -9 $(ps -ef | grep "airflow webserver" | awk '{print $2}')
kill -9 $(ps -ef | grep "airflow-webserver" | awk '{print $2}')

AIRFLOW_SCHEDULER_PID=`cat $AIRFLOW_HOME/airflow-scheduler.pid`
echo "AIRFLOW_SCHEDULER_PID: $AIRFLOW_SCHEDULER_PID"

echo "Kill airflow scheduler"
kill -9 $AIRFLOW_SCHEDULER_PID
kill -9 $(ps -ef | grep "airflow scheduler" | awk '{print $2}')

echo 'remove pid files for webserver and scheduler'
rm $AIRFLOW_HOME/airflow-webserver.pid
rm $AIRFLOW_HOME/airflow-webserver-monitor.pid
rm $AIRFLOW_HOME/airflow-scheduler.pid




