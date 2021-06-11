# tfx-titanic-training

The repo provides examples how to run E2E tfx pipeline, 
taking "Titanic - Machine Learning from Disaster" kaggle competition.

https://www.kaggle.com/c/titanic

This branch works on tfx 0.30.0 version.

see:
https://github.com/tensorflow/tfx

The script to install dependencies is install_tfx_0.30.0.sh
It's recommended to do it using python virtual env, like so:

`python3 -m ~/venv ML-tfx-0.30.0

source ~/venv/ML-tfx-0.30.0/bin/activate

./install_tfx_0.30.0.sh
`

## Notebooks

### Local run example
[local-pipeline-run.ipynb](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/local-pipeline-run.ipynb)
### Kubeflow run example
[pipeline-template.ipynb](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline-template.ipynb)
### Tensorboard running notebook
[start_tensorboard.ipynb](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/start_tensorboard.ipynb)
### Metadata browsing, tfx components (Schema, Statistics, Anomalies) visualization notebook
[start_tensorboard.ipynb](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/browse-tfx-metadata.ipynb)



## The code for the pipeline, tfx components, preprocessing etc. 
[/pipeline](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline)


TFX pipeline creation, supporting differences/quirks between local, airflow and kubeflow:

[/pipeline/pipelines.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/pipelines.py)

class to wrap logic for env parameters setting:

[/pipeline/config.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/config.py)


Helper functions used when building keras wide & deep model and preprocessing: specify which features are numerical, 
which should be bucketized, how many buckets, how transformed features should be named, etc. :

[/pipeline/features.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/features.py)

 
Preprocessing functions, used in tfx Transform component:
[/pipeline/preprocessing.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/preprocessing.py)


Main file, where tuner_fn & run_fn are defined, which are tfx Trainer & Tuner components entry points:
[/pipeline/model.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/model.py)

 
Helper class for TrainerConfig, TunerConfig, PusherConfig, RuntimeParametersConfig
parameters grouping ( passed to create_pipeline )
[/pipeline/pipeline_args.py](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/pipeline_args.py)


Client code:

[/pipeline/client/](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/client/)

Both in-memory client ( using `tf.saved_model.load` )
and ProdClient ( using TF Serving ) 

Util functions in client_util.py for csv data reading, conversion to Proto etc

## Airlfow
[airflow](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/airflow)

helper scripts for installing airflow locally ( under $HOME/airflow )
starting/stopping airflow webserver & airflow scheduler

more on:
https://airflow.apache.org/docs/apache-airflow/stable/index.html

## Data
[data](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/data)
Train/test Titanic data, taken from Kaggle.
It can be also downloaded using kaggle-cli, like so:

`kaggle competitions download -c titanic -p {local_data_dirpath} --force`

## Tests
[tests](https://github.com/MichalGasiorowski/tfx-titanic-training/blob/master/pipeline/tests)

Tests for pipeline creation, local pipeline run, in-memory-client, prod client ( TF Serving )

