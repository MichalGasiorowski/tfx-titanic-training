#!/bin/bash
# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install KFP and TFX SDKs

# pip install -q -U -v --log /tmp/pip.log tfx==0.25.0 apache-beam[gcp]==2.25.0 ml-metadata==0.25.0 pyarrow==0.17.0 tensorflow==2.3.0 tensorflow-data-validation==0.25.0 tensorflow-metadata==0.25.0 tensorflow-model-analysis==0.25.0 tensorflow-serving-api==2.3.0 tensorflow-transform==0.25.0 tfx-bsl==0.25.0

cat > requirements.txt << EOF
pandas>=1.2.4
apache-beam[gcp]==2.28.0
ml-metadata==0.29.0
pyarrow==2.0.0
numpy==1.19.5
tensorflow==2.4.0
tensorflow-data-validation==0.30.0
tensorflow-metadata==0.30.0
tensorflow-model-analysis==0.30.0
tensorflow-serving-api==2.4.0
tensorflow-transform==0.30.0
tfx-bsl==0.30.0
tfx==0.30.0
tensorflow-cloud
tensorboard
kfp==1.4.0
EOF

python -m pip install -r requirements.txt

# tfx==0.30.0
# install from github since there is no tfx==0.30.0 in pypi yet
#pip install https://github.com/tensorflow/tfx/archive/refs/heads/r0.30.0.zip

# Install Skaffold
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
chmod +x skaffold
mv skaffold $HOME/.local/bin

jupyter nbextension enable --py tensorflow_model_analysis