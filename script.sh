#!/bin/bash

module load python/3.7
virtualenv --no-download ~/ift6759-env
source ~/ift6759-env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index --upgrade tensorflow_gpu
pip install -r requirements.txt
