#!/bin/bash
set -eo pipefail

cd notebooks

python Sentinels_features.py
python Model.py
