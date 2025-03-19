#!/bin/bash

# WARNING: This script may take a bit but don't worry it'll tell you if it's not working
curl -L -o ./data/xview-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hassanmojab/xview-dataset
cd ./data/
unzip xview-dataset.zip