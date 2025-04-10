#!/bin/bash

ZIP_FILE="dotav1-5.zip"
EXTRACT_FOLDER="DOTAv1.5"
ROOT_DIR=$(pwd)
curl -L -o $ZIP_FILE https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.5.zip
unzip $ZIP_FILE
rm $ZIP_FILE
mkdir -p $ROOT_DIR/data/dota
mv $EXTRACT_FOLDER/images $ROOT_DIR/data/dota/
mv $EXTRACT_FOLDER/labels $ROOT_DIR/data/dota/
rm -r $EXTRACT_FOLDER
cd src
python -m split_data.py