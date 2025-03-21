#!/bin/bash

# WARNING: This script may take a bit but don't worry it'll tell you if it's not working
# if it doesn't work, try running in root
curl -L -o ./data/xview-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hassanmojab/xview-dataset
cd ./data/
unzip xview-dataset.zip
rm xview-dataset.zip
rm __notebook_source__.ipynb
mv train_images/train_images/* train_images/ && rm -r train_images/train_images
mv val_images/val_images/* val_images/ && rm -r val_images/val_images
mkdir xview/
mkdir xview/train && mv train_images/* xview/train && rm -r train_images
mkdir xview/val && mv val_images/* xview/val && rm -r val_images
mv train_labels/xView_train.geojson xview/ && rm -r train_labels