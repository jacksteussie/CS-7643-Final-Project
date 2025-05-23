# Aerial Object Detection with DOTA

## Dependencies

- conda (of any form, but we recommend using Miniconda)
- git

## Environment Setup
Before running, run the following in your terminal:
Before you run the `setup.sh` script, make sure, if needed, you change the parameters in `src/split_data.py` to match your needs. This file splits the data such that there are multiple resolutions used in the dataset. If single scale is being
used, make sure to just set `DOTA_MOD_DIR=None` in `src/const.py`. 

Now run 
```bash
bash setup.sh cuda=true
```
if you have a GPU and want to use it. If you don't have a GPU, run 
```bash
bash setup.sh cuda=false
```
This will create a conda environment called `cs7643-project` and install all the required packages.
It will also download the dataset and split it if it's setup for that. 

If you don't want to download the dataset or don't want to split it, you can run the following commands:
```bash
bash setup.sh cuda=true download=false split=false
```

The transformer has a separate build script to run, so run that one instead for a separate environment.

## Dataset

The DOTA dataset is a large-scale aerial image dataset for object detection. It contains images with various objects, including planes, ships, storage tanks, and more. The dataset is divided into training and testing sets, with annotations provided in the form of bounding boxes.

The dataset contains the following classes:
- 0: plane
- 1: ship
- 2: storage tank
- 3: baseball diamond
- 4: tennis court
- 5: basketball court
- 6: ground track field
- 7: harbor
- 8: bridge
- 9: large vehicle
- 10: small vehicle
- 11: helicopter
- 12: roundabout
- 13: soccer ball field
- 14: swimming pool
- 15: container crane

## RoI Transformer
The RoI Transformer creates checkpoints at each iteration, to avoid uploading excess data, these chekpoints are omitted here. Please see "RoI_contributions.txt" for details.
Additionally, the following files need to be downloaded from the OBBDetection repo for the code to build properly:
- OBBDetection/mmdet/models
- OBBDetection/mmdet/ops

## Faster RCNN (AABB) Instructions

cd into the `src/` directory and run ```python -m models.FasterRCNN.train``` to train the model. 
Any parameters can be changed in the config files residing within `src/models/FasterRCNN/configs/`. 
The model will save the best weights to the `src/checkpoints/` directory. 
Visualizations of the model results can be found in the jupyter notebook at `src/viz.ipynb`. 

## Yolo Instructions
### all within models/YOLO

1. Compare 3 yolo model sizes: 
- run yolo_experiments_3models_10epochs.py
- run plot_3models_10epochs.py to get the results plots.
2. Train 50 epoch medium model:
- run yolo_experiments_mModel_50epochs.py
3. Plot train and validation losses:
- run plot_train_val_mModel_50epochs.py
4. Plot mAP validation
- run plot_map50.py
5. Predict on test file: 
- run predict.py
