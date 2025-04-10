# Aerial Object Detection with DOTA

## Environment Setup
Before running, run the following in your terminal:

```bash
conda env create -n environment.yaml
```

After that, do this if you have NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

and this if you are using only CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Before you run the `setup.sh` script, make sure, if needed, you change the parameters in `src/split_data.py` to match your needs. This file splits the data such that there are multiple resolutions used in the dataset. If single scale is being
used, make sure to just set `DOTA_MOD_DIR=None` in `src/const.py`. 

Now run 
```bash
bash setup.sh
```
to download the dataset and split it if it's setup for that. 

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