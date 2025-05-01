#!/bin/bash

set -eou pipefail

print_help() {
  echo "TRANSFORMER MODEL ONLY WORKS ON WINDOWS AND LINUX MACHINES"
  echo "Usage: $0 [--cuda=true|false] [--download=true|false] [--split=true|false]"
  echo
  echo "Arguments:"
  echo "  --cuda=true|false      Install CUDA-enabled PyTorch (default: true)"
  echo "  --download=true|false  Download the DOTA dataset (default: true)"
  echo "  --split=true|false     Run the dataset splitting script (default: true)"
  echo
  echo "Examples:"
  echo "  $0 --cuda=true --download=false --split=true"
  echo "  $0 --cuda=false --split=false"
  echo "  $0 --help"
}

# Defaults
USE_CUDA=true
DO_DOWNLOAD=true
DO_SPLIT=true


# Parse args
for arg in "$@"; do
  case $arg in
    --help|-h)
      print_help
      exit 0
      ;;
    --cuda=true)
      USE_CUDA=true
      ;;
    --cuda=false)
      USE_CUDA=false
      ;;
    --download=true)
      DO_DOWNLOAD=true
      ;;
    --download=false)
      DO_DOWNLOAD=false
      ;;
    --split=true)
      DO_SPLIT=true
      ;;
    --split=false)
      DO_SPLIT=false
      ;;
    *)
      echo "❌ Unknown argument: $arg"
      print_help
      exit 1
      ;;
  esac
done

# Load conda env
CONDA_BASE="$(conda info --base)"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  source "$CONDA_BASE/bin/activate"
fi

echo "🔧 Setting up conda environment..."
conda create --name cs7643-project-tran
conda activate cs7643-project-tran

echo "⚙️ Installing PyTorch..."
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
echo "🔥 PyTorch installed"

pip install -U openmim
pip install mmcv-full
pip install mmdet\<3.0.0

cd OBBDetection/BboxToolkit
pip install -v -e .
cd ..

pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -r requirements/build.txt
pip install mmpycocotools
pip install -v -e .

cd BboxToolkit/tools
python img_split.py --base_json split_configs/dota1_5/custom_ss_dota_train_realtime.json
python img_split.py --base_json split_configs/dota1_5/custom_ss_dota_val.json

cd ../..

