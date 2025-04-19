#!/bin/bash
set -euo pipefail

print_help() {
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

ZIP_FILE="dotav1-5.zip"
EXTRACT_FOLDER="DOTAv1.5"
ROOT_DIR=$(pwd)

# Load conda env
CONDA_BASE="$(conda info --base)"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  source "$CONDA_BASE/bin/activate"
fi

echo "🔧 Setting up conda environment..."
if conda env list | grep -q "cs7643-project"; then
  echo "🔄 Updating existing conda environment... (cs7643-project)"
  conda env update -f environment.yaml --prune ||  echo "⚠️ Conda update failed, continuing..."
else
  echo "🆕 Creating new conda environment... (cs7643-project)"
  conda env create -f environment.yaml || true
fi
echo "🚀 Conda environment ready"


echo "📦 Activating environment..."
conda activate cs7643-project

echo "⚙️ Installing PyTorch..."
if [[ "$USE_CUDA" == true ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo "🔥 PyTorch installed"

echo "🛰️ (Re)Installing DOTA dev kit..."
cd src
if [[ -d DOTA_devkit ]]; then
  echo "🧹 Removing existing DOTA_devkit..."
  rm -rf DOTA_devkit
fi
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
echo "✅ DOTA dev kit installed!"
cd "$ROOT_DIR"

if [[ "$DO_DOWNLOAD" == true ]]; then
  echo "📥 Downloading DOTA dataset..."
  curl -L -o "$ZIP_FILE" https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.5.zip
  unzip "$ZIP_FILE"
  rm "$ZIP_FILE"

  mkdir -p "$ROOT_DIR/data/dota"
  mv "$EXTRACT_FOLDER/images" "$ROOT_DIR/data/dota/"
  mv "$EXTRACT_FOLDER/labels" "$ROOT_DIR/data/dota/"
  rm -rf "$EXTRACT_FOLDER"
fi

if [[ "$DO_SPLIT" == true ]]; then
  echo "✂️ Splitting data..."
  cd src
  python -m split_data.py
fi

echo "Environment ready! Happy training! 🙂"