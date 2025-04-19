import os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOTA_DIR = os.path.join(DATA_DIR, "dota")
DOTA_MOD_DIR = os.path.join(DATA_DIR, "dota-mod")
# DOTA_MOD_DIR = None
MODEL_STORAGE = os.path.join(PROJECT_ROOT, "models")