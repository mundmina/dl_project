import os
import random
from pathlib import Path

import numpy as np

# Colab layout: data is copied to fast local disk, archives are cached on Drive.
# On a local machine set BGBIAS_DATA (any local folder) and BGBIAS_DRIVE (the bgbias folder as
# Google Drive for desktop shows it, e.g. "G:/My Drive/bgbias"), so checkpoints land in the shared Drive.
LOCAL_DATA = Path(os.environ.get("BGBIAS_DATA", "/content/data"))
DRIVE_ROOT = Path(os.environ.get("BGBIAS_DRIVE", "/content/drive/MyDrive/bgbias"))

PET_SUBDIR = "oxford-iiit-pet"
PLACES_SUBDIR = "places365"


def pet_root(data_dir=LOCAL_DATA):
    return Path(data_dir) / PET_SUBDIR


def places_root(data_dir=LOCAL_DATA):
    return Path(data_dir) / PLACES_SUBDIR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
