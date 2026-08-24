import os
import random
from typing import Dict, Any

import numpy as np
import torch

# Base project directories
DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(DIR_PROJECT, "data")
DIR_CHECKPOINTS = os.path.join(DIR_PROJECT, "checkpoints")
DIR_OUTPUT = os.path.join(DIR_PROJECT, "output")
PATH_DATASET_STATS = os.path.join(DIR_DATA, "dataset_stats.json")

# Ensure required directories exist on launch
for dir_path in [DIR_DATA, DIR_CHECKPOINTS, DIR_OUTPUT]:
	os.makedirs(dir_path, exist_ok=True)

# Redirect Hugging Face caches to local data directory
os.environ["HF_DATASETS_CACHE"] = DIR_DATA
os.environ["HF_HOME"] = DIR_DATA

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implemented datasets
DATASETS: Dict[str, Dict[str, Any]] = {
	"cifar10": {
		"hf_path": "uoft-cs/cifar10",
		"label_key": "label",
		"num_classes": 10,
	},
	"cifar100": {
		"hf_path": "uoft-cs/cifar100",
		"label_key": "fine_label",
		"num_classes": 100,
	}
}

# Default Hyperparams
CONFIG = {
	"seed": 42,
	"batch_size": 64 if not torch.cuda.is_available() else 128,
	"num_workers": 2 if torch.cuda.is_available() else 0,
	"lr": 0.1,
	"weight_decay": 5e-4,
	"epochs": 100,
	"momentum": 0.9,
	"device": DEVICE
}

# Try increase CPU performance?
num_physical_cores = os.cpu_count() // 2 or os.cpu_count()
torch.set_num_threads(num_physical_cores)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
os.environ["MKL_NUM_THREADS"] = str(num_physical_cores)


def set_seed(seed: int = 42) -> None:
	"""
	Ensure deterministic behaviour across NumPy and PyTorch.
	"""

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	return
