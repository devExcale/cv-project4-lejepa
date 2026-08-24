import os

import torch

from src.data import get_dataloaders
from src.globals import DEVICE, CONFIG, DIR_CHECKPOINTS, DATASETS, set_seed
from src.network import build_model


def get_checkpoint_path(dataset: str, arch: str, paradigm: str, tag: str = "best") -> str:
	filename = f"{dataset}_{arch}_{paradigm}_{tag}.pt"
	return os.path.join(DIR_CHECKPOINTS, filename)


def test_cuda():
	print(f"PyTorch Version: {torch.__version__}")
	print(f"CUDA Available:  {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"Device Name:     {torch.cuda.get_device_name(0)}")


def test_config(dataset: str, arch: str, paradigm: str):
	set_seed(CONFIG["seed"])
	print(f"Active Device:       {DEVICE}")
	print(f"Target Dataset:      {dataset}")
	print(f"Architecture:        {arch}")
	print(f"Training Paradigm:   {paradigm}")
	print(f"Number of Classes:   {DATASETS[dataset]['num_classes']}")
	print(f"Checkpoint Target:   {get_checkpoint_path(dataset, arch, paradigm)}")
	print("Configuration diagnostic verified successfully.")


def test_pipeline(dataset: str, arch: str, paradigm: str):
	print(f"Testing DataLoaders for {dataset}...")
	train_loader, val_loader = get_dataloaders(dataset_name=dataset, batch_size=8)
	images, labels = next(iter(train_loader))
	print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

	print(f"Instantiating model ({arch}, {paradigm}) for {dataset}...")
	model = build_model(arch=arch, dataset=dataset, paradigm=paradigm).to(DEVICE)
	images = images.to(DEVICE)

	if hasattr(model, "forward_features"):
		s1, s2, s3, s4 = model.forward_features(images)
		print(f"Stage 1 feature shape: {s1.shape}")
		print(f"Stage 2 feature shape: {s2.shape}")
		print(f"Stage 3 feature shape: {s3.shape}")
		print(f"Stage 4 feature shape: {s4.shape}")

	logits = model(images)
	print(f"Logits output shape:   {logits.shape}")
	print("Pipeline diagnostic completed successfully.")
