import os

import torch

from src.data import get_dataloaders
from src.globals import DEVICE, CONFIG, DIR_CHECKPOINTS, DATASETS, set_seed
from src.network import build_model


def get_checkpoint_path(dataset: str, arch: str, paradigm: str, tag: str = "best") -> str:
	return os.path.join(DIR_CHECKPOINTS, f"{dataset}_{arch}_{paradigm}_{tag}.pt")


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


def test_pipeline(dataset: str, arch: str, paradigm: str):
	train_loader, _ = get_dataloaders(dataset_name=dataset, batch_size=8, paradigm=paradigm)
	batch, labels = next(iter(train_loader))
	model = build_model(arch=arch, dataset=dataset, paradigm=paradigm).to(DEVICE)

	if paradigm == "lejepa":
		global_views = [x.to(DEVICE) for x in batch["global"]]
		local_views = [x.to(DEVICE) for x in batch["local"]]
		loss, inv, sig = model(global_views=global_views, local_views=local_views)
		print(f"LeJEPA loss: {loss.item():.4f} | Inv: {inv.item():.4f} | SIGReg: {sig.item():.4f}")
		images = global_views[0]
	else:
		images = batch.to(DEVICE)
		print(f"Logits output shape: {model(images).shape}")

	features = model.forward_features(images)
	for i, feature in enumerate(features, 1):
		print(f"Layer {i} feature shape: {tuple(feature.shape)}")
