import json
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import get_dataloaders, get_or_compute_stats
from src.evaluation import pca_outputs
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


def denormalize(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
	mean, std = get_or_compute_stats(dataset_name)
	mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
	std_tensor = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
	return (images.cpu() * std_tensor + mean_tensor).clamp(0, 1)


def checkpoint_metadata(checkpoint: dict) -> dict:
	keys = (
		"epoch",
		"dataset",
		"arch",
		"paradigm",
		"val_acc",
		"train_lejepa_loss",
		"loss",
	)
	return {key: checkpoint[key] for key in keys if key in checkpoint}


def save_pca(
		images: torch.Tensor,
		labels: torch.Tensor,
		features,
		layer_indices,
		sample_indices,
		args,
		checkpoint: dict,
		checkpoint_path: str,
):
	originals = denormalize(images, args.dataset)
	root = os.path.join(args.output_dir, "pca", args.dataset, f"{args.arch}_{args.paradigm}")
	saved = []

	for layer_index in layer_indices:
		feature_map = features[layer_index]
		for batch_index, sample_index in enumerate(sample_indices):
			result = pca_outputs(feature_map, image_index=batch_index)
			result_dir = os.path.join(
				root,
				f"layer_{layer_index:02d}",
				f"sample_{sample_index:05d}",
			)
			os.makedirs(result_dir, exist_ok=True)

			metadata = {
				"dataset": args.dataset,
				"split_seed": CONFIG["seed"],
				"sample_index": sample_index,
				"label": int(labels[batch_index]),
				"architecture": args.arch,
				"paradigm": args.paradigm,
				"layer_index": layer_index,
				"pca_components": 3,
				"checkpoint_path": checkpoint_path,
				"checkpoint": checkpoint_metadata(checkpoint),
			}

			payload = {"original": originals[batch_index], "components": result["components"], "mask": result["mask"], "rgb": result["rgb"], "metadata": metadata}
			torch.save(payload, os.path.join(result_dir, "pca.pt"))
			save_image(originals[batch_index], os.path.join(result_dir, "original.png"))
			save_image(result["rgb"].permute(2, 0, 1), os.path.join(result_dir, "pca_rgb.png"))
			save_image(result["mask"].float().unsqueeze(0), os.path.join(result_dir, "pca_mask.png"))
			with open(os.path.join(result_dir, "metadata.json"), "w") as file:
				json.dump(metadata, file, indent=2)
			saved.append(result_dir)
	return saved