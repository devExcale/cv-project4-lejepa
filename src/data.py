import json
import os
from typing import Tuple, Optional, Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.globals import DIR_DATA, PATH_DATASET_STATS, DATASETS, CONFIG


class HuggingFaceDataset(Dataset):
	"""
	PyTorch Dataset wrapper for Hugging Face datasets.
	"""

	def __init__(
			self,
			dataset_name: str,
			split: str = "train",
			transform: Optional[transforms.Compose] = None,
			data_dir: str = DIR_DATA
	):
		if dataset_name not in DATASETS:
			raise ValueError(f"Dataset '{dataset_name}' not recognized. Registered: {list(DATASETS.keys())}")

		self.spec = DATASETS[dataset_name]
		self.dataset = load_dataset(
			self.spec["hf_path"],
			split=split,
			cache_dir=data_dir
		)
		self.transform = transform
		self.label_key = self.spec["label_key"]
		self.image_key = self.spec.get("image_key", "image")

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		item = self.dataset[idx]

		raw_img = item.get(self.image_key) or item.get("img") or item.get("image")
		image = raw_img.convert("RGB")
		label = item[self.label_key]

		if self.transform:
			image = self.transform(image)

		return image, label


def get_or_compute_stats(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
	stats_cache: Dict[str, Dict[str, list]] = {}
	if os.path.exists(PATH_DATASET_STATS):
		try:
			with open(PATH_DATASET_STATS, "r") as f:
				stats_cache = json.load(f)
		except Exception:
			stats_cache = {}

	if dataset_name in stats_cache:
		mean = tuple(stats_cache[dataset_name]["mean"])
		std = tuple(stats_cache[dataset_name]["std"])
		return mean, std

	# Return standard constants for ImageNet/ImageNet-100
	if "imagenet" in dataset_name.lower():
		mean_list = [0.485, 0.456, 0.406]
		std_list = [0.229, 0.224, 0.225]
		stats_cache[dataset_name] = {"mean": mean_list, "std": std_list}
		with open(PATH_DATASET_STATS, "w") as f:
			json.dump(stats_cache, f, indent=4)
		return tuple(mean_list), tuple(std_list)

	print(f"Channel statistics not cached for '{dataset_name}'. Computing from training split...")
	raw_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split="train",
		transform=transforms.ToTensor(),
		data_dir=DIR_DATA
	)

	channel_sum = torch.zeros(3, dtype=torch.float64)
	channel_sum_sq = torch.zeros(3, dtype=torch.float64)
	num_pixels = 0

	# Iterate sample-by-sample to support variable (non-homogeneous) image sizes
	for img, _ in tqdm(raw_dataset, desc=f"Calculating Stats [{dataset_name}]"):
		# img shape: [C, H, W]
		c, h, w = img.shape
		num_pixels += h * w
		channel_sum += img.to(torch.float64).sum(dim=[1, 2])
		channel_sum_sq += (img.to(torch.float64) ** 2).sum(dim=[1, 2])

	mean = channel_sum / num_pixels
	std = torch.sqrt((channel_sum_sq / num_pixels) - (mean ** 2))

	mean_list = [round(float(m), 4) for m in mean]
	std_list = [round(float(s), 4) for s in std]

	stats_cache[dataset_name] = {"mean": mean_list, "std": std_list}
	with open(PATH_DATASET_STATS, "w") as f:
		json.dump(stats_cache, f, indent=4)

	print(f"Calculated & cached stats for {dataset_name}: Mean={mean_list}, Std={std_list}")
	return tuple(mean_list), tuple(std_list)


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
	spec = DATASETS[dataset_name]
	input_size = spec.get("input_size", 32)

	# Use standard ImageNet stats for ImageNet, otherwise compute/fetch stats
	if "imagenet" in dataset_name:
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
	else:
		mean, std = get_or_compute_stats(dataset_name)

	# 32x32 transforms (CIFAR)
	if input_size == 32:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])
		val_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])
	# 224x224 transforms (ImageNet-100 / ImageNet)
	else:
		train_transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])
		val_transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])

	return train_transform, val_transform


def get_dataloaders(
		dataset_name: str,
		batch_size: int = CONFIG["batch_size"],
		num_workers: int = CONFIG["num_workers"],
		data_dir: str = DIR_DATA
) -> Tuple[DataLoader, DataLoader]:
	train_transform, val_transform = get_transforms(dataset_name)
	val_split = DATASETS[dataset_name].get("val_split", "test")

	train_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split="train",
		transform=train_transform,
		data_dir=data_dir
	)
	val_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split=val_split,
		transform=val_transform,
		data_dir=data_dir
	)

	is_cuda = torch.cuda.is_available()

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=is_cuda,
		drop_last=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=is_cuda,
		drop_last=False
	)

	return train_loader, val_loader
