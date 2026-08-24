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

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		item = self.dataset[idx]
		image = item["img"].convert("RGB")
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

	print(f"Channel statistics not cached for '{dataset_name}'. Computing from training split...")
	raw_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split="train",
		transform=transforms.ToTensor(),
		data_dir=DIR_DATA
	)
	raw_loader = DataLoader(raw_dataset, batch_size=256, shuffle=False, num_workers=0)

	channel_sum = torch.zeros(3, dtype=torch.float64)
	channel_sum_sq = torch.zeros(3, dtype=torch.float64)
	num_pixels = 0

	for images, _ in tqdm(raw_loader, desc=f"Calculating Stats [{dataset_name}]"):
		# images shape: [B, C, H, W]
		b, c, h, w = images.shape
		num_pixels += b * h * w
		channel_sum += images.sum(dim=[0, 2, 3])
		channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

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
	mean, std = get_or_compute_stats(dataset_name)

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

	return train_transform, val_transform


def get_dataloaders(
		dataset_name: str,
		batch_size: int = CONFIG["batch_size"],
		num_workers: int = CONFIG["num_workers"],
		data_dir: str = DIR_DATA
) -> Tuple[DataLoader, DataLoader]:
	train_transform, val_transform = get_transforms(dataset_name)

	train_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split="train",
		transform=train_transform,
		data_dir=data_dir
	)
	val_dataset = HuggingFaceDataset(
		dataset_name=dataset_name,
		split="test",
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
