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
			transform = None,
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


class MultiViewDataset(HuggingFaceDataset):
	def __init__(
			self,
			dataset_name: str,
			global_transform,
			local_transform,
			split: str = "train",
			data_dir: str = DIR_DATA
	):
		super().__init__(
			dataset_name,
			split=split,
			transform=None,
			data_dir=data_dir
		)
		self.global_transform = global_transform
		self.local_transform = local_transform

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		item = self.dataset[idx]
		image = item["img"].convert("RGB")
		label = item[self.label_key]

		views = {
			"global": [self.global_transform(image) for _ in range(2)],
			"local": [self.local_transform(image) for _ in range(4)],
		}

		return views, label


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
		dataset_name,
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
		b, _, h, w = images.shape
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


def get_transforms(dataset_name: str):
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


def get_lejepa_transforms(dataset_name: str):
	mean, std = get_or_compute_stats(dataset_name)

	common = [
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	]

	global_transform = transforms.Compose([
		transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
		*common,
	])

	local_transform = transforms.Compose([
		transforms.RandomResizedCrop(16, scale=(0.2, 0.5)),
		*common,
	])

	return global_transform, local_transform


def get_dataloaders(
    dataset_name: str,
    batch_size: int = CONFIG["batch_size"],
    num_workers: int = CONFIG["num_workers"],
    data_dir: str = DIR_DATA,
    paradigm: str = "std",
    val_fraction: float = CONFIG["val_fraction"],
    seed: int = CONFIG["seed"],
    include_test: bool = False,
):
    if paradigm not in ("std", "lejepa"):
        raise ValueError("Unrecognized paradigm.")
    
    meta_data = DATASETS[dataset_name]
    raw_data = load_dataset(meta_data["hf_path"], split="train", cache_dir=data_dir)
    try:
        splits = raw_data.train_test_split(test_size=val_fraction, seed=seed, stratify_by_column=meta_data["label_key"])
    except (ValueError, TypeError):
        splits = raw_data.train_test_split(test_size=val_fraction, seed=seed)

    train_transform, eval_transform = get_transforms(dataset_name)
    if paradigm == "std":
        train_dataset = HuggingFaceDataset(dataset_name, transform=train_transform, data_dir=data_dir, dataset=splits["train"])
    else:
        global_transform, local_transform = get_lejepa_transforms(dataset_name)
        train_dataset = MultiViewDataset(dataset_name, global_transform, local_transform, data_dir=data_dir, dataset=splits["train"])
    #For LeJEPA this validation dataset will be used to test the linear probe.
    val_dataset = HuggingFaceDataset(dataset_name, transform=eval_transform, data_dir=data_dir, dataset=splits["test"])

    is_cuda = torch.cuda.is_available()
    common_loader = dict(num_workers=num_workers, pin_memory=is_cuda, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common_loader)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common_loader)

    if not include_test:
        return train_loader, val_loader

    test_dataset = HuggingFaceDataset(dataset_name, split="test", transform=eval_transform, data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common_loader)
    return train_loader, val_loader, test_loader

