import json
import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from datasets import Dataset as HFDataset, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from src.globals import CONFIG, DATASETS, DIR_DATA, PATH_DATASET_STATS, DIR_OUTPUT


class HuggingFaceDataset(Dataset):
    """PyTorch Dataset wrapper for Hugging Face image datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str | HFDataset = "train",
        transform=None,
        data_dir: str = DIR_DATA,
    ):
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not recognized. Registered: {list(DATASETS.keys())}"
            )

        self.spec = DATASETS[dataset_name]
        if isinstance(split, str):
            self.dataset = load_dataset(
                self.spec["hf_path"],
                split=split,
                cache_dir=data_dir,
            )
        else:
            self.dataset = split
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
        split: str | HFDataset = "train",
        data_dir: str = DIR_DATA,
    ):
        super().__init__(
            dataset_name,
            split=split,
            transform=None,
            data_dir=data_dir,
        )
        self.global_transform = global_transform
        self.local_transform = local_transform

    def __getitem__(self, idx: int) -> Tuple[dict, int]:
        item = self.dataset[idx]
        image = item["img"].convert("RGB")
        label = item[self.label_key]

        views = {
            "global": [self.global_transform(image) for _ in range(2)],
            "local": [self.local_transform(image) for _ in range(4)],
        }

        return views, label


def _split_training_data(
    dataset_name: str,
    data_dir: str,
    val_fraction: float,
    seed: int,
):
    meta_data = DATASETS[dataset_name]
    raw_data = load_dataset(meta_data["hf_path"], split="train", cache_dir=data_dir)
    try:
        return raw_data.train_test_split(
            test_size=val_fraction,
            seed=seed,
            stratify_by_column=meta_data["label_key"],
        )
    except (ValueError, TypeError):
        return raw_data.train_test_split(test_size=val_fraction, seed=seed)


def _stats_cache_key(dataset_name: str, val_fraction: float, seed: int) -> str:
    return f"{dataset_name}::val_fraction={val_fraction:.12g}::seed={seed}"


def _load_stats_cache() -> Dict[str, Dict[str, list]]:
    if not os.path.exists(PATH_DATASET_STATS):
        return {}
    try:
        with open(PATH_DATASET_STATS, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _get_or_compute_stats_for_split(
    dataset_name: str,
    train_split: HFDataset,
    val_fraction: float,
    seed: int,
    data_dir: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    stats_cache = _load_stats_cache()
    cache_key = _stats_cache_key(dataset_name, val_fraction, seed)

    if cache_key in stats_cache:
        mean = tuple(stats_cache[cache_key]["mean"])
        std = tuple(stats_cache[cache_key]["std"])
        return mean, std

    print(
        f"Channel statistics not cached for '{dataset_name}' "
        f"(val_fraction={val_fraction}, seed={seed}). "
        "Computing from the post-split training partition only..."
    )
    raw_dataset = HuggingFaceDataset(
        dataset_name,
        split=train_split,
        transform=transforms.ToTensor(),
        data_dir=data_dir,
    )
    raw_loader = DataLoader(raw_dataset, batch_size=256, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for images, _ in tqdm(raw_loader, desc=f"Calculating Stats [{dataset_name}]"):
        b, _, h, w = images.shape
        num_pixels += b * h * w
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_sum_sq / num_pixels) - (mean ** 2))

    mean_list = [round(float(value), 4) for value in mean]
    std_list = [round(float(value), 4) for value in std]

    stats_cache[cache_key] = {"mean": mean_list, "std": std_list}
    with open(PATH_DATASET_STATS, "w", encoding="utf-8") as file:
        json.dump(stats_cache, file, indent=4)

    print(
        f"Calculated & cached train-only stats for {dataset_name}: "
        f"Mean={mean_list}, Std={std_list}"
    )
    return tuple(mean_list), tuple(std_list)


def get_or_compute_stats(
    dataset_name: str,
    val_fraction: float = CONFIG["val_fraction"],
    seed: int = CONFIG["seed"],
    data_dir: str = DIR_DATA,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return normalization stats computed only from the post-split training partition."""
    stats_cache = _load_stats_cache()
    cache_key = _stats_cache_key(dataset_name, val_fraction, seed)
    if cache_key in stats_cache:
        return (
            tuple(stats_cache[cache_key]["mean"]),
            tuple(stats_cache[cache_key]["std"]),
        )

    splits = _split_training_data(dataset_name, data_dir, val_fraction, seed)
    return _get_or_compute_stats_for_split(
        dataset_name,
        splits["train"],
        val_fraction,
        seed,
        data_dir,
    )


def get_transforms(
    dataset_name: str,
    mean: Tuple[float, ...] | None = None,
    std: Tuple[float, ...] | None = None,
    val_fraction: float = CONFIG["val_fraction"],
    seed: int = CONFIG["seed"],
    data_dir: str = DIR_DATA,
):
    if mean is None or std is None:
        mean, std = get_or_compute_stats(
            dataset_name,
            val_fraction=val_fraction,
            seed=seed,
            data_dir=data_dir,
        )

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


def get_lejepa_transforms(
    dataset_name: str,
    mean: Tuple[float, ...] | None = None,
    std: Tuple[float, ...] | None = None,
    val_fraction: float = CONFIG["val_fraction"],
    seed: int = CONFIG["seed"],
    data_dir: str = DIR_DATA,
):
    if mean is None or std is None:
        mean, std = get_or_compute_stats(
            dataset_name,
            val_fraction=val_fraction,
            seed=seed,
            data_dir=data_dir,
        )

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

    splits = _split_training_data(dataset_name, data_dir, val_fraction, seed)
    mean, std = _get_or_compute_stats_for_split(
        dataset_name,
        splits["train"],
        val_fraction,
        seed,
        data_dir,
    )
    train_transform, eval_transform = get_transforms(
        dataset_name,
        mean=mean,
        std=std,
        val_fraction=val_fraction,
        seed=seed,
        data_dir=data_dir,
    )

    if paradigm == "std":
        train_dataset = HuggingFaceDataset(
            dataset_name,
            split=splits["train"],
            transform=train_transform,
            data_dir=data_dir,
        )
    else:
        global_transform, local_transform = get_lejepa_transforms(
            dataset_name,
            mean=mean,
            std=std,
            val_fraction=val_fraction,
            seed=seed,
            data_dir=data_dir,
        )
        train_dataset = MultiViewDataset(
            dataset_name,
            global_transform,
            local_transform,
            split=splits["train"],
            data_dir=data_dir,
        )

    # For LeJEPA this validation partition is used by the later linear probe.
    val_dataset = HuggingFaceDataset(
        dataset_name,
        split=splits["test"],
        transform=eval_transform,
        data_dir=data_dir,
    )

    is_cuda = torch.cuda.is_available()
    common_loader = dict(
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader,
    )

    if not include_test:
        return train_loader, val_loader

    test_dataset = HuggingFaceDataset(
        dataset_name,
        split="test",
        transform=eval_transform,
        data_dir=data_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader,
    )
    return train_loader, val_loader, test_loader


def get_balanced_test_indices(
        dataset_name: str,
        samples_per_class: int = 50,
        seed: int = CONFIG["seed"],
        data_dir: str = DIR_DATA,
        interleave: bool = False,
) -> List[int]:
    """
    Directly queries the dataset label column to deterministically select
    an equal number of sample indices per class without reading image data.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    meta = DATASETS[dataset_name]
    label_key = meta["label_key"]
    num_classes = meta["num_classes"]

    # Loads metadata table from cache_dir; does NOT decode image bytes
    raw_test = load_dataset(meta["hf_path"], split="test", cache_dir=data_dir)
    labels = np.asarray(raw_test[label_key])

    # Pinned NumPy generator ensures cross-platform determinism
    rng = np.random.default_rng(seed)
    per_class_indices: list[list[int]] = []

    for c in range(num_classes):
        cls_indices = np.where(labels == c)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Class {c} has only {len(cls_indices)} samples, "
                f"but {samples_per_class} were requested."
            )
        # Deterministically permute and take the first N samples
        perm = rng.permutation(cls_indices)
        per_class_indices.append(perm[:samples_per_class].tolist())

    if interleave:
        # Round-robin: [c0_0, c1_0, ..., c9_0, c0_1, c1_1, ...]
        # Useful if evaluating mini-batches with uniform class distribution
        indices = [
            idx
            for sample_group in zip(*per_class_indices)
            for idx in sample_group
        ]
    else:
        # Grouped: [c0_0..c0_49, c1_0..c1_49, ...]
        indices = [idx for cls_list in per_class_indices for idx in cls_list]

    return indices


def get_balanced_test_loader(
        dataset_name: str,
        samples_per_class: int = 50,
        batch_size: int = CONFIG["batch_size"],
        num_workers: int = CONFIG["num_workers"],
        data_dir: str = DIR_DATA,
        val_fraction: float = CONFIG["val_fraction"],
        seed: int = CONFIG["seed"],
        interleave: bool = False,
) -> Tuple[DataLoader, List[int]]:
    """
    Creates a deterministic DataLoader containing exactly (samples_per_class * num_classes)
    images evaluated on identical validation/test transforms.
    """

    indices = get_balanced_test_indices(
        dataset_name=dataset_name,
        samples_per_class=samples_per_class,
        seed=seed,
        data_dir=data_dir,
        interleave=interleave,
    )

    _, eval_transform = get_transforms(
        dataset_name,
        val_fraction=val_fraction,
        seed=seed,
        data_dir=data_dir,
    )

    full_test_dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        split="test",
        transform=eval_transform,
        data_dir=data_dir,
    )

    subset = Subset(full_test_dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, indices

def _find_model_output_dir(
    method: str,
    dataset: str,
    arch: str,
    paradigm: str,
    epoch: int,
    output_dir: str = DIR_OUTPUT,
) -> Tuple[str, str]:
    """
    Locates the directory in output/{method}/ matching {dataset}_{arch}_{paradigm}_epoch_{epoch}.
    Returns:
        (full_dir_path, full_model_id)
    """
    method_dir = os.path.join(output_dir, method)
    if not os.path.isdir(method_dir):
        raise FileNotFoundError(f"Method directory '{method_dir}' does not exist.")

    # Primary targets to match
    exact_stems = [
        f"{dataset}_{arch}_{paradigm}_epoch_{epoch}",
        f"{dataset}_{arch}_{paradigm}_epoch_{epoch:04d}",
    ]

    for stem in exact_stems:
        candidate = os.path.join(method_dir, stem)
        if os.path.isdir(candidate):
            return candidate, stem

    # Match directories that start with or contain the epoch identifier
    prefix = f"{dataset}_{arch}_{paradigm}"
    candidates = []
    for d in os.listdir(method_dir):
        full_path = os.path.join(method_dir, d)
        if not os.path.isdir(full_path):
            continue
        if any(d.startswith(stem) for stem in exact_stems):
            candidates.append((d, full_path))
        elif d.startswith(prefix) and (
            f"_epoch_{epoch}_" in d
            or f"_epoch_{epoch:04d}_" in d
            or d.endswith(f"_epoch_{epoch}")
            or d.endswith(f"_epoch_{epoch:04d}")
        ):
            candidates.append((d, full_path))

    if not candidates:
        available = os.listdir(method_dir)
        raise FileNotFoundError(
            f"Could not find model output directory for method='{method}', "
            f"dataset='{dataset}', arch='{arch}', paradigm='{paradigm}', epoch={epoch} in '{method_dir}'.\n"
            f"Available folders: {available}"
        )

    candidates.sort(key=lambda x: x[0])
    model_id, full_path = candidates[0]
    return full_path, model_id


def _get_correct_sample_keys(model_dir: str) -> set[str]:
    """Scans the 'correct/' subfolder and extracts sample identifiers (c{x}_{y})."""
    correct_dir = os.path.join(model_dir, "correct")
    if not os.path.isdir(correct_dir):
        raise FileNotFoundError(f"Missing 'correct' subfolder in '{model_dir}'")

    key_pattern = re.compile(r"_(c\d+_\d+)\.pt$")
    keys = set()
    for fname in os.listdir(correct_dir):
        if not fname.endswith(".pt"):
            continue
        match = key_pattern.search(fname)
        if match:
            keys.add(match.group(1))
    return keys


def get_correct_heatmap_batches(
    dataset_name: str,
    arch: str,
    paradigm: str,
    epoch: int,
    other_epoch: Optional[int] = None,
    output_dir: str = DIR_OUTPUT,
    device: torch.device | str = "cpu",
    return_both_paradigms: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Retrieves the intersection of correctly classified heatmaps across both paradigms
    ('std' and 'lejepa') and returns batched [B, C, H, W] tensors for XAI and PCA.

    Args:
        dataset_name: 'cifar10', 'cifar100', etc.
        arch: 'cnn' or 'vit'.
        paradigm: 'std' (supervised) or 'lejepa'.
        epoch: Epoch index for the requested paradigm.
        other_epoch: Epoch index for the counterpart paradigm (defaults to `epoch`).
        output_dir: Root directory of outputs (defaults to DIR_OUTPUT).
        device: Torch device or string for the loaded tensors.
        return_both_paradigms: If True, returns ((xai_req, pca_req), (xai_other, pca_other)).

    Returns:
        batch_xai [B, C, H, W], batch_pca [B, C, H, W]
    """

    # Normalize aliases
    arch = "cnn" if arch in ("cnn", "resnet") else arch
    paradigm = "std" if paradigm in ("std", "supervised") else paradigm
    if arch not in ("cnn", "vit"):
        raise ValueError(f"Unknown architecture '{arch}'. Expected 'cnn' or 'vit'.")
    if paradigm not in ("std", "lejepa"):
        raise ValueError(f"Unknown paradigm '{paradigm}'. Expected 'std' or 'lejepa'.")

    other_paradigm = "lejepa" if paradigm == "std" else "std"
    other_epoch = epoch if other_epoch is None else other_epoch
    xai_method = "gradcam" if arch == "cnn" else "gmar"

    # 1. Locate directories for the requested paradigm
    dir_xai_req, id_xai_req = _find_model_output_dir(
        xai_method, dataset_name, arch, paradigm, epoch, output_dir
    )
    dir_pca_req, id_pca_req = _find_model_output_dir(
        "pca", dataset_name, arch, paradigm, epoch, output_dir
    )

    # 2. Locate directories for the counterpart paradigm
    dir_xai_other, id_xai_other = _find_model_output_dir(
        xai_method, dataset_name, arch, other_paradigm, other_epoch, output_dir
    )
    dir_pca_other, id_pca_other = _find_model_output_dir(
        "pca", dataset_name, arch, other_paradigm, other_epoch, output_dir
    )

    # 3. Retrieve sample keys correctly predicted by BOTH models (Option B)
    keys_xai_req = _get_correct_sample_keys(dir_xai_req)
    keys_pca_req = _get_correct_sample_keys(dir_pca_req)
    keys_xai_other = _get_correct_sample_keys(dir_xai_other)
    keys_pca_other = _get_correct_sample_keys(dir_pca_other)

    common_keys = keys_xai_req & keys_pca_req & keys_xai_other & keys_pca_other
    if not common_keys:
        raise RuntimeError(
            f"No intersecting correct heatmaps found between {paradigm} (epoch {epoch}) "
            f"and {other_paradigm} (epoch {other_epoch}) for {dataset_name} {arch}."
        )

    # Numerically deterministic order: c0_0, c0_1, ..., c1_0, ...
    def _parse_key(key: str) -> Tuple[int, int]:
        c_part, s_part = key[1:].split("_")
        return int(c_part), int(s_part)

    sorted_keys = sorted(common_keys, key=_parse_key)

    # 4. Helper to load and batch tensors from [H, W, C] to [B, C, H, W]
    def _load_batch(dir_path: str, method: str, model_id: str) -> torch.Tensor:
        correct_folder = os.path.join(dir_path, "correct")
        tensors = []
        for k in sorted_keys:
            filepath = os.path.join(correct_folder, f"{method}_{model_id}_{k}.pt")
            t = torch.load(filepath, map_location=device)
            # Permute [H, W, C] -> [C, H, W]
            if t.ndim == 3 and t.shape[-1] in (4, 6):
                t = t.permute(2, 0, 1)
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    batch_xai_req = _load_batch(dir_xai_req, xai_method, id_xai_req)
    batch_pca_req = _load_batch(dir_pca_req, "pca", id_pca_req)

    if return_both_paradigms:
        batch_xai_other = _load_batch(dir_xai_other, xai_method, id_xai_other)
        batch_pca_other = _load_batch(dir_pca_other, "pca", id_pca_other)
        return (batch_xai_req, batch_pca_req), (batch_xai_other, batch_pca_other)

    return batch_xai_req, batch_pca_req
