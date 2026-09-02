import argparse
import json
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import get_dataloaders
from src.evaluation import evaluate_model, linear_probe
from src.globals import CONFIG, DEVICE, set_seed, DATASETS
from src.network import build_model
from src.train import train_supervised, train_lejepa
from src.utils import save_pca, test_cuda, test_config, test_pipeline, get_checkpoint_path


def parse_args():
	parser = argparse.ArgumentParser(description="LeJEPA vs Supervised Interpretability Pipeline")
	parser.add_argument("mode", choices=["train", "eval", "pca", "linear_probe", "gradcam", "test_cuda", "test_config", "test_pipeline"])
	parser.add_argument("-d", "--dataset", choices=list(DATASETS.keys()))
	parser.add_argument("-a", "--arch", choices=["cnn", "vit"])
	parser.add_argument("-p", "--paradigm", choices=["std", "lejepa"])
	parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
	parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--device", default=str(DEVICE))
	parser.add_argument("--val_fraction", type=float, default=0.1)
	parser.add_argument("--sigreg_slices", type=int, default=CONFIG["sigreg_slices"])
	parser.add_argument("--sigreg_tmax", type=float, default=CONFIG["sigreg_tmax"])
	parser.add_argument("--sigreg_points", type=int, default=CONFIG["sigreg_points"])
	parser.add_argument("--lejepa_lambda", type=float, default=CONFIG["lejepa_lambda"])
	return parser.parse_args()


def main():
	args = parse_args()
	if args.mode == "test_cuda":
		test_cuda(); return
	if not (args.dataset and args.arch and args.paradigm):
		raise ValueError(f"Mode '{args.mode}' requires dataset, arch and paradigm")
	if args.mode == "test_config":
		test_config(args.dataset, args.arch, args.paradigm); return
	if args.mode == "test_pipeline":
		test_pipeline(args.dataset, args.arch, args.paradigm); return

	set_seed(CONFIG["seed"])
	device = torch.device(args.device)
	loaders = get_dataloaders(args.dataset, args.batch_size, paradigm=args.paradigm, val_fraction=args.val_fraction, include_test=args.mode != "train")
	if args.mode == "train":
		train_loader, val_loader = loaders
		test_loader = None
	else:
		train_loader, val_loader, test_loader = loaders
	model = build_model(
		args.arch, args.dataset, args.paradigm,
		num_slices=args.sigreg_slices,
		t_max=args.sigreg_tmax,
		n_points=args.sigreg_points,
		lamb=args.lejepa_lambda,
	).to(device)
	ckpt_path = get_checkpoint_path(args.dataset, args.arch, args.paradigm)

	if args.mode == "train":
		if args.paradigm == "lejepa":
			train_lejepa(model, train_loader, args.dataset, args.arch, epochs=args.epochs, lr=args.lr or 1e-3, device=device)
		else:
			train_supervised(model, train_loader, val_loader, args.dataset, args.arch, args.paradigm, epochs=args.epochs, lr=args.lr or CONFIG["lr"], device=device)
		return

	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'. Run training first.")
	checkpoint = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])

	if args.mode == "eval":
		if args.paradigm == "lejepa":
			raise ValueError("LeJEPA has no trained classification head. Use 'linear_probe' for optional downstream accuracy.")
		print("[Official test split]")
		evaluate_model(model, test_loader, device, verbose=True)
		return
	
	if args.mode == "linear_probe":
		probe_train, probe_val, probe_test = get_dataloaders(args.dataset, args.batch_size, paradigm="std", include_test=True)
		result = linear_probe(model, probe_train, probe_val, probe_test, num_classes=DATASETS[args.dataset]["num_classes"], device=device, repochs=args.epochs or 50, lr=args.lr or 0.1)

		probe_dir = os.path.join(args.output_dir, "linear_probe", args.dataset, f"{args.arch}_{args.paradigm}")
		os.makedirs(probe_dir, exist_ok=True)
		torch.save(result.pop("head_state_dict"), os.path.join(probe_dir, "linear_head.pt"))

		result.update({"dataset": args.dataset, "architecture": args.arch, "paradigm": args.paradigm, "split_seed": CONFIG["seed"], "val_fraction": args.val_fraction, "probe_epochs": args.epochs or CONFIG["probe_epochs"], "probe_lr": args.lr or CONFIG["probe_lr"]})

		print(
			f"Linear probe | Best Val Acc: {result['best_val_acc']:.2f}% | "
			f"Test Acc: {result['test_acc']:.2f}%"
		)
		print(f"Saved linear-probe results to '{probe_dir}'")
		return
	
	if args.mode == "pca":
		model.eval()
		num_saved = 0
		sample_offset = 0
		num_layers = None

		for images, labels in tqdm(test_loader, desc="[PCA: all test images]"):
			batch_size = images.size(0)
			sample_indices = list(range(sample_offset, sample_offset + batch_size))
			sample_offset += batch_size

			with torch.no_grad():
				features = model.forward_features(images.to(device, non_blocking=True))

			layer_indices = list(range(len(features)))
			num_layers = len(layer_indices)
			saved = save_pca(images, labels, features, layer_indices, sample_indices, args, checkpoint, ckpt_path)
			num_saved += len(saved)

		output_root = os.path.join(args.output_dir, "pca", args.dataset, f"{args.arch}_{args.paradigm}")
		print(
			f"Saved {num_saved} PCA result(s) for all {sample_offset} test images "
			f"across all {num_layers} feature layers to '{output_root}'."
		)
		return

	if args.mode == "gradcam":
		print("# TODO: Grad-CAM implementation")

if __name__ == "__main__":
	main()
