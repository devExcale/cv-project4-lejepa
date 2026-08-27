import argparse
import os

import torch

from src.data import get_dataloaders
from src.evaluation import evaluate_model, evaluate_lejepa, pca_mask, pca_rgb
from src.globals import CONFIG, DEVICE, set_seed, DATASETS
from src.network import build_model
from src.train import train_supervised, train_lejepa
from src.utils import test_cuda, test_config, test_pipeline, get_checkpoint_path


def parse_args():
	parser = argparse.ArgumentParser(description="LeJEPA vs Supervised Interpretability Pipeline")
	parser.add_argument("mode", choices=["train", "eval", "pca", "gradcam", "test_cuda", "test_config", "test_pipeline"])
	parser.add_argument("-d", "--dataset", choices=list(DATASETS.keys()))
	parser.add_argument("-a", "--arch", choices=["cnn", "vit"])
	parser.add_argument("-p", "--paradigm", choices=["std", "lejepa"])
	parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
	parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--device", default=str(DEVICE))
	parser.add_argument("--sigreg_slices", type=int, default=CONFIG["sigreg_slices"])
	parser.add_argument("--sigreg_tmax", type=float, default=CONFIG["sigreg_tmax"])
	parser.add_argument("--sigreg_points", type=int, default=CONFIG["sigreg_points"])
	parser.add_argument("--lejepa_lambda", type=float, default=CONFIG["lejepa_lambda"])
	parser.add_argument("--layer", type=int, default=-1, help="Layer index for PCA (default: last)")
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
	train_loader, val_loader = get_dataloaders(args.dataset, args.batch_size, paradigm=args.paradigm)
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
			evaluate_lejepa(model, val_loader, device)
		else:
			evaluate_model(model, val_loader, device, verbose=True)
		return

	if args.mode == "pca":
		images, _ = next(iter(val_loader))
		images = images.to(device)
		model.eval()
		with torch.no_grad():
			layers = model.forward_features(images)
		feature_map = layers[args.layer]
		mask = pca_mask(feature_map)
		rgb = pca_rgb(feature_map)
		print(f"Selected layer shape: {tuple(feature_map.shape)}")
		print(f"PCA mask shape: {tuple(mask.shape)}")
		print(f"PCA RGB shape:  {tuple(rgb.shape)}")
		return

	if args.mode == "gradcam":
		print("# TODO: Grad-CAM implementation")


if __name__ == "__main__":
	main()
