import argparse
import os

import torch

from src.data import get_dataloaders
from src.evaluation import evaluate_model, run_gradcam_pipeline, evaluate_lejepa, pca_mask, pca_rgb
from src.globals import CONFIG, DEVICE, set_seed, DATASETS
from src.network import build_model
from src.train import train_supervised, train_lejepa
from src.utils import test_cuda, test_config, test_pipeline, get_checkpoint_path


def parse_args():
	"""
	Parse command-line arguments for the LeJEPA vs Supervised Interpretability Pipeline.
	:return: Parsed arguments
	"""

	parser = argparse.ArgumentParser(description="LeJEPA vs Supervised Interpretability Pipeline")
	parser.add_argument(
		"mode",
		type=str,
		choices=["train", "eval", "pca", "gradcam", "test_cuda", "test_config", "test_pipeline"],
		help="Execution mode",
	)
	parser.add_argument(
		"-d", "--dataset",
		type=str,
		choices=list(DATASETS.keys()),
		help="Target dataset (e.g., cifar10, cifar100",
	)
	parser.add_argument(
		"-a", "--arch",
		type=str,
		choices=["cnn", "vit"],
		help="Model architecture (cnn, vit)"
	)
	parser.add_argument(
		"-p", "--paradigm",
		type=str,
		choices=["std", "lejepa"],
		help="Training paradigm (supervised or lejepa)",
	)
	parser.add_argument(
		"-e", "--epochs",
		type=int,
		default=CONFIG["epochs"],
		help="Number of training epochs",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=CONFIG["batch_size"],
		help="Batch size",
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=None,
		help="Base learning rate",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=str(DEVICE),
		help="Target device (cpu, cuda)",
	)
	parser.add_argument(
		"--sigreg_slices",
		type=int,
		default=CONFIG["sigreg_slices"]
	)
	parser.add_argument(
		"--sigreg_tmax",
		type=float,
		default=CONFIG["sigreg_tmax"]
	)
	parser.add_argument(
		"--sigreg_points",
		type=int,
		default=CONFIG["sigreg_points"]
	)
	parser.add_argument(
		"--lejepa_lambda",
		type=float,
		default=CONFIG["lejepa_lambda"]
	)
	parser.add_argument(
		"--layer",
		type=int,
		default=-1,
		help="Layer index for PCA (default: last)"
	)
	parser.add_argument(
		"-r", "--resume",
		action="store_true",
		help="Resume training from checkpoint (if available)"
	)
	return parser.parse_args()


def main():
	"""
	Main entry point for the LeJEPA vs Supervised Interpretability Project.
	:return: ``None``
	"""

	# Load provided arguments
	args = parse_args()

	# --- Mode: TEST_CUDA --- #
	if args.mode == "test_cuda":
		test_cuda()
		return

	# Other modes require explicit dataset, architecture, and paradigm parameters
	if not (args.dataset and args.arch and args.paradigm):
		raise ValueError(f"Mode '{args.mode}' requires dataset, arch and paradigm")

	# --- Mode: TEST_CONFIG --- #
	if args.mode == "test_config":
		test_config(dataset=args.dataset, arch=args.arch, paradigm=args.paradigm)
		return

	# --- Mode: TEST_PIPELINE --- #
	if args.mode == "test_pipeline":
		test_pipeline(dataset=args.dataset, arch=args.arch, paradigm=args.paradigm)
		return

	# Set deterministic seed
	set_seed(CONFIG["seed"])
	device = torch.device(args.device)

	# Load specified dataset and model
	train_loader, val_loader = get_dataloaders(args.dataset, args.batch_size, paradigm=args.paradigm)
	model = build_model(
		args.arch, args.dataset, args.paradigm,
		num_slices=args.sigreg_slices,
		t_max=args.sigreg_tmax,
		n_points=args.sigreg_points,
		lamb=args.lejepa_lambda,
	).to(device)
	ckpt_path = get_checkpoint_path(args.dataset, args.arch, args.paradigm)

	# --- Mode: TRAIN --- #
	if args.mode == "train":
		if args.paradigm == "lejepa":
			train_lejepa(
				model,
				train_loader,
				args.dataset,
				args.arch,
				epochs=args.epochs,
				lr=args.lr or 1e-3,
				device=device,
				resume=args.resume,
			)
		elif args.paradigm == "std":
			train_supervised(
				model,
				train_loader,
				val_loader,
				args.dataset,
				args.arch,
				args.paradigm,
				epochs=args.epochs,
				lr=args.lr or CONFIG["lr"],
				device=device,
				resume=args.resume,
			)
		else:
			raise ValueError(f"Unknown training paradigm '{args.paradigm}.'")
		return

	# Check if checkpoint exists
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'. Run training first.")

	# Load model checkpoint
	checkpoint = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])

	print(f"Loaded checkpoint from '{ckpt_path}'.")

	# --- Mode: EVAL --- #
	if args.mode == "eval":
		if args.paradigm == "lejepa":
			evaluate_lejepa(model, val_loader, device)
		elif args.paradigm == "std":
			evaluate_model(model, val_loader, device, verbose=True)
		else:
			raise ValueError(f"Unknown training paradigm '{args.paradigm}.'")
		return

	# --- Mode: PCA --- #
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

	# --- Mode: GRADCAM --- #
	if args.mode == "gradcam":

		run_gradcam_pipeline(
			model=model,
			val_loader=val_loader,
			dataset_name=args.dataset,
			arch=args.arch,
			paradigm=args.paradigm,
			device=device,
			num_samples=8
		)

	return


if __name__ == "__main__":
	main()
