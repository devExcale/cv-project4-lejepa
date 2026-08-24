import argparse
import os

import torch

from src.data import get_dataloaders
from src.evaluation import evaluate_model
from src.globals import CONFIG, DEVICE, set_seed, DATASETS
from src.network import build_model
from src.train import train_supervised
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
		choices=["train", "eval", "gradcam", "test_cuda", "test_config", "test_pipeline"],
		help="Execution mode"
	)
	parser.add_argument(
		"-d", "--dataset",
		type=str,
		choices=list(DATASETS.keys()),
		help="Target dataset (cifar10, cifar100)"
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
		help="Training paradigm (std, lejepa)"
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=CONFIG["epochs"],
		help="Number of training epochs"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=CONFIG["batch_size"],
		help="Batch size"
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=CONFIG["lr"],
		help="Base learning rate"
	)
	parser.add_argument(
		"--device",
		type=str,
		default=str(DEVICE),
		help="Target device (cpu, cuda)"
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
		raise ValueError(f"Mode '{args.mode}' requires explicit 'dataset', 'arch', and 'paradigm' arguments.")

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
	train_loader, val_loader = get_dataloaders(dataset_name=args.dataset, batch_size=args.batch_size)
	model = build_model(arch=args.arch, dataset=args.dataset, paradigm=args.paradigm).to(device)
	ckpt_path = get_checkpoint_path(dataset=args.dataset, arch=args.arch, paradigm=args.paradigm)

	# --- Mode: TRAIN --- #
	if args.mode == "train":
		train_supervised(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			dataset_name=args.dataset,
			arch=args.arch,
			paradigm=args.paradigm,
			epochs=args.epochs,
			lr=args.lr,
			device=device
		)
		return

	# --- Mode: EVAL --- #
	if args.mode == "eval":

		# Check if checkpoint exists
		if not os.path.exists(ckpt_path):
			raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'. Run training first.")

		# Load model checkpoint
		checkpoint = torch.load(ckpt_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])

		print(f"Loaded checkpoint from '{ckpt_path}' (Trained for {checkpoint.get('epoch', '?')} epochs)")

		# Evaluate model checkpoint on validation set
		evaluate_model(model=model, val_loader=val_loader, device=device, verbose=True)

		return

	# --- Mode: GRADCAM --- #
	if args.mode == "gradcam":
		# TODO: Grad-CAM implementation
		print("# TODO: Grad-CAM implementation")

		return

	return


if __name__ == "__main__":
	main()
