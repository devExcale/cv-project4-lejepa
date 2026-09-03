import argparse
import json
import os

import torch

from src.data import get_dataloaders
from src.evaluation import evaluate_lejepa, evaluate_model, run_gradcam_pipeline
from src.globals import CONFIG, DATASETS, DEVICE, DIR_CHECKPOINTS, set_seed
from src.network import build_model
from src.train import train_lejepa, train_supervised
from src.utils import (
	run_pca_for_milestones,
	select_and_prune_milestones,
	test_config,
	test_cuda,
	test_pipeline,
)


def parse_args():
	parser = argparse.ArgumentParser(description="LeJEPA vs Supervised Interpretability Pipeline")
	parser.add_argument("mode", choices=["train", "select_milestones", "eval", "pca", "gradcam", "test_cuda", "test_config", "test_pipeline"])
	parser.add_argument("-d", "--dataset", choices=list(DATASETS.keys()))
	parser.add_argument("-a", "--arch", choices=["cnn", "vit"])
	parser.add_argument("-p", "--paradigm", choices=["std", "lejepa"])
	parser.add_argument("-e", "--epochs", type=int, default=CONFIG["epochs"])
	parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--device", default=str(DEVICE))
	parser.add_argument("--val_fraction", type=float, default=CONFIG["val_fraction"])
	parser.add_argument("--checkpoint_interval", type=int, default=CONFIG["checkpoint_interval"])
	parser.add_argument("--probe_epochs", type=int, default=CONFIG["probe_epochs"])
	parser.add_argument("--probe_lr", type=float, default=CONFIG["probe_lr"])
	parser.add_argument("--pca_samples", type=int, default=CONFIG["pca_samples"])
	parser.add_argument("--sigreg_slices", type=int, default=CONFIG["sigreg_slices"])
	parser.add_argument("--sigreg_tmax", type=float, default=CONFIG["sigreg_tmax"])
	parser.add_argument("--sigreg_points", type=int, default=CONFIG["sigreg_points"])
	parser.add_argument("--lejepa_lambda", type=float, default=CONFIG["lejepa_lambda"])
	parser.add_argument("-r", "--resume", action="store_true")
	parser.add_argument("--skip_postprocess", action="store_true")
	return parser.parse_args()


def _summary_path(args):
	return os.path.join(DIR_CHECKPOINTS, f"{args.dataset}_{args.arch}_{args.paradigm}_milestones.json")


def _load_summary(args):
	path = _summary_path(args)
	if not os.path.exists(path):
		raise FileNotFoundError(
			f"No milestone summary found at '{path}'. "
			"Run 'select_milestones' first."
		)
	with open(path, "r") as file:
		return json.load(file)


def _best_retained_checkpoint(summary):
	best_epoch = int(summary["best_epoch"])
	for path in summary["retained_checkpoints"]:
		stem = os.path.splitext(path)[0]
		if int(stem.rsplit("_", 1)[-1]) == best_epoch:
			return path
	raise FileNotFoundError(f"Retained checkpoint for best probe epoch {best_epoch} is missing.")


def _select_milestones(args, device):
	return select_and_prune_milestones(
		args.dataset,
		args.arch,
		args.paradigm,
		batch_size=args.batch_size,
		device=device,
		val_fraction=args.val_fraction,
		probe_epochs=args.probe_epochs,
		probe_lr=args.probe_lr,
		num_slices=args.sigreg_slices,
		t_max=args.sigreg_tmax,
		n_points=args.sigreg_points,
		lamb=args.lejepa_lambda,
	)


def main():
	args = parse_args()

	if args.mode == "test_cuda":
		test_cuda()
		return

	if not (args.dataset and args.arch and args.paradigm):
		raise ValueError(f"Mode '{args.mode}' requires dataset, arch and paradigm")

	if args.mode == "test_config":
		test_config(args.dataset, args.arch, args.paradigm)
		return

	if args.mode == "test_pipeline":
		test_pipeline(args.dataset, args.arch, args.paradigm)
		return

	set_seed(CONFIG["seed"])
	device = torch.device(args.device)

	# Milestone selection is deliberately available as a standalone operation.
	# It loads the periodic checkpoint files already present on disk.
	if args.mode == "select_milestones":
		_select_milestones(args, device)
		return

	model = build_model(
		args.arch,
		args.dataset,
		args.paradigm,
		num_slices=args.sigreg_slices,
		t_max=args.sigreg_tmax,
		n_points=args.sigreg_points,
		lamb=args.lejepa_lambda,
	).to(device)

	if args.mode == "train":
		train_loader, val_loader = get_dataloaders(
			args.dataset,
			args.batch_size,
			paradigm=args.paradigm,
			val_fraction=args.val_fraction,
			include_test=False,
		)

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
				checkpoint_interval=args.checkpoint_interval,
			)
		else:
			train_supervised(
				model,
				train_loader,
				val_loader,
				args.dataset,
				args.arch,
				"std",
				epochs=args.epochs,
				lr=args.lr or CONFIG["lr"],
				device=device,
				resume=args.resume,
				checkpoint_interval=args.checkpoint_interval,
			)

		if args.skip_postprocess:
			print(
				"[Postprocess skipped] Periodic checkpoints were left untouched. "
				"Run mode 'select_milestones' later."
			)
			return

		summary = _select_milestones(args, device)
		return

	summary = _load_summary(args)

	if args.mode == "pca":
		run_pca_for_milestones(
			summary,
			args.batch_size,
			device,
			args.pca_samples,
			val_fraction=args.val_fraction,
		)
		return

	if args.mode in ("eval", "gradcam"):
		checkpoint_path = _best_retained_checkpoint(summary)
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])

		_, _, test_loader = get_dataloaders(
			args.dataset,
			args.batch_size,
			paradigm="std",
			val_fraction=args.val_fraction,
			include_test=True,
		)

		print(f"Best checkpoint selected by linear-probe validation accuracy: {checkpoint_path}")
		print(f"Stored probe val accuracy:  {checkpoint['linear_probe']['best_val_acc']:.2f}%")
		print(f"Stored probe test accuracy: {checkpoint['linear_probe']['test_acc']:.2f}%")

		if args.mode == "gradcam":
			if args.arch != "cnn" or args.paradigm != "std":
				raise ValueError("Current Grad-CAM pipeline supports the standard CNN classifier only.")
			run_gradcam_pipeline(model, test_loader, args.dataset, args.arch, args.paradigm, device)
			return

		if args.paradigm == "std":
			print("Original supervised head test performance (separate from milestone-selection probe):")
			evaluate_model(model, test_loader, device, verbose=True)
		else:
			evaluate_lejepa(model, test_loader, device)
		return


if __name__ == "__main__":
	main()
