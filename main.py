import argparse
import json
import os

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.evaluation import evaluate_model, linear_probe, run_GMAR_pipeline, run_gradcam_pipeline
from src.globals import CONFIG, DATASETS, DEVICE, DIR_CHECKPOINTS, set_seed
from src.network import LinearProbeModel, build_model
from src.train import train_lejepa, train_supervised
from src.utils import (
    run_pca_for_checkpoint,
    run_pca_for_milestones,
    select_and_prune_milestones,
    test_config,
    test_cuda,
    test_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LeJEPA vs Supervised Interpretability Pipeline")
    parser.add_argument(
        "mode",
        choices=[
            "train",
            "select_milestones",
            "eval",
            "pca",
            "gradcam",
            "GMAR",
            "test_cuda",
            "test_config",
            "test_pipeline",
        ],
    )
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
    parser.add_argument("--skip_milestones", action="store_true")
    return parser.parse_args()


def _summary_path(args):
    return os.path.join(
        DIR_CHECKPOINTS,
        f"{args.dataset}_{args.arch}_{args.paradigm}_milestones.json",
    )


def _training_best_path(args):
    return os.path.join(
        DIR_CHECKPOINTS,
        f"{args.dataset}_{args.arch}_{args.paradigm}_best.pt",
    )


def _load_summary(args):
    path = _summary_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No milestone summary found at '{path}'. "
            "Run 'select_milestones' first, or use --skip_milestones for pca/gradcam/GMAR."
        )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _best_retained_checkpoint(summary):
    best_epoch = int(summary["best_epoch"])
    for path in summary["retained_checkpoints"]:
        stem = os.path.splitext(path)[0]
        if int(stem.rsplit("_", 1)[-1]) == best_epoch:
            return path
    raise FileNotFoundError(
        f"Retained checkpoint for best probe epoch {best_epoch} is missing."
    )


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


def _model_config(args):
    return {
        "num_slices": args.sigreg_slices,
        "t_max": args.sigreg_tmax,
        "n_points": args.sigreg_points,
        "lamb": args.lejepa_lambda,
    }


def _load_checkpoint_into_model(path: str, model: nn.Module):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _probe_backbone(model: nn.Module, paradigm: str) -> nn.Module:
    return model.backbone if paradigm == "lejepa" else model


def _ensure_linear_probe(
    checkpoint_path: str,
    checkpoint: dict,
    model: nn.Module,
    args,
    device: torch.device,
    probe_train,
    probe_val,
    probe_test,
):
    stored = checkpoint.get("linear_probe")
    if stored and stored.get("head_state_dict") is not None:
        return checkpoint

    backbone = _probe_backbone(model, args.paradigm)
    print(f"[Linear probe] No cached probe found for '{checkpoint_path}'. Training one probe now.")
    result = linear_probe(
        backbone,
        probe_train,
        probe_val,
        probe_test,
        num_classes=DATASETS[args.dataset]["num_classes"],
        device=device,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
    )
    checkpoint["linear_probe"] = {
        "best_val_acc": float(result["best_val_acc"]),
        "test_acc": float(result["test_acc"]),
        "best_probe_epoch": int(result["best_epoch"]),
        "head_state_dict": result["head_state_dict"],
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[Linear probe] Cached probe results in '{checkpoint_path}'.")
    return checkpoint


def _build_probe_classifier(model: nn.Module, checkpoint: dict, args, device: torch.device):
    backbone = _probe_backbone(model, args.paradigm)
    head = nn.Linear(backbone.embed_dim, DATASETS[args.dataset]["num_classes"])
    head.load_state_dict(checkpoint["linear_probe"]["head_state_dict"])
    return LinearProbeModel(backbone, head).to(device)


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

    if args.skip_milestones and args.mode not in ("pca", "gradcam", "GMAR"):
        raise ValueError("--skip_milestones is only valid with pca, gradcam, or GMAR mode.")

    if args.mode == "eval" and args.paradigm == "lejepa":
        raise ValueError(
            "LeJEPA models have no standalone validation method. "
            "Use linear-probe validation/test accuracy to judge representation quality."
        )

    set_seed(CONFIG["seed"])
    device = torch.device(args.device)

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
                "Run mode 'select_milestones' later if milestone analysis is wanted."
            )
            return

        _select_milestones(args, device)
        return

    if args.mode == "pca":
        if args.skip_milestones:
            checkpoint_path = _training_best_path(args)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Training-time best checkpoint not found at '{checkpoint_path}'. Train first."
                )
            print(f"[PCA] Using training-time best checkpoint: {checkpoint_path}")
            run_pca_for_checkpoint(
                checkpoint_path,
                args.dataset,
                args.arch,
                args.paradigm,
                args.batch_size,
                device,
                num_samples=args.pca_samples,
                val_fraction=args.val_fraction,
                model_config=_model_config(args),
            )
        else:
            summary = _load_summary(args)
            run_pca_for_milestones(
                summary,
                args.batch_size,
                device,
                args.pca_samples,
                val_fraction=args.val_fraction,
            )
        return

    if args.mode == "eval":
        summary = _load_summary(args)
        checkpoint_path = _best_retained_checkpoint(summary)
        checkpoint = _load_checkpoint_into_model(checkpoint_path, model)
        _, _, test_loader = get_dataloaders(
            args.dataset,
            args.batch_size,
            paradigm="std",
            val_fraction=args.val_fraction,
            include_test=True,
        )
        print(f"Best checkpoint selected by linear-probe validation accuracy: {checkpoint_path}")
        if "linear_probe" in checkpoint:
            print(f"Stored probe val accuracy:  {checkpoint['linear_probe']['best_val_acc']:.2f}%")
            print(f"Stored probe test accuracy: {checkpoint['linear_probe']['test_acc']:.2f}%")
        print("Original supervised head test performance:")
        evaluate_model(model, test_loader, device, verbose=True)
        return

    if args.mode in ("gradcam", "GMAR"):
        if args.mode == "gradcam" and args.arch != "cnn":
            raise ValueError("Grad-CAM requires arch='cnn'.")
        if args.mode == "GMAR" and args.arch != "vit":
            raise ValueError("GMAR requires arch='vit'.")

        if args.skip_milestones:
            checkpoint_path = _training_best_path(args)
            checkpoint_source = "training-time best checkpoint"
        else:
            summary = _load_summary(args)
            checkpoint_path = _best_retained_checkpoint(summary)
            checkpoint_source = "best milestone checkpoint by probe validation accuracy"

        checkpoint = _load_checkpoint_into_model(checkpoint_path, model)
        probe_train, probe_val, test_loader = get_dataloaders(
            args.dataset,
            args.batch_size,
            paradigm="std",
            val_fraction=args.val_fraction,
            include_test=True,
        )
        checkpoint = _ensure_linear_probe(
            checkpoint_path,
            checkpoint,
            model,
            args,
            device,
            probe_train,
            probe_val,
            test_loader,
        )
        probe_model = _build_probe_classifier(model, checkpoint, args, device)

        print(f"Using {checkpoint_source}: {checkpoint_path}")
        print(f"Linear-probe val accuracy:  {checkpoint['linear_probe']['best_val_acc']:.2f}%")
        print(f"Linear-probe test accuracy: {checkpoint['linear_probe']['test_acc']:.2f}%")

        if args.mode == "gradcam":
            run_gradcam_pipeline(
                probe_model,
                test_loader,
                args.dataset,
                args.arch,
                args.paradigm,
                device,
                num_samples=8,
                val_fraction=args.val_fraction,
            )
        else:
            run_GMAR_pipeline(
                probe_model,
                test_loader,
                args.dataset,
                args.arch,
                args.paradigm,
                device,
                num_samples=8,
                val_fraction=args.val_fraction,
            )
        return


if __name__ == "__main__":
    main()
