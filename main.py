import argparse
import os

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.evaluation import evaluate_model, run_GMAR_pipeline, run_gradcam_pipeline
from src.globals import CONFIG, DATASETS, DEVICE, set_seed
from src.network import LinearProbeModel, build_model
from src.train import train_lejepa, train_supervised
from src.utils import (
    build_relative_accuracy_comparison,
    load_probe_summary,
    probe_all_checkpoints,
    run_pca_for_all_checkpoints,
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
            "probe",
            "compare_relative",
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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "test_cuda":
        test_cuda()
        return

    if args.mode == "compare_relative":
        if not (args.dataset and args.arch):
            raise ValueError("Mode 'compare_relative' requires dataset and arch")
        build_relative_accuracy_comparison(args.dataset, args.arch)
        return

    if not (args.dataset and args.arch and args.paradigm):
        raise ValueError(f"Mode '{args.mode}' requires dataset, arch and paradigm")

    if args.mode == "test_config":
        test_config(args.dataset, args.arch, args.paradigm)
        return

    if args.mode == "test_pipeline":
        test_pipeline(args.dataset, args.arch, args.paradigm)
        return

    if args.mode == "eval" and args.paradigm == "lejepa":
        raise ValueError(
            "LeJEPA models have no standalone supervised head to evaluate. "
            "Use the stored linear-probe validation/test accuracy instead."
        )

    set_seed(CONFIG["seed"])
    device = torch.device(args.device)

    if args.mode == "probe":
        probe_all_checkpoints(
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
            resume=args.resume,
        )
        return

    if args.mode == "train":
        model = build_model(
            args.arch,
            args.dataset,
            args.paradigm,
            num_slices=args.sigreg_slices,
            t_max=args.sigreg_tmax,
            n_points=args.sigreg_points,
            lamb=args.lejepa_lambda,
        ).to(device)
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
                "[Postprocess skipped] Backbone checkpoints were left untouched. "
                "Run mode 'probe' later to train/resume one probe for every periodic checkpoint."
            )
            return

        probe_all_checkpoints(
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
            resume=args.resume,
        )
        return

    if args.mode == "pca":
        summary = load_probe_summary(args.dataset, args.arch, args.paradigm)
        run_pca_for_all_checkpoints(
            summary,
            args.batch_size,
            device,
            args.pca_samples,
            val_fraction=args.val_fraction,
        )
        return

    if args.mode == "eval":
        summary = load_probe_summary(args.dataset, args.arch, args.paradigm)
        checkpoint_path = summary["best_checkpoint_path"]
        model = build_model(
            args.arch,
            args.dataset,
            args.paradigm,
            num_slices=args.sigreg_slices,
            t_max=args.sigreg_tmax,
            n_points=args.sigreg_points,
            lamb=args.lejepa_lambda,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        _, _, test_loader = get_dataloaders(
            args.dataset,
            args.batch_size,
            paradigm="std",
            val_fraction=args.val_fraction,
            include_test=True,
        )
        print(f"Best checkpoint by linear-probe validation accuracy: {checkpoint_path}")
        print(f"Stored probe val accuracy:  {float(summary['best_val_acc']):.2f}%")
        print(f"Stored probe test accuracy: {float(summary['best_test_acc']):.2f}%")
        print("Original supervised head test performance:")
        evaluate_model(model, test_loader, device, verbose=True)
        return

    if args.mode in ("gradcam", "GMAR"):
        if args.mode == "gradcam" and args.arch != "cnn":
            raise ValueError("Grad-CAM requires arch='cnn'.")
        if args.mode == "GMAR" and args.arch != "vit":
            raise ValueError("GMAR requires arch='vit'.")

        summary = load_probe_summary(args.dataset, args.arch, args.paradigm)
        _, _, test_loader = get_dataloaders(
            args.dataset,
            args.batch_size,
            paradigm="std",
            val_fraction=args.val_fraction,
            include_test=True,
        )

        for record in summary["probe_results"]:
            if not os.path.exists(record["probe_path"]):
                raise FileNotFoundError(
                    f"Probe checkpoint missing for backbone epoch {record['epoch']}: "
                    f"'{record['probe_path']}'. Run mode 'probe' first."
                )

            model = build_model(
                args.arch,
                args.dataset,
                args.paradigm,
                num_slices=args.sigreg_slices,
                t_max=args.sigreg_tmax,
                n_points=args.sigreg_points,
                lamb=args.lejepa_lambda,
            ).to(device)
            checkpoint = torch.load(record["checkpoint_path"], map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

            probe_checkpoint = torch.load(record["probe_path"], map_location="cpu")
            if not probe_checkpoint.get("completed", False):
                raise RuntimeError(
                    f"Probe for backbone epoch {record['epoch']} is incomplete. "
                    "Resume mode 'probe' before running interpretability analysis."
                )

            backbone = model.backbone if args.paradigm == "lejepa" else model
            head = nn.Linear(backbone.embed_dim, DATASETS[args.dataset]["num_classes"])
            head.load_state_dict(probe_checkpoint["head_state_dict"])
            probe_model = LinearProbeModel(backbone, head).to(device)

            relative = record.get("relative_accuracy")
            relative_tag = "na" if relative is None else f"{float(relative):06.2f}"
            output_name = f"epoch_{int(record['epoch']):04d}_relative_{relative_tag}"
            print(
                f"[{args.mode}] Epoch {int(record['epoch']):04d} | "
                f"Val Acc {float(probe_checkpoint['best_val_acc']):.2f}% | "
                f"Test Acc {float(probe_checkpoint['test_acc']):.2f}% | "
                f"Relative Acc {relative}"
            )

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
                    output_name=output_name,
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
                    output_name=output_name,
                )

            del probe_model, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return


if __name__ == "__main__":
    main()
