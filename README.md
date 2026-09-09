# Emerging Interpretability in LeJEPA vs. Supervised Learning

## Abstract

This project investigates the emergence of "zero-shot" semantic segmentation in Joint-Embedding Predictive
Architectures (JEPAs) compared to traditional supervised models. While supervised training is driven by categorical
cross-entropy labels, LeJEPA utilizes Sketched Isotropic Gaussian Regularization (SIGReg) and invariance constraints to
learn representations. This framework conducts a layer-by-layer comparative analysis evaluating whether semantic
structures emerging in LeJEPA's latent embedding space (via spatial PCA) align more naturally with internal attention
and gradients than in supervised counterparts.

## Environment & Requirements

The codebase requires **Python 3.10+** and is built on PyTorch.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory & Artifact Structure

All data, model checkpoints, and generated heatmaps adhere to the canonical structure managed in `src/globals.py`,
`src/train.py`, and `src/evaluation.py`:

```text
/                                              # Project root
├── data/
│   ├── dataset_stats.json                     # Cached dataset statistics (mean, std, class counts)
│   └── ...                                    # HuggingFace cached dataset files
├── checkpoints/
│   ├── {dataset}_{arch}_{paradigm}/           # e.g., cifar10_cnn_std, cifar10_vit_lejepa
│   │   ├── best/
│   │   │   ├── checkpoint_best.pt             # Best backbone checkpoint
│   │   │   └── probe_best.pt                  # Probe evaluation state for best checkpoint
│   │   ├── epoch_0010/
│   │   │   ├── checkpoint_0010.pt             # Periodic backbone checkpoint
│   │   │   └── probe_0010.pt                  # Trained frozen-backbone linear probe
│   │   ├── ...
│   │   └── probe_results.json                 # Validation/test/relative accuracy trajectory
│   └── {dataset}_{arch}_relative_accuracy_comparison.json  # Checkpoints matched by relative acc
└── output/
    ├── gradcam/
    │   └── {full_model_id}/                   # e.g., cifar10_cnn_std_epoch_0100_relative_100.00
    │       ├── correct/                       # gradcam_{model_id}_c{label}_{sample}.pt
    │       ├── missed/                        # gradcam_{model_id}_c{label}_{sample}.pt
    │       └── gradcam_{model_id}.pdf         # Visualization preview PDF (if --plot)
    ├── gmar/
    │   └── {full_model_id}/                   # e.g., cifar10_vit_lejepa_epoch_0100_relative_100.00
    │       ├── correct/                       # gmar_{model_id}_c{label}_{sample}.pt
    │       ├── missed/                        # gmar_{model_id}_c{label}_{sample}.pt
    │       └── gmar_{model_id}.pdf            # Visualization preview PDF (if --plot)
    └── pca/
        └── {full_model_id}/
            ├── correct/                       # pca_{model_id}_c{label}_{sample}.pt
            ├── missed/                        # pca_{model_id}_c{label}_{sample}.pt
            └── pca_{model_id}.pdf             # Visualization preview PDF (if --plot)

```

### Tensor Shapes & Specifications

* **Raw Heatmap Files (`.pt`):** Stored as shape $[H, W, C]$ where $H = W = 32$ for CIFAR-10.
* **Channels ($C$):** Represents model depth:
    * CNN (`CIFARResNet18`): $C = 4$ stages (`layer1` through `layer4`).
    * ViT (`VisionTransformer`): $C = 6$ encoder transformer blocks.
* **Batched Heatmaps:** Loaded and permuted to $[B, C, 32, 32]$ for vectorized evaluation.

## Execution Workflow

All routines are orchestrated via `main.py`.

### 1. Verification

Verify CUDA availability and pipeline data-flow before starting large runs:

```bash
# Check GPU recognition
python main.py test_cuda

# Validate hyperparameter configuration and class counts
python main.py test_config -d cifar10 -a cnn -p std

# Test forward passes, loss calculations, and feature extraction
python main.py test_pipeline -d cifar10 -a vit -p lejepa
```

### 2. Model Training

Train backbones across architectures (`cnn`, `vit`) and paradigms (`std`, `lejepa`). By default, periodic checkpoints
are saved every 10 epochs and linear probes are trained automatically post-training unless `--skip_postprocess` is
flagged.

* **Supervised ResNet18 (`cnn/std`):**

```bash
python main.py train -d cifar10 -a cnn -p std
```

* **Supervised Vision Transformer (`vit/std`):**

```bash
python main.py train -d cifar10 -a vit -p std
```

* **LeJEPA ResNet18 (`cnn/lejepa`):**

```bash
python main.py train -d cifar10 -a cnn -p lejepa
```

* **LeJEPA Vision Transformer (`vit/lejepa`):**

```bash
python main.py train -d cifar10 -a vit -p lejepa
```

(Add `-r` / `--resume` to resume training from the latest checkpoint).

### 3. Linear Probing Trajectory

To evaluate representation quality throughout training (especially for self-supervised LeJEPA representations), linear
probing trains a linear head on top of frozen backbone checkpoints across all epochs:

```bash
python main.py probe -d cifar10 -a cnn -p lejepa --probe_epochs 50
```

This writes `probe_results.json` containing validation and test accuracy, alongside relative accuracy normalized against
chance and peak accuracy.

### 4. Cross-Paradigm Relative Accuracy Alignment

To ensure comparison of models at comparable stages of representational maturity, match checkpoints between Supervised
and LeJEPA by nearest relative accuracy:

```bash
python main.py compare_relative -d cifar10 -a cnn
python main.py compare_relative -d cifar10 -a vit
```

This generates `checkpoints/{dataset}_{arch}_relative_accuracy_comparison.json`, pairing each epoch checkpoint with its
counterpart.
Relative accuracy rescales validation accuracy so chance performance is 0% and the best probed validation accuracy for that training trajectory is 100%, using 100 * (val_acc - chance_accuracy) / (accuracy_final - chance_accuracy).

### 5. Heatmap & Latent Representation Generation

Extract interpretability maps on a balanced test subset.

#### A. XAI Saliency Maps

* **Grad-CAM (for CNNs):**

```bash
# Save raw tensors [32, 32, 4] for SAS computation
python main.py gradcam -d cifar10 -a cnn -p std --eval-class-samples 50
python main.py gradcam -d cifar10 -a cnn -p lejepa --eval-class-samples 50

# Save preview PDF overlaying Grad-CAM and Guided Backpropagation
python main.py gradcam -d cifar10 -a cnn -p std --eval-class-samples 5 --plot
python main.py gradcam -d cifar10 -a cnn -p lejepa --eval-class-samples 5 --plot
```

* **GMAR (for ViTs):**

```bash
# Save raw tensors [32, 32, 6] for SAS computation
python main.py gmar -d cifar10 -a vit -p std --eval-class-samples 50
python main.py gmar -d cifar10 -a vit -p lejepa --eval-class-samples 50

# Save preview PDF showing attention rollout across blocks
python main.py gmar -d cifar10 -a vit -p std --eval-class-samples 5 --plot
python main.py gmar -d cifar10 -a vit -p lejepa --eval-class-samples 5 --plot
```

#### B. Spatial PCA Semantic Maps

Extract spatial PCA representations (projecting spatial feature maps across layers into semantic masks and pseudo-RGB
maps):

```bash
# Save raw PCA tensors [32, 32, C] for SAS computation
python main.py pca -d cifar10 -a cnn -p std --eval-class-samples 50
python main.py pca -d cifar10 -a cnn -p lejepa --eval-class-samples 50
python main.py pca -d cifar10 -a vit -p std --eval-class-samples 50
python main.py pca -d cifar10 -a vit -p lejepa --eval-class-samples 50

# Save preview PDF showing pseudo-RGB principal components
python main.py pca -d cifar10 -a cnn -p std --eval-class-samples 5 --plot
python main.py pca -d cifar10 -a cnn -p lejepa --eval-class-samples 5 --plot
python main.py pca -d cifar10 -a vit -p std --eval-class-samples 5 --plot
python main.py pca -d cifar10 -a vit -p lejepa --eval-class-samples 5 --plot
```

### 6. Computing Semantic Alignment Score (SAS)

The Semantic Alignment Score quantitatively measures the correlation/overlap between a model's own XAI saliency map and
its internal PCA semantic map.
When run, SAS loads the matching correct Grad-CAM (CNN) or GMAR (ViT) and PCA tensors and evaluates only sample IDs that are correctly classified by both the supervised and LeJEPA counterparts.
Use --epoch N and --other-epoch to select the checkpoint epoch whose saved XAI/PCA tensors SAS evaluates for both paradigms (e.g. --epoch 100 --other-epoch 100).


```python main.py sas -d cifar10 -a cnn -p std --epoch 100 --other-epoch 100
```

## CLI Options Reference

| Argument                | Type    | Default      | Description                                                                                  |
|-------------------------|---------|--------------|----------------------------------------------------------------------------------------------|
| `mode`                  | `str`   | *required*   | Action: `train`, `probe`, `compare_relative`, `eval`, `pca`, `gradcam`, `gmar`, `test_*`<br> |
| `-d`, `--dataset`       | `str`   | `cifar10`    | Dataset choice: `cifar10`, `cifar100`                                                        |
| `-a`, `--arch`          | `str`   | `cnn`        | Architecture: `cnn` (ResNet18) or `vit` (VisionTransformer)                                  |
| `-p`, `--paradigm`      | `str`   | `std`        | Training paradigm: `std` (supervised) or `lejepa` (LeJEPA)                                   |
| `-e`, `--epochs`        | `int`   | `100`        | Number of training epochs                                                                    |
| `--batch_size`          | `int`   | `64` / `128` | Batch size for train/eval                                                                    |
| `--lr`                  | `float` | `None`       | Learning rate (defaults to `0.1` for SGD on CNN, `1e-3` for AdamW on ViT)                    |
| `--checkpoint_interval` | `int`   | `10`         | Frequency (epochs) for saving evaluation checkpoints                                         |
| `--probe_epochs`        | `int`   | `50`         | Maximum epochs for training linear head                                                      |
| `--eval-class-samples`  | `int`   | `1`          | Number of samples per class evaluated during Grad-CAM / GMAR                                 |
| `--plot`                | `flag`  | `False`      | Renders and saves PDF visualizations alongside `.pt` tensors                                 |
| `-r`, `--resume`        | `flag`  | `False`      | Resumes existing run from checkpoint/probes if available                                     |
