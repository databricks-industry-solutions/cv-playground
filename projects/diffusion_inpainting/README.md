# Diffusion Inpainting — SD2 Fine-tuning Reference Code

> **Status: Work in Progress** — This code is under active development. Interfaces, configurations, and documentation may change.

## Overview

Reference notebooks for fine-tuning [Stable Diffusion 2 Inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) on Databricks, using the [Overlay Text Removal (OTR)](https://huggingface.co/datasets/cyberagent/OTR) dataset as an example.

**Use case:** Remove embedded text from images and reconstruct clean backgrounds using diffusion-based inpainting.

## Notebooks

### Data Pipeline
| Notebook | Description |
|---|---|
| `download_otr_v0.1.py` | Downloads OTR dataset (~80 GB) from HuggingFace to UC Volumes. Extracts paired images + generates masks. |
| `convert_otr_to_mds_v0.1.py` | Converts downloaded images to Mosaic Data Shards (MDS) for efficient streaming during training. Serverless compute. |
| `convert_otr_to_mds_classic.ipynb` | Same as above, optimized for classic compute (local NVMe staging). |

### Fine-tuning
| Notebook | Compute | Description |
|---|---|---|
| `finetune_sd2_inpainting_v0.1.5.py` | Serverless GPU (`@distributed`) | DDP training with MLflow logging, checkpointing, inference. Supports A10, H100. |
| `finetune_sd2_inpainting_v0.1.5_classic.py` | Classic GPU (`TorchDistributor`) | Same training pipeline on classic Spark clusters (e.g. g5.12xlarge 4xA10G). |

## Quick Start

1. **Configure widgets:** Update `Catalog`, `Schema`, `Volume` to your own Unity Catalog paths
2. **Download data:** Run `download_otr_v0.1.py` to stage the OTR dataset
3. **Convert to MDS:** Run `convert_otr_to_mds_v0.1.py` (or classic variant)
4. **Train:** Run either fine-tuning notebook with `Smoke Test = true` for quick validation

## Key Features

- **GPU auto-detection** — presets auto-configure from GPU type (batch size, accumulation, precision)
- **MLflow integration** — metrics, system metrics, checkpoints, sample images, per-rank logs
- **Checkpoint resume** — save/restore full training state for interrupted runs
- **Best checkpoint selection** — auto-saves on val_loss improvement
- **fp16 precision** — all GPU types use fp16 (bf16 causes numerical instability in diffusion training)

## Requirements

- Databricks Runtime ML 16.x LTS or AI Runtime
- GPU compute: A10G, A100, or H100
- Unity Catalog with a Volume for data storage
- HuggingFace token (for gated model access)

## Notes

- This is reference code — adapt paths, hyperparameters, and data pipeline to your use case
- The serverless notebook includes Private Preview features (`remote=True`) that may change
- For production, use Service Principal authentication instead of notebook context tokens
