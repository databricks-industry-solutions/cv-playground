# Distributed Deep Learning

> **Note:** This project is currently a work in progress (WIP) and is being actively updated.

## Overview

This project contains notebooks and scripts for distributed deep learning training on Databricks, focusing on PyTorch distributed training with hyperparameter optimization via Optuna and experiment tracking with MLflow.

## Contents

### `torch_distributedDL_classic/`

- **`01_prepare_dataset_v0.1.py`** / **`v0.2.py`** — Data preparation scripts for ImageNet datasets.
- **`imgnet_mbnetv2_TorchDistr_Optuna_mlflow_v0.1 (dataAug).ipynb`** — MobileNetV2 distributed training with Optuna hyperparameter tuning and data augmentation.
- **`imgnet_mbnetv2_TorchDistr_OptunaTPE_mlflow_v0.1 (dataAug).ipynb`** — Variant using Optuna's TPE sampler.
- **`imgnet_mbnetv2_TorchDistr_Optuna_mlflow_v0.2.3.1 (UCVols_chkpt).ipynb`** — Updated version with Unity Catalog Volumes checkpointing.
- **`test_resource_detect_v0.1.ipynb`** — Utility notebook for detecting available compute resources.
