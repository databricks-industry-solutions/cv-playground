# CV-Playground
**Computer Vision Applications on Databricks**


[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

## Overview

This repository is intended to be a _`"playground"`_ where we share **Applications of Computer Vision Solutions on Databricks**.

We currently have four projects:

1. [**Instance Segmentation**](projects/NuInsSeg/) in medical imaging — fine-tuning [YOLO11-seg](https://docs.ultralytics.com/tasks/segment/) on the [NuInsSeg dataset](https://github.com/masih4/NuInsSeg) for nuclei instance segmentation in H&E-stained histological images, using [Databricks AI Runtime](https://docs.databricks.com/aws/en/compute/serverless/gpu); Databricks' Serverless GPU compute.

2. [**Object Detection**](projects/ultralytics_databricks_examples/) on Databricks — end-to-end [YOLO11n](https://docs.ultralytics.com/models/yolo11/) training with [COCO128](https://www.kaggle.com/datasets/ultralytics/coco128), demonstrating both [**Databricks AI Runtime**](https://docs.databricks.com/aws/en/compute/serverless/gpu) and [**Classic**](https://docs.databricks.com/aws/en/compute/classic.html) compute across single and multi-GPU configurations.

3. [**Deep Learning Developer Experience**](projects/deep_learning_devex/) — a local developer experience for multi-node/multi-GPU training on Databricks with a CLIP model, using asset bundles, Python wheel tasks, shell scripts, and a patched TorchDistributor framework. Uses synthetic data for local testing and user-provided MDS datasets for distributed training. Focus is on functionality, developer experience, and cluster log export to Volumes.

4. [**Batch Inference with I/O**](projects/cv_batch_inference_with_io/) — six batch-inference patterns for scoring images with a [ViT](https://huggingface.co/docs/transformers/model_doc/vit) classifier, benchmarked head-to-head on [Databricks AI Runtime](https://docs.databricks.com/aws/en/compute/serverless/gpu) and Serverless Spark. Compares *compute* strategies (single-node GPU vs. distributed CPU vs. [Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/) endpoint via `ai_query`) against *I/O* strategies (serial vs. thread-parallel vs. distributed writes), with cost and throughput numbers at 4K/5K/118K images. All six are parameterized by job params, so you can swap in your own UC model + images.

Other CV applications will be added when available.

## Projects

| **Project** | **Model** | **Use Case** | **Compute** | **Status** |
|-------------|-----------|-------------|-------------|------------|
| [`NuInsSeg`](projects/NuInsSeg/) | YOLO11-seg | Instance Segmentation of Cell Nuclei (Medical Imaging) | Serverless GPU | Available |
| `NuInsSeg` — Classic | YOLO11-seg | Custom Data to YOLO Format + Fine-Tuning (Medical Imaging) | Classic Compute | Coming Soon |
| [`ultralytics_databricks_examples`](projects/ultralytics_databricks_examples/) | YOLO11n-detect | Object Detection (COCO128) | AI Runtime + Classic | Available |
| [`deep_learning_devex`](projects/deep_learning_devex/) | CLIP | Multi-Node Multi-GPU Training DevEx | Classic Compute | Available |
| [`cv_batch_inference_with_io`](projects/cv_batch_inference_with_io/) | ViT-Base/16 | Image Classification Batch Inference + I/O Pattern Benchmark (P1–P6) | AI Runtime + Serverless Spark + Model Serving | Available |

All projects leverage [MLflow](https://docs.databricks.com/aws/en/mlflow/) for experiment tracking and [Unity Catalog Volumes](https://docs.databricks.com/data-governance/unity-catalog/index.html) for governed data storage.

---

## Getting Started

Clone this repository to your Databricks Workspace.
You will find the example applications in each subfolder within the `/projects` folder.
Follow the `README.md` within each of the `/projects/{application_example_folder}/README.md` to get started.

---

## Contributing

**We welcome contributions!**

Please refer to [REPO_structure.md](REPO_structure.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for more details and guidance.

---

## How to get help

Databricks support doesn't cover this content. For questions or bugs, please open a GitHub issue and the team will help on a best effort basis.

---

## Licenses

&copy; 2025 Databricks, Inc. All rights reserved.
The source in this project is provided subject to the Databricks License [https://databricks.com/db-license-source].
All included or referenced third party libraries are subject to the licenses set forth below.

| Package | License | Copyright |
|---------|---------|-----------|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Ultralytics |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | PyTorch Contributors |
| [MLflow](https://mlflow.org/) | Apache-2.0 | MLflow Project |
| [OpenCV](https://opencv.org/) | Apache-2.0 | OpenCV Contributors |
| [NumPy](https://numpy.org/) | BSD-3-Clause | NumPy Developers |
| [Pandas](https://pandas.pydata.org/) | BSD-3-Clause | Pandas Development Team |
| [Matplotlib](https://matplotlib.org/) | PSF-based | Matplotlib Development Team |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause | scikit-learn Developers |
| [NuInsSeg](https://github.com/masih4/NuInsSeg) | Creative Commons Attribution 4.0 International | Authors — https://doi.org/10.1038/s41597-024-03117-2 |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Hugging Face, Inc. |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | Soumith Chintala and contributors |
| [Ray](https://github.com/ray-project/ray) | Apache-2.0 | The Ray Authors / Anyscale, Inc. |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | Jeffrey A. Clark and contributors |
| [COCO 2017](https://cocodataset.org/) | Annotations: Creative Commons Attribution 4.0; images subject to Flickr Terms of Use | COCO Consortium |
| [Imagenette](https://github.com/fastai/imagenette) | Apache-2.0 | fast.ai |

---
