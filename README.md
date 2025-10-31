# CV-Playground
**Computer Vision Applications on Databricks**  


[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

## Overview

This repository is intended to be a _`"playground"`_ where we share **Applications of Computer Vision Solutions on Databricks**. As a start we will highlight [Instance Segmentation](https://www.ultralytics.com/glossary/instance-segmentation) application(s) in medical imaging using [Ultralytics **YOLO** (You Only Look Once)](https://github.com/ultralytics/ultralytics) framework. Other CV applications will be added when available.    
<!-- _... stay tuned!_   -->    

---   

#### YOLO Instance Segmentation of Nuclei 

The application demonstrates how we can [fine-tune](https://docs.ultralytics.com/guides/model-evaluation-insights/#how-does-fine-tuning-work) custom data ([pre-formatted for YOLO framework](https://docs.ultralytics.com/datasets/segment/)) using [YOLO Instance Segmentation CV model](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml).

Our example walkthrough highlights the following:    

| **Feature** | **Details**    |
|---------|-------------|
| Custom Data for Medical Imaging | [NuInsSeg](https://github.com/masih4/NuInsSeg) — one of the largest publicly available datasets for nuclei instance segmentation in H&E-stained histological images. <br>Using real-world, high-resolution data with complex tissue structures and challenging nuclei boundaries. <br>Pixel-level masks enable robust benchmarking for production medical imaging scenarios. |
| [Databricks Serverless](https://www.databricks.com/glossary/serverless-computing) [GPU](https://docs.databricks.com/aws/en/compute/serverless/gpu) Compute| Leverage automatic scaling, fast startup, and GPU acceleration (A10/H100) to efficiently run deep learning workloads without cluster management. <br>Enables cost-effective training, distributed multi-node support, and seamless integration with [Databricks ML tools](https://docs.databricks.com/aws/en/machine-learning). |
| Transfer Learning | Using [YOLO11 for instance segmentation](https://docs.ultralytics.com/tasks/segment/). <br>Transfer learning reduces training time and improves performance on custom datasets—even with limited labeled data. <br>Highly relevant e.g. in medical imaging, where annotated data is scarce and model generalization is critical. |
| [MLflow Integration](https://mlflow.org/docs/latest/) | Integrates [Databricks managed MLflow](https://docs.databricks.com/aws/en/mlflow/#databricks-managed-mlflow) for experiment tracking, model logging, and artifact management—essential for reproducibility, model versioning, and collaborative workflow. |
| [Unity Catalog Volumes](https://docs.databricks.com/data-governance/unity-catalog/index.html) | Provides data governance and storage, crucial for modern MLOps. Enables secure, governed, and scalable storage with fine-grained access control, reproducible workflows, data lineage, and compliance across collaborative ML projects. |    
   

---   

## Getting Started

Clone this repository to your Databricks Workspace.  
You will find the example applications in each subfolder within the `/notebooks` folder.   
Follow the `README.md` within each of the `/notebooks/{application_example_folder}/README.md` to get started.  

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

---   

