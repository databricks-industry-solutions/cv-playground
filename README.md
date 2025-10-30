# Computer Vision Playground - YOLO Instance Segmentation on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

## 📖 Overview

This repository showcases **Computer Vision solutions on Databricks**, with a focus on **YOLO (You Only Look Once) Instance Segmentation** for medical imaging and other CV applications. The project demonstrates end-to-end ML workflows including:

- 🎯 **Transfer Learning** with YOLO11 for instance segmentation
- 🚀 **Scalable Training** on Databricks Serverless GPU (A10) and multi-node GPU compute
- 📊 **MLflow Integration** for experiment tracking, model logging, and artifact management
- 🗄️ **Unity Catalog Volumes** for data governance and storage
- 🔬 **Medical Imaging** use case with NuInsSeg dataset (nuclei segmentation in H&E-stained histological images)

## 🏗️ Project Structure

```
cv-playground-1/
├── notebooks/
│   ├── 01_MultiNode A10 training on SGC using Ultralytics YOLO CV model with coco128 image dataset.ipynb
│   ├── 02_CellTypes_InstanceSeg_TransferLearn_sgcA10_MultipleGPU_MlflowLoggingModel.py
│   └── NuInsSeg/
│       └── InstanceSegmentation_sgc/
│           ├── CellTypes_InstanceSeg_TransferLearn_serverlessA10_v0.3.3.py
│           ├── data.yaml
│           ├── datasets/  # YOLO-formatted image datasets (train/val/test)
│           ├── imgs/
│           └── utils/     # Modular utility functions
│               ├── mlflow_callbacks.py     # MLflow integration & checkpointing
│               ├── inference_utils.py      # Model inference utilities
│               ├── visualization_utils.py  # Results visualization
│               ├── yolo_utils.py          # YOLO path & environment setup
│               ├── summary_utils.py       # Reporting utilities
│               ├── cache_utils.py         # CUDA/memory management
│               └── resume_callbacks.py    # Training resume utilities
├── requirements.txt
├── databricks.yml
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Databricks workspace with GPU compute access (Serverless GPU or classic GPU clusters)
- Unity Catalog enabled
- Python 3.10+

### Installation

1. **Clone this repository to your Databricks Workspace**

   ```bash
   git clone https://github.com/your-org/cv-playground-1.git
   ```

2. **Install dependencies**

   The required packages are specified in `requirements.txt`. Key dependencies include:
   - `ultralytics` - YOLO framework
   - `torch` - PyTorch deep learning
   - `mlflow` - Experiment tracking
   - `opencv-python` - Image processing
   - See `requirements.txt` for full list

3. **Set up Unity Catalog resources**

   Configure your catalog, schema, and volume names in the notebook widgets:
   ```python
   CATALOG_NAME = "your_catalog"
   SCHEMA_NAME = "computer_vision"
   VOLUME_NAME = "projects"
   ```

### Quick Start Example

Run the instance segmentation notebook:

```python
# notebooks/NuInsSeg/InstanceSegmentation_sgc/CellTypes_InstanceSeg_TransferLearn_serverlessA10_v0.3.3.py

# 1. Setup paths and data
# 2. Train YOLO model with MLflow tracking
# 3. Run inference on test set
# 4. Visualize results
```

## 📊 Key Features

### 1. YOLO Instance Segmentation

- **Model**: YOLO11n-seg (nano segmentation variant)
- **Task**: Instance segmentation with pixel-level masks
- **Use Case**: Nuclei segmentation in histological images (NuInsSeg dataset)

### 2. Databricks Integration

- **Serverless GPU Compute**: A10 GPU instances with automatic scaling
- **Multi-node Training**: Distributed training across multiple GPUs
- **Unity Catalog Volumes**: Governed data storage with POSIX-like file access

### 3. MLflow Experiment Tracking

- Custom callback functions for epoch-level logging
- Comprehensive metrics tracking (mAP, precision, recall, fitness)
- Checkpoint management with configurable frequency
- Best model tracking and artifact logging
- Training/validation/test metrics with visualizations

### 4. Modular Utilities

Well-organized utility modules in `notebooks/NuInsSeg/InstanceSegmentation_sgc/utils/`:

- `mlflow_callbacks.py` - MLflow integration, checkpointing, best model tracking
- `inference_utils.py` - Model loading, inference, metrics calculation
- `visualization_utils.py` - Prediction visualizations, comparison plots
- `yolo_utils.py` - Path management, environment setup, data validation
- `summary_utils.py` - Training and inference summaries, markdown export
- `cache_utils.py` - CUDA cache management, GPU monitoring
- `resume_callbacks.py` - Training resume functionality

## 📝 Example Notebooks

### 1. Single-Node Serverless GPU Training

`CellTypes_InstanceSeg_TransferLearn_serverlessA10_v0.3.3.py`

- Transfer learning with YOLO11n-seg
- NuInsSeg dataset (665 images: 399 train, 133 val, 133 test)
- Single A10 GPU on Serverless Compute
- Full MLflow integration with checkpointing
- Inference with metrics and visualizations

**Key Results**: 50 epochs training (~2 hours), comprehensive metrics tracking, automated inference pipeline

### 2. Multi-Node Multi-GPU Training

`02_CellTypes_InstanceSeg_TransferLearn_sgcA10_MultipleGPU_MlflowLoggingModel.py`

- Distributed training with 8 A10 GPUs
- PyTorch DDP (DistributedDataParallel)
- NCCL backend for inter-GPU communication
- Detailed analysis of distributed training challenges

**Note**: Includes comprehensive troubleshooting documentation for multi-node NCCL issues

## 🔬 Dataset: NuInsSeg

[NuInsSeg](https://github.com/masih4/NuInsSeg) is one of the largest publicly available datasets for nuclei instance segmentation in H&E-stained histological images.

- **Size**: 665 images (1024x1024)
- **Classes**: 1 (Nuclei)
- **Splits**: Train (399), Val (133), Test (133)
- **Format**: YOLO segmentation format (images + polygon annotations)

## 📈 MLflow Integration Highlights

The project includes production-ready MLflow integration:

### Custom Callbacks

```python
from utils import mlflow_epoch_logger, configure_checkpoint_logging

# Configure checkpoint frequency
configure_checkpoint_logging(
    frequency=10,      # Log every 10 epochs
    log_best=True,     # Always log best model
    log_final=True,    # Log final epoch
    log_first=True     # Log first epoch
)

# Register callback
model.add_callback("on_fit_epoch_end", mlflow_epoch_logger)
```

### Metrics Tracked

- **Training**: Box loss, segmentation loss, class loss, DFL loss
- **Validation**: mAP50, mAP50-95 (box and mask), precision, recall
- **Best Model**: Fitness score (0.1 × mAP50 + 0.9 × mAP50-95)

### Artifacts Logged

- Model checkpoints (configurable frequency)
- Best model weights
- Training plots and metrics CSV
- Dataset configuration (data.yaml)
- Inference visualizations

## 🛠️ Advanced Topics

### Distributed Training Considerations

The repository includes detailed analysis of multi-node training challenges:

1. **NCCL Communication Issues**: EFA/libfabric configuration
2. **Dataset Loading**: Large dataset handling with distributed coordination
3. **Timeout Management**: SGC timeout environment variables
4. **Network Transport**: Socket fallback vs. high-speed interconnects

See `02_CellTypes_InstanceSeg_TransferLearn_sgcA10_MultipleGPU_MlflowLoggingModel.py` for full details.

### UC Volumes for Data Governance

Benefits of using Unity Catalog Volumes:

- ✅ No 500MB workspace file size limits
- ✅ POSIX-like file operations
- ✅ Access control via Unity Catalog
- ✅ Data lineage tracking
- ✅ Organized artifact management

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** this repository
2. **Clone** locally and make your changes
3. **Test** with Databricks CLI against a development workspace
4. **Submit** a pull request with detailed description
5. **Code Review** by at least one team member

Please ensure:
- Code follows existing patterns and style
- Notebooks run end-to-end without errors
- Documentation is updated for new features
- MLflow tracking works correctly

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 📄 Third-Party Package Licenses

&copy; 2025 Databricks, Inc. All rights reserved. The source in this project is provided subject to the Databricks License [https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses set forth below.

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

## 📚 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Databricks Machine Learning Guide](https://docs.databricks.com/machine-learning/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [NuInsSeg Dataset](https://github.com/masih4/NuInsSeg)

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Authors**: 
- may.merkletan@databricks.com
- yang.yang@databricks.com

**Last Updated**: October 2025
