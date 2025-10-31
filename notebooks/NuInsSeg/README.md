### YOLO Instance Segmentation of CellType Nuclei 

[Fine-tune](https://docs.ultralytics.com/guides/model-evaluation-insights/#how-does-fine-tuning-work) custom data ([pre-formatted for YOLO framework](https://docs.ultralytics.com/datasets/segment/)) using [YOLO Instance Segmentation CV model](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml).

An overivew of the features integrated in our example walkthrough:    

| **Feature** | **Details**    |
|---------|-------------|
| Custom Data for Medical Imaging | [NuInsSeg](https://github.com/masih4/NuInsSeg) — one of the largest publicly available datasets for nuclei instance segmentation in H&E-stained histological images. <br>Using real-world, high-resolution data with complex tissue structures and challenging nuclei boundaries. <br>Pixel-level masks enable robust benchmarking for production medical imaging scenarios. <br><br> Use Case Processing: <br>- Size: 665 images (1024x1024) <br>- Classes: 1 (Nuclei)<br>- Splits: Train (399), Val (133), Test (133) <br>- Format: YOLO segmentation format (images + polygon annotations)|
| [Databricks Serverless](https://www.databricks.com/glossary/serverless-computing) [GPU](https://docs.databricks.com/aws/en/compute/serverless/gpu) Compute| Leverage automatic scaling, fast startup, and GPU acceleration (A10/H100) to efficiently run deep learning workloads without cluster management. <br>Enables cost-effective training, distributed multi-node support, and seamless integration with [Databricks ML tools](https://docs.databricks.com/aws/en/machine-learning). |
| Transfer Learning | Reduces training time and improves performance on custom datasets—even with limited labeled data. <br>Highly relevant e.g. in medical imaging, where annotated data is scarce and model generalization is critical. <br> <br>- Use Case: CellType Nuclei Segmentation <br>- Model: YOLO11n-seg (nano segmentation variant) <br>- Task: [Instance segmentation](https://docs.ultralytics.com/tasks/segment/) with pixel-level masks|
| [MLflow Integration](https://mlflow.org/docs/latest/) | Integrates [Databricks managed MLflow](https://docs.databricks.com/aws/en/mlflow/#databricks-managed-mlflow) for experiment tracking, model logging, and artifact management—essential for reproducibility, model versioning, and collaborative workflow. <br><br> Example Tracking: <br>- Custom callback functions for epoch-level logging <br>- Comprehensive metrics tracking (`mAP, precision, recall, fitness`) <br>- Checkpoint management with configurable frequency <br>- Best model tracking and artifact logging <br>- Training/validation/test metrics with visualizations|
| [Unity Catalog Volumes](https://docs.databricks.com/data-governance/unity-catalog/index.html) | Provides data governance and storage, crucial for modern MLOps. Enables secure, governed, and scalable storage with fine-grained access control, reproducible workflows, data lineage, and compliance across collaborative ML projects. <br><br> Benefits: <br>- Avoid 500MB workspace file size limits <br>- `POSIX`-like file operations <br>- Access control via Unity Catalog <br>- Data lineage tracking <br>- (Customizable) Artifact management |    
   
---   

#### Basic Requirements

- Databricks workspace with GPU compute access (Serverless GPU or classic GPU clusters)
- Unity Catalog enabled
- Set up Unity Catalog resources
   Configure your catalog, schema, and volume names in the notebook widgets e.g.:
   ```python
   CATALOG_NAME = "<your_catalog_name>"
   SCHEMA_NAME = "<computer_vision>"
   VOLUME_NAME = "<cv_project_name>"
   ```     

- Python 3.10+   
---   

#### Main Path & Notebooks:    
**PATH: `~/notebooks/NuInsSeg/InstanceSegmentation_sgc/`**

<!-- **1.**  -->
- **Single-Node Serverless GPU Training**    
    - **NOTEBOOK: `CellTypes_InstanceSeg_TransferLearn_serverlessA10_v0.3.3.py`**    
        <!-- - Transfer learning with YOLO11n-seg 
            1. Default YOLO framework
            1. MLflow tracking | logging | inferencing + UC Volumes integration  
        - NuInsSeg dataset (665 images: 399 train, 133 val, 133 test)  
        - Single A10 GPU on Serverless Compute  
        - Full MLflow integration with checkpointing  
        - Inference with metrics and visualizations   -->
        
    - **Key Results**: 50 epochs training (~2 hours), comprehensive metrics tracking, automated inference pipeline
        - _update to use smaller number of epochs for preliminary quick tests_  
              
    - **Utility Helper Modules: `notebooks/NuInsSeg/InstanceSegmentation_sgc/utils/`**    
            <!-- - `mlflow_callbacks.py` - MLflow integration, checkpointing, best model tracking    
            - `inference_utils.py` - Model loading, inference, metrics calculation    
            - `visualization_utils.py` - Prediction visualizations, comparison plots    
            - `yolo_utils.py` - Path management, environment setup, data validation    
            - `summary_utils.py` - Training and inference summaries, markdown export    
            - `cache_utils.py` - CUDA cache management, GPU monitoring    
            - `resume_callbacks.py` - Training resume functionality       -->
        <br> 

        <!-- - **Production-ready MLflow integration:**  
            - **Custom Callbacks**
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

            - **Metrics Tracked & Logged**
                - **`Training`**: Box loss, segmentation loss, class loss, DFL loss
                - **`Validation`**: mAP50, mAP50-95 (box and mask), precision, recall
                - **`Best Model`**: Fitness score (0.1 × mAP50 + 0.9 × mAP50-95)

            - **Artifacts Logged** [including YOLO defaults]
                - Model checkpoints (configurable frequency)
                - Best model weights
                - Training plots and metrics CSV
                - Dataset configuration (`data.yaml`)
                - Inference visualizations (customizable via code)   -->

<!-- ---     -->
<!-- **2.**  -->

- **Multi-Node Serverless GPU Training** [forthcoming...]   
<!-- ---     -->

<!-- - **NOTEBOOK: `02_CellTypes_InstanceSeg_TransferLearn_sgcA10_MultipleGPU_MlflowLoggingModel.py`** [to standardize nameing convention]  
- Distributed training with 8 A10 GPUs
- PyTorch DDP (DistributedDataParallel)
- NCCL backend for inter-GPU communication
- Detailed analysis of distributed training challenges   -->

<!-- **Note**: Includes comprehensive troubleshooting documentation for multi-node NCCL issues -->

<!-- ---     -->

<!-- ### Distributed Training Considerations

The repository includes detailed analysis of multi-node training challenges:  

1. **NCCL Communication Issues**: EFA/libfabric configuration
2. **Dataset Loading**: Large dataset handling with distributed coordination
3. **Timeout Management**: SGC timeout environment variables
4. **Network Transport**: Socket fallback vs. high-speed interconnects

See `02_CellTypes_InstanceSeg_TransferLearn_sgcA10_MultipleGPU_MlflowLoggingModel.py` [to standardize nameing convention] for full details. --> 

---  

##### Dependencies used in this cv-application use case and corresponding licenses:

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
