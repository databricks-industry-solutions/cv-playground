# Ultralytics YOLO on Databricks — Examples

Train and deploy [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) object-detection models on Databricks across **AI Runtime (Serverless GPU)** and **Classic GPU Compute**, with [MLflow](https://docs.databricks.com/en/mlflow/index.html) tracking, [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html) governance, and [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html). All examples train [YOLO11n](https://docs.ultralytics.com/models/yolo11/) on [COCO128](https://www.kaggle.com/datasets/ultralytics/coco128) (demo-scale).

## Notebooks

| Notebook | Compute | Description |
|---|---|---|
| `air-yolo11n-detect-coco128-singleGPU.ipynb` | AI Runtime (Serverless GPU) — single GPU (A10 / H100) | Single-GPU training → MLflow tracking → custom PyFunc registration → Model Serving. Basis of the published blog. |
| `air-yolo11n-detect-coco128-multiGPU.ipynb` | AI Runtime (Serverless GPU) — distributed multi-GPU | Distributed multi-GPU training via `@distributed` (A10 / H100). |
| `classic-yolo11n-detect-coco128-singleA10.ipynb` | Classic GPU compute — single A10 | Single-GPU training on a classic GPU cluster. |
| `classic-yolo11n-detect-coco128-SingleNodeMultipleGPUs.ipynb` | Classic GPU compute — single-node multi-GPU | Single-node multi-GPU distributed training via [`TorchDistributor`](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html). |

## Requirements

- Databricks workspace with GPU compute (AI Runtime Serverless GPU, or classic GPU clusters)
- Unity Catalog enabled — set catalog / schema / volume via the notebook widgets
- Python 3.10+

## Related

- Blog: [Train and Deploy YOLO Vision Model on Databricks AI Runtime (AIR)](https://community.databricks.com/t5/technical-blog/train-and-deploy-yolo-vision-model-on-databricks-ai-runtime-air/ba-p/151558)
- [`../NuInsSeg/`](../NuInsSeg/) — YOLO instance-segmentation example (medical imaging)
