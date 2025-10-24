# Databricks notebook source
# MAGIC %md
# MAGIC - This is a standalone example of how to run YOLO Instance Segmentation on a custom dataset with [serverless gpu compute (SGC)](https://www.databricks.com/glossary/serverless-computing).   
# MAGIC - The example solution will be part of the assets within the forthcoming [databricks-industry-solutions/cv-playground](https://github.com/databricks-industry-solutions/cv-playground) that will show case other CV-related solutions on Databricks.     
# MAGIC - Developed and last tested [`2025Oct24`] using `sgc_A10` with pinned dependencies by `may.merkletan@databricks.com`.  
# MAGIC - **v0.3.3**: Simplified version using utility modules - includes both default YOLO approach and improved MLflow-integrated approach.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Overview: 
# MAGIC <br> 
# MAGIC
# MAGIC #### YOLO Model
# MAGIC [YOLO (You Only Look Once)](https://ieeexplore.ieee.org/document/7780460) is a state-of-the-art, real-time object detection system. It frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. This approach allows YOLO to achieve high accuracy and speed, making it suitable for real-time applications.    
# MAGIC
# MAGIC <!-- ![Computer Vision Tasks supported by Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif)    -->
# MAGIC
# MAGIC Offered as part of the [Ultralytics AI framework](https://www.ultralytics.com/), [YOLO11 supports multiple computer vision tasks](https://docs.ultralytics.com/tasks/).    
# MAGIC #### Instance Segmentation
# MAGIC Recent updates to the YOLO model have introduced capabilities for instance segmentation. [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/#models) not only detects objects but also delineates the exact shape of each object, providing pixel-level masks. 
# MAGIC ![what_is_instance_segmentation](./imgs/what_is_instance_segmentation.png)
# MAGIC
# MAGIC <!-- ![https://manipulation.csail.mit.edu/segmentation.html](https://manipulation.csail.mit.edu/data/coco_instance_segmentation.jpeg)  -->
# MAGIC       
# MAGIC This is particularly useful in applications requiring precise object boundaries, for example in medical imaging, autonomous driving, as well as robotics e.g. : 
# MAGIC <img src="https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif"
# MAGIC      alt="YOLO Instance Segmentation Example"
# MAGIC      width="800"
# MAGIC      style="margin: 200px;"/>    
# MAGIC      `image from https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif`
# MAGIC <!-- ![YOLO Instance Segmentation Example](https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif) -->
# MAGIC <br>
# MAGIC
# MAGIC #### Transfer Learning
# MAGIC [Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) involves taking a pre-trained model and fine-tuning it on a new dataset. This approach leverages the knowledge gained from a large dataset and applies it to a specific task, reducing the need for extensive computational resources and training time. In the context of YOLO, transfer learning allows us to adapt the model to new object classes or domains with limited data. 
# MAGIC <br>
# MAGIC
# MAGIC #### Note on architecture and segmentation implementation 
# MAGIC - **YOLO Architecture**: The YOLO architecture consists of convolutional layers followed by fully connected layers, designed to predict bounding boxes and class probabilities directly from the input image.
# MAGIC - **Instance Segmentation**: The recent YOLO models incorporate segmentation heads that output pixel-level masks for each detected object, enhancing the model's ability to perform instance segmentation.    
# MAGIC <br> 
# MAGIC
# MAGIC For more detailed information, refer to the original YOLO model and the latest research on instance segmentation.    
# MAGIC `YOLO Refs:`
# MAGIC [`v1`](https://arxiv.org/abs/1506.02640), [`v2`](https://arxiv.org/abs/1612.08242), [`v3`](https://arxiv.org/abs/1804.02767), [`v4`](https://arxiv.org/abs/2004.10934), 
# MAGIC [`v5`](https://docs.ultralytics.com/models/yolov5/), 
# MAGIC [`v6`](https://arxiv.org/abs/2209.02976), [`v7`](https://arxiv.org/abs/2207.02696), [`v8`](https://docs.ultralytics.com/models/yolov8/), ... [`v11`](https://docs.ultralytics.com/models/yolo11/#overview) (recent version focused on improved performance and ease of use, used in this example)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Applying `YOLO_v11` Instance Segmentation within Databricks 
# MAGIC In the rest of this notebook, we will provide an example of how to leverage [YOLO Instance Segmentation](https://docs.ultralytics.com/tasks/segment/) model in _transfer learning_.     
# MAGIC  
# MAGIC Specifically, we will _finetune_ the `YOLO_v11 Instance Segmentation` model on a new dataset, the [NuInsSeg Dataset](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images), _one of the largest publicly available datasets of segmented nuclei in [H&E-Stained](https://en.wikipedia.org/wiki/H%26E_stain) Histological Images_ (images below of the flow of processes illustrate how these sample data are typically derived).   
# MAGIC
# MAGIC ---     
# MAGIC
# MAGIC [<img src="https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/prepration.png" width="800"/>](https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/prepration.png)
# MAGIC [<img src="https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/segmentation%20sample.jpg" width="800"/>](https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/segmentation%20sample.jpg)
# MAGIC
# MAGIC ---     
# MAGIC
# MAGIC We will run the _finetuning_ on the Databricks Intelligence platform using [serverless compute](https://www.databricks.com/glossary/serverless-computing). 
# MAGIC <!-- preprocessed in workspace folder -->
# MAGIC To focus our example on the application of YOLO Instance Segmentation, we have already pre-processed the [NuInsSeg Dataset](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images) images in [YOLO format](https://docs.ultralytics.com/datasets/segment/) and included them within the `datasets` folder within the workspace path where this notebook resides.    
# MAGIC Along with the `datasets`, we also have information on how the data is organized within the corresponding `data.yaml`. 
# MAGIC ### What this notebook walks you through:  
# MAGIC **[1] _`Default`_ YOLO setup on serverless compute for transfer learning + quick inference    
# MAGIC [2] Integration with [Databricks managed MLflow](https://www.databricks.com/product/managed-mlflow) wrt model development tracking and logging + inference using the best checkpoint of the trained YOLO segmentation model.**
# MAGIC **This simplified v0.3 notebook demonstrates:**
# MAGIC 1. **Default YOLO approach** - Quick setup with default Ultralytics settings and its limitations ([a] workspace path limits, [b] minimal MLflow integration)
# MAGIC 2. **Improved approach addressing [a] & [b]:**
# MAGIC    - **Modular utility functions** - Organized in separate modules for better maintainability
# MAGIC    - **YOLO setup on serverless compute** - Transfer learning with comprehensive MLflow integration
# MAGIC    - **Databricks managed MLflow integration** - Full tracking of parameters, metrics, checkpoints, and artifacts
# MAGIC    - **Inference with best checkpoint** - Automated inference and visualization on validation/test sets
# MAGIC      
# MAGIC For [distributed PyTorch training across multiple GPUS](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/), as well as [registering to Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) and [serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/manage-serving-endpoints) the YOLO model via [MLflow Custom Pyfunc wrapper](https://mlflow.org/docs/2.22.1/traditional-ml/creating-custom-pyfunc), please refer to the linked Databricks reference documentations (and the forthcoming solution accelerator **`cv-playground`** within [Databricks-Industry-Solutions](https://github.com/databricks-industry-solutions).) 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Pinned Dependencies
# MAGIC %pip install -q \
# MAGIC     ultralytics==8.3.200 \
# MAGIC     torch==2.6.0+cu124 \
# MAGIC     mlflow==2.21.3 \
# MAGIC     scikit-learn==1.7.2 \
# MAGIC     matplotlib==3.10.7 \
# MAGIC     nvidia-ml-py>=12.0.0 \
# MAGIC     threadpoolctl==3.6.0 \
# MAGIC     --upgrade
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import subprocess
import shutil
import tempfile
import json
import random
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
import mlflow
import torch
import torch.distributed as dist
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms as T
import cv2
from ultralytics import YOLO

# Import utility modules from utils package
# Force reload all utils modules
import sys
import importlib

# Remove all utils modules from cache
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('utils')]
for module in modules_to_remove:
    del sys.modules[module]

# Now import fresh
import utils
importlib.reload(utils)

from utils import (
    # yolo_utils
    set_seeds,
    path_exists,
    get_organized_paths,
    setup_yolo_paths, 
    check_yolo_environment,
    get_yolo_paths,
    get_inference_output_path,
    validate_data_yaml,
    get_split_info,
    copy_to_uc_volumes_with_yaml,

    # mlflow_callbacks
    mlflow_epoch_logger,
    configure_checkpoint_logging,
    copy_training_artifacts,
    log_training_artifacts_to_mlflow,
    finalize_training_run,

    # inference_utils
    find_model_by_run_id,
    load_model_from_run,
    run_inference_with_metrics,
    inspect_inference_output,

    # visualization_utils
    visualize_inference_results,
    visualize_predictions_vs_ground_truth,
    
    # summary_utils
    print_inference_summary,
    print_multi_split_summary,
    export_inference_summary_markdown,

    # cache_utils
    clear_cuda_cache,
    clear_cuda_cache_aggressive,
    gpu_status,
    clear_all_caches
)

# COMMAND ----------

# DBTITLE 1,RAND seeds
set_seeds()

# COMMAND ----------

# DBTITLE 1,Catalog-Schema-Volume NAMES
# Replace with your specific catalog and schema etc. names
dbutils.widgets.text("CATALOG_NAME", "mmt","Catalog Name")
dbutils.widgets.text("SCHEMA_NAME","cv","Schema Name")
dbutils.widgets.text("VOLUME_NAME","projects","Volume Name")

#Get the catalog, schema and volume variables
CATALOG_NAME = dbutils.widgets.get("CATALOG_NAME")
SCHEMA_NAME = dbutils.widgets.get("SCHEMA_NAME")
VOLUME_NAME = dbutils.widgets.get("VOLUME_NAME")

# COMMAND ----------

# DBTITLE 1,CREATE IF NOT EXISTS catalog.schema.volumes
# Create catalog if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")

# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

# Create volume if not exists
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}")

# COMMAND ----------

# DBTITLE 1,PATH NAMES
## Volumes path prefix
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}"

PROJECTS_DIR = f"{VOLUME_PATH}/projects"
PROJECT_PATH = f"{PROJECTS_DIR}/NuInsSeg"

YOLO_DATA_DIR = f"{PROJECT_PATH}/yolo_dataset"

# Get the current working directory
nb_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
current_path = f"/Workspace{nb_context}"
WS_PROJ_DIR = '/'.join(current_path.split('/')[:-1]) 

USER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_WORKSPACE_PATH = f"/Users/{USER}"

### Define experiment name
project_name = "yolo_CellTypesNuclei_InstanceSeg_scg"
experiment_name = f"{USER_WORKSPACE_PATH}/{project_name}"

mlflow.set_experiment(experiment_name)
print(f"Setting experiment name to be {experiment_name}")

# COMMAND ----------

# DBTITLE 1,check paths
# ============================================================================
# SETUP & CHECK YOLO Paths and Environment
# ============================================================================

# Setup all paths including Ultralytics config
paths = setup_yolo_paths(
    project_path='/Volumes/mmt/cv/projects/NuInsSeg',
    set_artifacts_path=True,
    verbose=True
)


# COMMAND ----------

# DBTITLE 1,preprocessed DATA in YOLO format in workspace
!ls -lah {WS_PROJ_DIR}/datasets/ 

# COMMAND ----------

# DBTITLE 1,data.yaml specifying data paths
!cat data.yaml

# COMMAND ----------

# MAGIC %md
# MAGIC #### We will first illustrate the `Default` local workspace paths used by Ultralytics.
# MAGIC We will re-define these default paths later to illustrate how one would use the preprocessed image datasets ingested and written to UC Vols. and organize the generated assets in model training 

# COMMAND ----------

# DBTITLE 1,[Default] Ultralytics PATHS
from ultralytics import settings

# View all default settings
print(settings)

# datasets_dir	'/path/to/datasets'	str	The workspace sub-directory relative to notebook path where the datasets are stored
# weights_dir	'/path/to/weights'	str	The workspace sub-directory relative to notebook path where the model weights are stored
# runs_dir	'/path/to/runs'	str	The workspace sub-directory relative to notebook path where the experiment runs are stored

## by default:
# "mlflow": true, # however this isn't exactly logged to the mlflow expt.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run a 'quick' model training to illustrate where data/folder paths are found 'by default' 

# COMMAND ----------

# DBTITLE 1,test writing results to UC Vols
# yolo_default = False # if we have already ran & will go straight to using UC Volumes and mlflow checkpoints

yolo_default = True # 

# COMMAND ----------

# DBTITLE 1,"DEFAULT" SETTINGS
## if True, we will run the transfer learning using Default YOLO settings 

if yolo_default:

    ## "DEFAULT" Quick Start 

    import torch.distributed as dist
    from ultralytics import YOLO

    # Initialize the process group | wrt serverless requires specifying world_size and rank
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',  # required for cuda
            world_size=1,    # set world_size to 1 for single process
            rank=0           # set rank to 0 for single process
        )

    try:
        # Transfer the weights from a pretrained model (recommended for training)
        model = YOLO("yolo11n-seg.pt") 
        ## also this will be loaded to the path; if not specified this will be loaded to directory where model training code nb resides WS_PROJ_DIR 

        results = model.train(                          
                                data=f"{WS_PROJ_DIR}/data.yaml", ## default settings, YOLO assumes a "datasets" dir in directory where model training code nb resides WS_PROJ_DIR
                                
                                epochs=50, #10, # at least 50 for a decent inference, 
                                ## reduce epochs to e.g. 10 for a quicker transfer learning training run
                                patience=0,  ## setting patience=0 to disable early stopping
                                batch=8, ## increase or decrease depending on GPU memory
                                imgsz=1024, ## size of NuInsSeg images
                                optimizer="adam", ## alternatives e.g. "adamw","sgd",
                                device=-1, ## most idle GPU
                                
                                save=True,
                                
                                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                                project=WS_PROJ_DIR,                                                          
                            )
        
    finally:
        # Destroy the process group
        dist.destroy_process_group()

# COMMAND ----------

# MAGIC %md
# MAGIC Results saved to path relative to this notebook: `./runs/segment/train_sgc/weights/`     
# MAGIC *NB: If you retrain with same configs -- addiitonal training paths and subdirs will be added to `./runs/segment/train_sgc{#}/`

# COMMAND ----------

# DBTITLE 1,list workspace weights folder path
!ls -lah ./runs/segment/train_sgc/weights/

# COMMAND ----------

# DBTITLE 1,Image Results
if yolo_default==True:
  display(Image(filename="runs/segment/train_sgc/results.png"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Inference with default YOLO framework + Workspace datapath best weights 

# COMMAND ----------

# DBTITLE 1,Load model with WS best trained weights and make inference predictions
# Define the project directory
test_data_path = f"{WS_PROJ_DIR}/datasets/test/images"

# Load the trained YOLO model
model = YOLO("./runs/segment/train_sgc/weights/best.pt")

# Load test data
print(test_data_path)

# Check if the directory exists
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"The directory {test_data_path} does not exist.")

test_images = [os.path.join(test_data_path, img) for img in os.listdir(test_data_path) if img.endswith('.png')]

# Randomly select 25 images
selected_images = random.sample(test_images, 25)

# Resize images to a smaller size
resized_images = []
for img_path in selected_images:
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (640, 640))  # Resize to 640x640 or any desired size
    resized_images.append(resized_img)

# Batch Predict using the loaded model
inferece_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = model.predict(resized_images,
                        save=True,  # Save predictions to disk
                        project="./runs/predict",  # Base directory on workspace path for saving
                        name=f"test_inference_{inferece_timestamp}"    # Subdirectory name
                       )


# Plot 5x5 grid of test images with predictions
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()
for img, result, ax in zip(resized_images, results, axes):
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Flatten the list before mapping to int
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # cv2.putText(img, f"{box.cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
plt.tight_layout()
plt.show()

## NB processing speed is much faster after the initial first inference ~50ms vs ~5ms

# COMMAND ----------

# MAGIC %md
# MAGIC Using the `Default` Ultralytics `model.predict()` on a random sample of test images we observe the relative efficiency the inference can be performed on YOLO-formatted test images.
# MAGIC Depending on the length of training epochs and different batch sizes, inference performance can further improve. 

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC Overall, it is quick to set up training using default YOLO setttings as long as images are preprocessed and formatted in [YOLO format](https://docs.ultralytics.com/datasets/detect/#which-datasets-are-supported-by-ultralytics-yolo-for-object-detection).    
# MAGIC Model training and validation using YOLO-based metrics tracking within workspace path are provided "out-of-the-box".  
# MAGIC However, there is minimal MLflow integration.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limitations of `Default` Ultralytics `settings`: 
# MAGIC - **[a] Workspace folder path with files written there by default**
# MAGIC   - Workspace [limitations](https://docs.databricks.com/aws/en/files/workspace#limitations) exist and files are subjected to [500mb filesize limits](https://docs.databricks.com/aws/en/files/workspace#file-size-limit)
# MAGIC   - Training `runs` and `results` currently saved to `./runs/segment/train_sgc{#}`
# MAGIC   - Ideally it would be good to update and/or write to UC volumes project-related paths e.g. 
# MAGIC     - location where `yolo_dataset` was originally downloaded and preprocessed for yolo-formatted image reference
# MAGIC     - organize `runs` by `run_name` and `datetime`
# MAGIC - **[b] Not Databricks-managed MLflow logged or tracked**   
# MAGIC   - `experiment_id`; `experiment_name`: (when not specified, the default "Ultralytics" is set e.g. on Classic MLdbr compute)
# MAGIC   - `run_id`; `run_name`
# MAGIC   - `model` metrics -- not logged to MLflow
# MAGIC   - `system` metrics -- GPU usage is shown in the training printouts but not logged to MLFlow
# MAGIC   - `checkpointing` is missing and not best inference meal is racked 
# MAGIC - **[c] Model not registered as UC model**
# MAGIC   - Requires first defining the YOLO model as [MLflow Custom Pyfunc](https://mlflow.org/docs/2.22.1/traditional-ml/creating-custom-pyfunc) with load-context linked to artifacth paths before logging to MLflow and subsequently registered a UC model ([Databricks ref](https://docs.databricks.com/aws/en/machine-learning/model-serving/deploy-custom-python-code))
# MAGIC - **[d] GPU resources not fully leveraged where multiple GPUs exsits** (e.g. in Classic MLdbr and Serverless -- forthcoming)
# MAGIC     - single node -- multiple workers; 
# MAGIC     - multiple nodes -- multiple workers
# MAGIC     
# MAGIC      
# MAGIC For a fuller integration with the Databricks ecosystem would require a few additional tweaks. 

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### We will address [a] & [b] in this section:
# MAGIC [c] and [d] will be addressed in a separate notebook for distributed DL training using either Classic and/or Serverless GPU Compute (forthcoming).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation: Copy YOLO Images to Unity Catalog Volumes
# MAGIC To make sure we have sufficient space and paths to organize our training outputs, MLflow tracking, and logging, we will copy our workspace YOLO formatted data to UC Volumes. 
# MAGIC _The [NuInsSeg data](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images), while not huge in size, takes about 5-10 mins for the copying to UC Volumes. Ideally the data is downloaded to UC Vols and the preprocessed versions are updated in UC via medallion ETL. For the simplicity of this example we make them available via the workspace path as a start._

# COMMAND ----------

# DBTITLE 1,YOLO_DATA_UCVol_path
## Specify the UC Volumes destination path for the YOLO dataset
YOLO_DATA_UCVol_path = f'{PROJECT_PATH}/yolo_dataset_on_vols'

# COMMAND ----------

# DBTITLE 1,copy datasets to UC Vols (if needed)
## Execute the copy ~8mins | leave commented out if already done else uncomment & run
# data_yaml_path = copy_to_uc_volumes_with_yaml(WS_PROJ_DIR, YOLO_DATA_UCVol_path)

##NB: rsync will attempt to make copy -- you may see initial errors/warnings. 

# COMMAND ----------

# DBTITLE 1,example copy output

# ======================================================================
# COPYING DATASET TO UC VOLUMES
# ======================================================================
# Source: /Workspace/Users/may.merkletan@databricks.com/REPOs/cv-playground/notebooks/NuInsSeg/TESTING_final/datasets/
# Destination: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols

# 1. Creating destination directory...
#    ✓ Created: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols

# 2. Copying dataset files...
# sending incremental file list
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/.": Operation not permitted (1)
# ./
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/test": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/test/images": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/test/labels": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/train": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/train/images": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/train/labels": Operation not permitted (1)
# test/
# test/images/
# test/images/human_bladder_01.png
#         434,387 100%    2.02MB/s    0:00:00 (xfr#1, ir-chk=1202/1209)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/val": Operation not permitted (1)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/val/images": Operation not permitted (1)
# test/images/human_bladder_11.png
#         429,415 100%  710.76kB/s    0:00:00 (xfr#2, ir-chk=1201/1209)
# test/images/human_bladder_12.png
#         428,915 100%  448.46kB/s    0:00:00 (xfr#3, ir-chk=1200/1209)
# test/images/human_brain_12.png
#         420,380 100%  333.76kB/s    0:00:01 (xfr#4, ir-chk=1199/1209)
# test/images/human_brain_8.png
#         435,915 100%    1.03MB/s    0:00:00 (xfr#5, ir-chk=1198/1209)
# test/images/human_brain_9.png
#         423,701 100%  509.57kB/s    0:00:00 (xfr#6, ir-chk=1197/1209)
# test/images/human_cardia_1.png
#         406,425 100%  324.53kB/s    0:00:01 (xfr#7, ir-chk=1196/1209)
# test/images/human_cardia_3.png
#         410,366 100%    1.06MB/s    0:00:00 (xfr#8, ir-chk=1195/1209)
# test/images/human_cardia_5.png
#         416,013 100%  540.96kB/s    0:00:00 (xfr#9, ir-chk=1194/1209)
# test/images/human_cerebellum_4.png
#         427,189 100%  402.29kB/s    0:00:01 (xfr#10, ir-chk=1193/1209)
# test/images/human_cerebellum_5.png
#         433,643 100%  845.27kB/s    0:00:00 (xfr#11, ir-chk=1192/1209)
# test/images/human_cerebellum_6.png
#         440,188 100%  445.00kB/s    0:00:00 (xfr#12, ir-chk=1191/1209)
# test/images/human_epiglottis_10.png
#         432,971 100%  314.60kB/s    0:00:01 (xfr#13, ir-chk=1190/1209)
# test/images/human_epiglottis_4.png
#         440,255 100%  574.78kB/s    0:00:00 (xfr#14, ir-chk=1189/1209)
# test/images/human_epiglottis_6.png
#         444,078 100%  353.73kB/s    0:00:01 (xfr#15, ir-chk=1188/1209)
# test/images/human_jejunum_02.png
#         403,443 100%  992.41kB/s    0:00:00 (xfr#16, ir-chk=1187/1209)
# test/images/human_jejunum_03.png
#         414,402 100%  466.77kB/s    0:00:00 (xfr#17, ir-chk=1186/1209)
# test/images/human_jejunum_05.png
#         418,327 100%  340.15kB/s    0:00:01 (xfr#18, ir-chk=1185/1209)
# test/images/human_jejunum_08.png
#         410,908 100%    1.08MB/s    0:00:00 (xfr#19, ir-chk=1184/1209)
# test/images/human_kidney_02.png
#         415,369 100%  566.53kB/s    0:00:00 (xfr#20, ir-chk=1183/1209)
# test/images/human_kidney_07.png
#         401,969 100%  364.14kB/s    0:00:01 (xfr#21, ir-chk=1182/1209)
# test/images/human_kidney_08.png
#         401,044 100%    1.15MB/s    0:00:00 (xfr#22, ir-chk=1181/1209)
# test/images/human_liver_02.png
#         411,678 100%  610.06kB/s    0:00:00 (xfr#23, ir-chk=1180/1209)
# test/images/human_liver_03.png
#         410,668 100%  399.84kB/s    0:00:01 (xfr#24, ir-chk=1179/1209)
# test/images/human_liver_04.png
#         410,969 100%    1.37MB/s    0:00:00 (xfr#25, ir-chk=1178/1209)
# test/images/human_liver_09.png
#         411,844 100%  479.94kB/s    0:00:00 (xfr#26, ir-chk=1177/1209)
# test/images/human_liver_10.png
#         413,132 100%  343.65kB/s    0:00:01 (xfr#27, ir-chk=1176/1209)
# test/images/human_liver_11.png
#         409,837 100%    1.20MB/s    0:00:00 (xfr#28, ir-chk=1175/1209)
# test/images/human_liver_13.png
#         410,208 100%  585.66kB/s    0:00:00 (xfr#29, ir-chk=1174/1209)
# test/images/human_liver_22.png
#         410,241 100%  393.54kB/s    0:00:01 (xfr#30, ir-chk=1173/1209)
# test/images/human_liver_29.png
#         417,481 100%    1.17MB/s    0:00:00 (xfr#31, ir-chk=1172/1209)
# test/images/human_liver_31.png
#         537,989 100%  701.44kB/s    0:00:00 (xfr#32, ir-chk=1171/1209)
# test/images/human_liver_38.png
#         418,160 100%  376.72kB/s    0:00:01 (xfr#33, ir-chk=1170/1209)
# test/images/human_melanoma_01.png
#         428,313 100%    1.26MB/s    0:00:00 (xfr#34, ir-chk=1169/1209)
# test/images/human_melanoma_02.png
#         431,318 100%  461.35kB/s    0:00:00 (xfr#35, ir-chk=1168/1209)
# test/images/human_melanoma_03.png
#         438,290 100%  295.18kB/s    0:00:01 (xfr#36, ir-chk=1167/1209)
# test/images/human_melanoma_05.png
#         426,229 100%  507.61kB/s    0:00:00 (xfr#37, ir-chk=1166/1209)
# test/images/human_muscle_6.png
#         424,189 100%  354.66kB/s    0:00:01 (xfr#38, ir-chk=1165/1209)
# test/images/human_oesophagus_02.png
#         404,625 100%  978.07kB/s    0:00:00 (xfr#39, ir-chk=1164/1209)
# test/images/human_oesophagus_04.png
#         407,429 100%  436.75kB/s    0:00:00 (xfr#40, ir-chk=1163/1209)
# test/images/human_oesophagus_12.png
#         401,827 100%  299.32kB/s    0:00:01 (xfr#41, ir-chk=1162/1209)
# test/images/human_oesophagus_31.png
#         405,555 100%  651.40kB/s    0:00:00 (xfr#42, ir-chk=1161/1209)
# test/images/human_oesophagus_34.png
#         391,617 100%  397.55kB/s    0:00:00 (xfr#43, ir-chk=1160/1209)
# test/images/human_oesophagus_38.png
#         401,400 100%  302.93kB/s    0:00:01 (xfr#44, ir-chk=1159/1209)
# test/images/human_oesophagus_42.png
#         393,923 100%  623.49kB/s    0:00:00 (xfr#45, ir-chk=1158/1209)
# test/images/human_oesophagus_46.png
#         405,820 100%  414.55kB/s    0:00:00 (xfr#46, ir-chk=1157/1209)
# test/images/human_pancreas_06.png
#         401,040 100%  305.02kB/s    0:00:01 (xfr#47, ir-chk=1156/1209)
# test/images/human_pancreas_12.png
#         404,611 100%  699.34kB/s    0:00:00 (xfr#48, ir-chk=1155/1209)
# test/images/human_pancreas_14.png
#         394,603 100%  431.05kB/s    0:00:00 (xfr#49, ir-chk=1154/1209)
# test/images/human_pancreas_15.png
#         399,864 100%  278.53kB/s    0:00:01 (xfr#50, ir-chk=1153/1209)
# test/images/human_pancreas_20.png
#         401,072 100%  608.19kB/s    0:00:00 (xfr#51, ir-chk=1152/1209)
# test/images/human_pancreas_23.png
#         401,137 100%  399.32kB/s    0:00:00 (xfr#52, ir-chk=1151/1209)
# test/images/human_pancreas_30.png
#         397,537 100%  289.50kB/s    0:00:01 (xfr#53, ir-chk=1150/1209)
# test/images/human_pancreas_37.png
#         396,961 100%  623.24kB/s    0:00:00 (xfr#54, ir-chk=1149/1209)
# test/images/human_peritoneum_2.png
#         418,859 100%  419.53kB/s    0:00:00 (xfr#55, ir-chk=1148/1209)
# test/images/human_peritoneum_3.png
#         423,373 100%  310.63kB/s    0:00:01 (xfr#56, ir-chk=1147/1209)
# test/images/human_peritoneum_5.png
#         428,053 100%  767.01kB/s    0:00:00 (xfr#57, ir-chk=1146/1209)
# test/images/human_placenta_02.png
#         424,963 100%  473.21kB/s    0:00:00 (xfr#58, ir-chk=1145/1209)
# test/images/human_placenta_05.png
#         409,958 100%  341.30kB/s    0:00:01 (xfr#59, ir-chk=1144/1209)
# test/images/human_placenta_06.png
#         419,485 100%    1.02MB/s    0:00:00 (xfr#60, ir-chk=1143/1209)
# test/images/human_placenta_10.png
#         420,074 100%  519.28kB/s    0:00:00 (xfr#61, ir-chk=1142/1209)
# test/images/human_placenta_17.png
#         419,592 100%  347.84kB/s    0:00:01 (xfr#62, ir-chk=1141/1209)
# test/images/human_placenta_24.png
#         412,979 100%    1.10MB/s    0:00:00 (xfr#63, ir-chk=1140/1209)
# test/images/human_pylorus_10.png
#         423,859 100%  553.38kB/s    0:00:00 (xfr#64, ir-chk=1139/1209)
# test/images/human_pylorus_3.png
#         416,402 100%  348.15kB/s    0:00:01 (xfr#65, ir-chk=1138/1209)
# test/images/human_pylorus_9.png
#         411,090 100%    1.41MB/s    0:00:00 (xfr#66, ir-chk=1137/1209)
# test/images/human_rectum_5.png
#         427,825 100%  715.41kB/s    0:00:00 (xfr#67, ir-chk=1136/1209)
# test/images/human_salivory_01.png
#         405,453 100%  415.91kB/s    0:00:00 (xfr#68, ir-chk=1135/1209)
# test/images/human_salivory_09.png
#         412,168 100%  313.24kB/s    0:00:01 (xfr#69, ir-chk=1134/1209)
# test/images/human_salivory_11.png
#         410,873 100%  716.51kB/s    0:00:00 (xfr#70, ir-chk=1133/1209)
# test/images/human_salivory_16.png
#         419,845 100%  433.87kB/s    0:00:00 (xfr#71, ir-chk=1132/1209)
# test/images/human_salivory_18.png
#         405,896 100%  287.65kB/s    0:00:01 (xfr#72, ir-chk=1131/1209)
# test/images/human_salivory_20.png
#         405,201 100%  599.55kB/s    0:00:00 (xfr#73, ir-chk=1130/1209)
# test/images/human_salivory_22.png
#         405,928 100%  378.26kB/s    0:00:01 (xfr#74, ir-chk=1129/1209)
# test/images/human_salivory_23.png
#         412,712 100%    1.31MB/s    0:00:00 (xfr#75, ir-chk=1128/1209)
# test/images/human_salivory_26.png
#         417,190 100%  594.76kB/s    0:00:00 (xfr#76, ir-chk=1127/1209)
# test/images/human_salivory_32.png
#         406,914 100%  365.24kB/s    0:00:01 (xfr#77, ir-chk=1126/1209)
# test/images/human_salivory_34.png
#         403,278 100%  989.51kB/s    0:00:00 (xfr#78, ir-chk=1125/1209)
# test/images/human_salivory_35.png
#         401,442 100%  461.76kB/s    0:00:00 (xfr#79, ir-chk=1124/1209)
# test/images/human_salivory_37.png
#         417,741 100%  341.67kB/s    0:00:01 (xfr#80, ir-chk=1123/1209)
# test/images/human_salivory_39.png
#         403,608 100%    1.01MB/s    0:00:00 (xfr#81, ir-chk=1122/1209)
# test/images/human_salivory_42.png
#         408,328 100%  553.06kB/s    0:00:00 (xfr#82, ir-chk=1121/1209)
# test/images/human_salivory_43.png
#         404,915 100%  368.18kB/s    0:00:01 (xfr#83, ir-chk=1120/1209)
# test/images/human_spleen_01.png
#         432,273 100%    1.09MB/s    0:00:00 (xfr#84, ir-chk=1119/1209)
# test/images/human_spleen_13.png
#         391,811 100%  537.40kB/s    0:00:00 (xfr#85, ir-chk=1118/1209)
# test/images/human_spleen_15.png
#         399,257 100%  366.10kB/s    0:00:01 (xfr#86, ir-chk=1117/1209)
# test/images/human_spleen_18.png
#         396,720 100%    1.07MB/s    0:00:00 (xfr#87, ir-chk=1116/1209)
# test/images/human_spleen_26.png
#         399,001 100%  571.33kB/s    0:00:00 (xfr#88, ir-chk=1115/1209)
# test/images/human_testis_9.png
#         440,114 100%  388.26kB/s    0:00:01 (xfr#89, ir-chk=1114/1209)
# test/images/human_tongue_16.png
#         398,080 100%    1.24MB/s    0:00:00 (xfr#90, ir-chk=1113/1209)
# test/images/human_tongue_21.png
#         391,761 100%  502.07kB/s    0:00:00 (xfr#91, ir-chk=1112/1209)
# test/images/human_tongue_24.png
#         382,685 100%  352.90kB/s    0:00:01 (xfr#92, ir-chk=1111/1209)
# test/images/human_tongue_37.png
#         387,364 100%  974.96kB/s    0:00:00 (xfr#93, ir-chk=1110/1209)
# test/images/human_tongue_38.png
#         385,005 100%  492.12kB/s    0:00:00 (xfr#94, ir-chk=1109/1209)
# test/images/human_tonsile_12.png
#         426,609 100%  396.39kB/s    0:00:01 (xfr#95, ir-chk=1108/1209)
# test/images/human_umbilical_cord_04.png
#         432,466 100%  844.66kB/s    0:00:00 (xfr#96, ir-chk=1107/1209)
# test/images/mouse_femur_06.png
#         431,917 100%  438.00kB/s    0:00:00 (xfr#97, ir-chk=1106/1209)
# test/images/mouse_heart_16.png
#         419,445 100%  303.64kB/s    0:00:01 (xfr#98, ir-chk=1105/1209)
# test/images/mouse_heart_22.png
#         417,252 100%  612.74kB/s    0:00:00 (xfr#99, ir-chk=1104/1209)
# test/images/mouse_heart_26.png
#         418,636 100%  355.50kB/s    0:00:01 (xfr#100, ir-chk=1103/1209)
# test/images/mouse_kidney_01.png
#         420,512 100%  991.92kB/s    0:00:00 (xfr#101, ir-chk=1102/1209)
# test/images/mouse_kidney_02.png
#         419,414 100%  488.76kB/s    0:00:00 (xfr#102, ir-chk=1101/1209)
# test/images/mouse_kidney_03.png
#         423,980 100%  302.00kB/s    0:00:01 (xfr#103, ir-chk=1100/1209)
# test/images/mouse_kidney_09.png
#         419,623 100%    1.12MB/s    0:00:00 (xfr#104, ir-chk=1099/1209)
# test/images/mouse_kidney_24.png
#         422,004 100%  564.54kB/s    0:00:00 (xfr#105, ir-chk=1098/1209)
# test/images/mouse_kidney_30.png
#         412,849 100%  334.86kB/s    0:00:01 (xfr#106, ir-chk=1097/1209)
# test/images/mouse_kidney_32.png
#         419,857 100%  755.10kB/s    0:00:00 (xfr#107, ir-chk=1096/1209)
# test/images/mouse_kidney_37.png
#         416,077 100%  410.84kB/s    0:00:00 (xfr#108, ir-chk=1095/1209)
# test/images/mouse_kidney_39.png
#         418,049 100%  282.92kB/s    0:00:01 (xfr#109, ir-chk=1094/1209)
# test/images/mouse_liver_02.png
#         418,345 100%  588.67kB/s    0:00:00 (xfr#110, ir-chk=1093/1209)
# test/images/mouse_liver_08.png
#         413,598 100%  344.04kB/s    0:00:01 (xfr#111, ir-chk=1092/1209)
# test/images/mouse_liver_09.png
#         413,602 100%  878.06kB/s    0:00:00 (xfr#112, ir-chk=1091/1209)
# test/images/mouse_liver_15.png
#         412,174 100%  490.27kB/s    0:00:00 (xfr#113, ir-chk=1090/1209)
# test/images/mouse_liver_22.png
#         416,002 100%  358.25kB/s    0:00:01 (xfr#114, ir-chk=1089/1209)
# test/images/mouse_liver_30.png
#         415,509 100%  765.60kB/s    0:00:00 (xfr#115, ir-chk=1088/1209)
# test/images/mouse_liver_31.png
#         416,477 100%  401.50kB/s    0:00:01 (xfr#116, ir-chk=1087/1209)
# test/images/mouse_liver_36.png
#         413,777 100%    1.30MB/s    0:00:00 (xfr#117, ir-chk=1086/1209)
# test/images/mouse_muscle_tibia_01.png
#         409,755 100%  551.93kB/s    0:00:00 (xfr#118, ir-chk=1085/1209)
# test/images/mouse_muscle_tibia_05.png
#         411,354 100%  391.15kB/s    0:00:01 (xfr#119, ir-chk=1084/1209)
# test/images/mouse_muscle_tibia_11.png
#         414,004 100%    1.39MB/s    0:00:00 (xfr#120, ir-chk=1083/1209)
# test/images/mouse_muscle_tibia_12.png
#         414,293 100%  442.17kB/s    0:00:00 (xfr#121, ir-chk=1082/1209)
# test/images/mouse_muscle_tibia_25.png
#         410,165 100%  312.93kB/s    0:00:01 (xfr#122, ir-chk=1081/1209)
# test/images/mouse_muscle_tibia_27.png
#         413,182 100%  655.03kB/s    0:00:00 (xfr#123, ir-chk=1080/1209)
# test/images/mouse_muscle_tibia_28.png
#         411,516 100%  415.59kB/s    0:00:00 (xfr#124, ir-chk=1079/1209)
# test/images/mouse_spleen_06.png
#         408,846 100%  303.62kB/s    0:00:01 (xfr#125, ir-chk=1078/1209)
# test/images/mouse_subscapula_02.png
#         435,366 100%  774.43kB/s    0:00:00 (xfr#126, ir-chk=1077/1209)
# test/images/mouse_subscapula_06.png
#         435,740 100%  446.98kB/s    0:00:00 (xfr#127, ir-chk=1076/1209)
# test/images/mouse_subscapula_11.png
#         435,368 100%  327.05kB/s    0:00:01 (xfr#128, ir-chk=1075/1209)
# test/images/mouse_subscapula_19.png
#         430,778 100%  710.61kB/s    0:00:00 (xfr#129, ir-chk=1074/1209)
# test/images/mouse_subscapula_35.png
#         433,065 100%  443.77kB/s    0:00:00 (xfr#130, ir-chk=1073/1209)
# test/images/mouse_subscapula_40.png
#         425,373 100%  308.39kB/s    0:00:01 (xfr#131, ir-chk=1072/1209)
# test/images/mouse_thymus_02.png
#         404,752 100%  644.81kB/s    0:00:00 (xfr#132, ir-chk=1071/1209)
# test/images/mouse_thymus_03.png
#         398,883 100%  395.07kB/s    0:00:00 (xfr#133, ir-chk=1070/1209)
# test/labels/
# test/labels/human_bladder_01.txt
#          28,236 100%   21.48kB/s    0:00:01 (xfr#134, to-chk=1202/1342)
# test/labels/human_bladder_11.txt
#          21,398 100%   14.32kB/s    0:00:01 (xfr#135, to-chk=1201/1342)
# test/labels/human_bladder_12.txt
#          13,448 100%    0.00kB/s    0:00:00 (xfr#136, to-chk=1200/1342)
# test/labels/human_brain_12.txt
#           4,476 100%   53.96kB/s    0:00:00 (xfr#137, to-chk=1199/1342)
# rsync: [generator] failed to set times on "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/val/labels": Operation not permitted (1)
# test/labels/human_brain_8.txt
#          17,984 100%  110.46kB/s    0:00:00 (xfr#138, to-chk=1198/1342)
# test/labels/human_brain_9.txt
#           6,894 100%   29.14kB/s    0:00:00 (xfr#139, to-chk=1197/1342)
# test/labels/human_cardia_1.txt
#          21,982 100%   73.27kB/s    0:00:00 (xfr#140, to-chk=1196/1342)
# test/labels/human_cardia_3.txt
#          86,550 100%   67.24kB/s    0:00:01 (xfr#141, to-chk=1195/1342)
# test/labels/human_cardia_5.txt
#          63,542 100%  805.88kB/s    0:00:00 (xfr#142, to-chk=1194/1342)
# test/labels/human_cerebellum_4.txt
#           7,952 100%   35.95kB/s    0:00:00 (xfr#143, to-chk=1193/1342)
# test/labels/human_cerebellum_5.txt
#         112,514 100%  248.03kB/s    0:00:00 (xfr#144, to-chk=1192/1342)
# test/labels/human_cerebellum_6.txt
#          49,208 100%   55.68kB/s    0:00:00 (xfr#145, to-chk=1191/1342)
# test/labels/human_epiglottis_10.txt
#          26,982 100%   28.46kB/s    0:00:00 (xfr#146, to-chk=1190/1342)
# test/labels/human_epiglottis_4.txt
#           9,204 100%    9.06kB/s    0:00:00 (xfr#147, to-chk=1189/1342)
# test/labels/human_epiglottis_6.txt
#          25,134 100%   23.51kB/s    0:00:01 (xfr#148, to-chk=1188/1342)
# test/labels/human_jejunum_02.txt
#          88,394 100%   57.90kB/s    0:00:01 (xfr#149, to-chk=1187/1342)
# test/labels/human_jejunum_03.txt
#          92,484 100%  212.51kB/s    0:00:00 (xfr#150, to-chk=1186/1342)
# test/labels/human_jejunum_05.txt
#          50,866 100%   98.95kB/s    0:00:00 (xfr#151, to-chk=1185/1342)
# test/labels/human_jejunum_08.txt
#          78,642 100%   90.46kB/s    0:00:00 (xfr#152, to-chk=1184/1342)
# test/labels/human_kidney_02.txt
#          72,748 100%   68.05kB/s    0:00:01 (xfr#153, to-chk=1183/1342)
# test/labels/human_kidney_07.txt
#         112,060 100%  338.80kB/s    0:00:00 (xfr#154, to-chk=1182/1342)
# test/labels/human_kidney_08.txt
#         129,152 100%  229.32kB/s    0:00:00 (xfr#155, to-chk=1181/1342)
# test/labels/human_liver_02.txt
#          44,698 100%   54.43kB/s    0:00:00 (xfr#156, to-chk=1180/1342)
# test/labels/human_liver_03.txt
#          37,350 100%   37.68kB/s    0:00:00 (xfr#157, to-chk=1179/1342)
# test/labels/human_liver_04.txt
#          39,604 100%   37.26kB/s    0:00:01 (xfr#158, to-chk=1178/1342)
# test/labels/human_liver_09.txt
#          29,560 100%  481.12kB/s    0:00:00 (xfr#159, to-chk=1177/1342)
# test/labels/human_liver_10.txt
#          24,058 100%   98.71kB/s    0:00:00 (xfr#160, to-chk=1176/1342)
# test/labels/human_liver_11.txt
#          34,214 100%  112.50kB/s    0:00:00 (xfr#161, to-chk=1175/1342)
# test/labels/human_liver_13.txt
#          31,124 100%   83.50kB/s    0:00:00 (xfr#162, to-chk=1174/1342)
# test/labels/human_liver_22.txt
#          31,232 100%   35.26kB/s    0:00:00 (xfr#163, to-chk=1173/1342)
# test/labels/human_liver_29.txt
#          13,726 100%   14.43kB/s    0:00:00 (xfr#164, to-chk=1172/1342)
# test/labels/human_liver_31.txt
#          30,946 100%   27.03kB/s    0:00:01 (xfr#165, to-chk=1171/1342)
# test/labels/human_liver_38.txt
#          54,874 100%   45.22kB/s    0:00:01 (xfr#166, to-chk=1170/1342)
# test/labels/human_melanoma_01.txt
#          27,038 100%  377.20kB/s    0:00:00 (xfr#167, to-chk=1169/1342)
# test/labels/human_melanoma_02.txt
#          21,468 100%   22.33kB/s    0:00:00 (xfr#168, to-chk=1168/1342)
# test/labels/human_melanoma_03.txt
#          37,278 100%   35.80kB/s    0:00:01 (xfr#169, to-chk=1167/1342)
# test/labels/human_melanoma_05.txt
#          30,174 100%  460.42kB/s    0:00:00 (xfr#170, to-chk=1166/1342)
# test/labels/human_muscle_6.txt
#           5,704 100%   39.51kB/s    0:00:00 (xfr#171, to-chk=1165/1342)
# test/labels/human_oesophagus_02.txt
#          54,046 100%  109.27kB/s    0:00:00 (xfr#172, to-chk=1164/1342)
# test/labels/human_oesophagus_04.txt
#          38,972 100%   66.07kB/s    0:00:00 (xfr#173, to-chk=1163/1342)
# test/labels/human_oesophagus_12.txt
#          18,154 100%   27.53kB/s    0:00:00 (xfr#174, to-chk=1162/1342)
# test/labels/human_oesophagus_31.txt
#          67,390 100%   54.03kB/s    0:00:01 (xfr#175, to-chk=1161/1342)
# test/labels/human_oesophagus_34.txt
#          81,594 100%  189.27kB/s    0:00:00 (xfr#176, to-chk=1160/1342)
# test/labels/human_oesophagus_38.txt
#          34,138 100%   63.62kB/s    0:00:00 (xfr#177, to-chk=1159/1342)
# test/labels/human_oesophagus_42.txt
#         102,886 100%  112.51kB/s    0:00:00 (xfr#178, to-chk=1158/1342)
# test/labels/human_oesophagus_46.txt
#          68,774 100%   62.19kB/s    0:00:01 (xfr#179, to-chk=1157/1342)
# test/labels/human_pancreas_06.txt
#          48,096 100%  733.89kB/s    0:00:00 (xfr#180, to-chk=1156/1342)
# test/labels/human_pancreas_12.txt
#          41,248 100%  206.57kB/s    0:00:00 (xfr#181, to-chk=1155/1342)
# test/labels/human_pancreas_14.txt
#          34,792 100%  138.12kB/s    0:00:00 (xfr#182, to-chk=1154/1342)
# test/labels/human_pancreas_15.txt
#          27,920 100%   84.68kB/s    0:00:00 (xfr#183, to-chk=1153/1342)
# test/labels/human_pancreas_20.txt
#          34,316 100%   85.93kB/s    0:00:00 (xfr#184, to-chk=1152/1342)
# test/labels/human_pancreas_23.txt
#          42,156 100%   48.38kB/s    0:00:00 (xfr#185, to-chk=1151/1342)
# test/labels/human_pancreas_30.txt
#          51,358 100%   54.46kB/s    0:00:00 (xfr#186, to-chk=1150/1342)
# test/labels/human_pancreas_37.txt
#          26,626 100%   21.02kB/s    0:00:01 (xfr#187, to-chk=1149/1342)
# test/labels/human_peritoneum_2.txt
#          25,736 100%   19.19kB/s    0:00:01 (xfr#188, to-chk=1148/1342)
# test/labels/human_peritoneum_3.txt
#          54,538 100%   20.76MB/s    0:00:00 (xfr#189, to-chk=1147/1342)
# test/labels/human_peritoneum_5.txt
#          32,300 100%   87.14kB/s    0:00:00 (xfr#190, to-chk=1146/1342)
# test/labels/human_placenta_02.txt
#          35,536 100%   66.10kB/s    0:00:00 (xfr#191, to-chk=1145/1342)
# test/labels/human_placenta_05.txt
#          34,640 100%   57.14kB/s    0:00:00 (xfr#192, to-chk=1144/1342)
# test/labels/human_placenta_06.txt
#          62,774 100%   67.51kB/s    0:00:00 (xfr#193, to-chk=1143/1342)
# test/labels/human_placenta_10.txt
#          47,338 100%   47.27kB/s    0:00:00 (xfr#194, to-chk=1142/1342)
# test/labels/human_placenta_17.txt
#          40,544 100%   31.05kB/s    0:00:01 (xfr#195, to-chk=1141/1342)
# test/labels/human_placenta_24.txt
#          70,470 100%  337.34kB/s    0:00:00 (xfr#196, to-chk=1140/1342)
# test/labels/human_pylorus_10.txt
#          49,620 100%  122.37kB/s    0:00:00 (xfr#197, to-chk=1139/1342)
# test/labels/human_pylorus_3.txt
#          27,588 100%   44.38kB/s    0:00:00 (xfr#198, to-chk=1138/1342)
# test/labels/human_pylorus_9.txt
#          25,506 100%   34.89kB/s    0:00:00 (xfr#199, to-chk=1137/1342)
# test/labels/human_rectum_5.txt
#          37,438 100%   47.54kB/s    0:00:00 (xfr#200, to-chk=1136/1342)
# test/labels/human_salivory_01.txt
#          63,598 100%   50.53kB/s    0:00:01 (xfr#201, to-chk=1135/1342)
# test/labels/human_salivory_09.txt
#          30,388 100%  345.07kB/s    0:00:00 (xfr#202, to-chk=1134/1342)
# test/labels/human_salivory_11.txt
#          55,052 100%  140.37kB/s    0:00:00 (xfr#203, to-chk=1133/1342)
# test/labels/human_salivory_16.txt
#          33,984 100%   70.46kB/s    0:00:00 (xfr#204, to-chk=1132/1342)
# test/labels/human_salivory_18.txt
#          67,268 100%   64.47kB/s    0:00:01 (xfr#205, to-chk=1131/1342)
# test/labels/human_salivory_20.txt
#          60,022 100%  658.60kB/s    0:00:00 (xfr#206, to-chk=1130/1342)
# test/labels/human_salivory_22.txt
#          53,280 100%  278.24kB/s    0:00:00 (xfr#207, to-chk=1129/1342)
# test/labels/human_salivory_23.txt
#          76,630 100%  116.02kB/s    0:00:00 (xfr#208, to-chk=1128/1342)
# test/labels/human_salivory_26.txt
#          10,598 100%   14.28kB/s    0:00:00 (xfr#209, to-chk=1127/1342)
# test/labels/human_salivory_32.txt
#          71,150 100%   77.12kB/s    0:00:00 (xfr#210, to-chk=1126/1342)
# test/labels/human_salivory_34.txt
#          79,046 100%   53.02kB/s    0:00:01 (xfr#211, to-chk=1125/1342)
# test/labels/human_salivory_35.txt
#          53,140 100%  283.58kB/s    0:00:00 (xfr#212, to-chk=1124/1342)
# test/labels/human_salivory_37.txt
#          64,694 100%  244.87kB/s    0:00:00 (xfr#213, to-chk=1123/1342)
# test/labels/human_salivory_39.txt
#          62,492 100%  152.57kB/s    0:00:00 (xfr#214, to-chk=1122/1342)
# test/labels/human_salivory_42.txt
#          63,404 100%  104.59kB/s    0:00:00 (xfr#215, to-chk=1121/1342)
# test/labels/human_salivory_43.txt
#          65,840 100%   66.70kB/s    0:00:00 (xfr#216, to-chk=1120/1342)
# test/labels/human_spleen_01.txt
#          33,352 100%   31.14kB/s    0:00:01 (xfr#217, to-chk=1119/1342)
# test/labels/human_spleen_13.txt
#          40,694 100%  467.53kB/s    0:00:00 (xfr#218, to-chk=1118/1342)
# test/labels/human_spleen_15.txt
#          55,192 100%  361.73kB/s    0:00:00 (xfr#219, to-chk=1117/1342)
# test/labels/human_spleen_18.txt
#          53,892 100%  200.87kB/s    0:00:00 (xfr#220, to-chk=1116/1342)
# test/labels/human_spleen_26.txt
#         158,582 100%  137.17kB/s    0:00:01 (xfr#221, to-chk=1115/1342)
# test/labels/human_testis_9.txt
#          34,874 100%  199.16kB/s    0:00:00 (xfr#222, to-chk=1114/1342)
# test/labels/human_tongue_16.txt
#          21,442 100%   81.48kB/s    0:00:00 (xfr#223, to-chk=1113/1342)
# test/labels/human_tongue_21.txt
#          32,240 100%   90.47kB/s    0:00:00 (xfr#224, to-chk=1112/1342)
# test/labels/human_tongue_24.txt
#          34,914 100%   73.64kB/s    0:00:00 (xfr#225, to-chk=1111/1342)
# test/labels/human_tongue_37.txt
#          18,000 100%   32.98kB/s    0:00:00 (xfr#226, to-

# ... [*** WARNING: max output size exceeded, skipping output. ***] ...

# 0%  580.90kB/s    0:00:00 (xfr#1103, to-chk=229/1342)
# val/images/human_oesophagus_26.png
#         406,124 100%  398.60kB/s    0:00:00 (xfr#1104, to-chk=228/1342)
# val/images/human_oesophagus_27.png
#         407,873 100%  303.82kB/s    0:00:01 (xfr#1105, to-chk=227/1342)
# val/images/human_oesophagus_28.png
#         401,907 100%  753.33kB/s    0:00:00 (xfr#1106, to-chk=226/1342)
# val/images/human_oesophagus_32.png
#         406,042 100%  457.88kB/s    0:00:00 (xfr#1107, to-chk=225/1342)
# val/images/human_pancreas_01.png
#         403,611 100%  300.19kB/s    0:00:01 (xfr#1108, to-chk=224/1342)
# val/images/human_pancreas_05.png
#         402,350 100%  665.97kB/s    0:00:00 (xfr#1109, to-chk=223/1342)
# val/images/human_pancreas_10.png
#         404,154 100%  376.25kB/s    0:00:01 (xfr#1110, to-chk=222/1342)
# val/images/human_pancreas_11.png
#         404,839 100%    1.18MB/s    0:00:00 (xfr#1111, to-chk=221/1342)
# val/images/human_pancreas_13.png
#         399,340 100%  572.66kB/s    0:00:00 (xfr#1112, to-chk=220/1342)
# val/images/human_pancreas_17.png
#         400,201 100%  355.29kB/s    0:00:01 (xfr#1113, to-chk=219/1342)
# val/images/human_pancreas_22.png
#         400,327 100%    1.19MB/s    0:00:00 (xfr#1114, to-chk=218/1342)
# val/images/human_pancreas_29.png
#         400,671 100%  618.14kB/s    0:00:00 (xfr#1115, to-chk=217/1342)
# val/images/human_pancreas_33.png
#         400,648 100%  384.34kB/s    0:00:01 (xfr#1116, to-chk=216/1342)
# val/images/human_pancreas_36.png
#         404,027 100% 1009.10kB/s    0:00:00 (xfr#1117, to-chk=215/1342)
# val/images/human_pancreas_38.png
#         404,601 100%  529.65kB/s    0:00:00 (xfr#1118, to-chk=214/1342)
# val/images/human_pancreas_40.png
#         408,239 100%  300.43kB/s    0:00:01 (xfr#1119, to-chk=213/1342)
# val/images/human_peritoneum_10.png
#         427,950 100%    1.09MB/s    0:00:00 (xfr#1120, to-chk=212/1342)
# val/images/human_peritoneum_4.png
#         419,897 100%  551.15kB/s    0:00:00 (xfr#1121, to-chk=211/1342)
# val/images/human_placenta_03.png
#         425,531 100%  354.87kB/s    0:00:01 (xfr#1122, to-chk=210/1342)
# val/images/human_placenta_11.png
#         422,499 100%    1.39MB/s    0:00:00 (xfr#1123, to-chk=209/1342)
# val/images/human_placenta_20.png
#         407,584 100%  650.38kB/s    0:00:00 (xfr#1124, to-chk=208/1342)
# val/images/human_placenta_21.png
#         426,801 100%  412.67kB/s    0:00:01 (xfr#1125, to-chk=207/1342)
# val/images/human_placenta_23.png
#         418,174 100%    1.46MB/s    0:00:00 (xfr#1126, to-chk=206/1342)
# val/images/human_placenta_27.png
#         415,281 100%  710.24kB/s    0:00:00 (xfr#1127, to-chk=205/1342)
# val/images/human_placenta_30.png
#         414,868 100%  423.35kB/s    0:00:00 (xfr#1128, to-chk=204/1342)
# val/images/human_placenta_32.png
#         416,159 100%  321.27kB/s    0:00:01 (xfr#1129, to-chk=203/1342)
# val/images/human_placenta_35.png
#         406,568 100%  637.30kB/s    0:00:00 (xfr#1130, to-chk=202/1342)
# val/images/human_placenta_36.png
#         410,552 100%  410.79kB/s    0:00:00 (xfr#1131, to-chk=201/1342)
# val/images/human_placenta_37.png
#         419,096 100%  322.01kB/s    0:00:01 (xfr#1132, to-chk=200/1342)
# val/images/human_placenta_38.png
#         425,224 100%  763.34kB/s    0:00:00 (xfr#1133, to-chk=199/1342)
# val/images/human_pylorus_7.png
#         406,571 100%  468.76kB/s    0:00:00 (xfr#1134, to-chk=198/1342)
# val/images/human_rectum_2.png
#         423,060 100%  363.04kB/s    0:00:01 (xfr#1135, to-chk=197/1342)
# val/images/human_rectum_3.png
#         428,735 100%  902.34kB/s    0:00:00 (xfr#1136, to-chk=196/1342)
# val/images/human_rectum_9.png
#         431,008 100%  527.45kB/s    0:00:00 (xfr#1137, to-chk=195/1342)
# val/images/human_salivory_03.png
#         404,857 100%  347.42kB/s    0:00:01 (xfr#1138, to-chk=194/1342)
# val/images/human_salivory_10.png
#         406,599 100%    1.23MB/s    0:00:00 (xfr#1139, to-chk=193/1342)
# val/images/human_salivory_14.png
#         404,413 100%  622.93kB/s    0:00:00 (xfr#1140, to-chk=192/1342)
# val/images/human_salivory_15.png
#         420,100 100%  334.63kB/s    0:00:01 (xfr#1141, to-chk=191/1342)
# val/images/human_salivory_17.png
#         405,891 100%    1.36MB/s    0:00:00 (xfr#1142, to-chk=190/1342)
# val/images/human_salivory_28.png
#         405,809 100%  545.86kB/s    0:00:00 (xfr#1143, to-chk=189/1342)
# val/images/human_salivory_31.png
#         403,274 100%  379.40kB/s    0:00:01 (xfr#1144, to-chk=188/1342)
# val/images/human_salivory_36.png
#         405,849 100%    1.17MB/s    0:00:00 (xfr#1145, to-chk=187/1342)
# val/images/human_salivory_44.png
#         406,632 100%  469.94kB/s    0:00:00 (xfr#1146, to-chk=186/1342)
# val/images/human_spleen_07.png
#         433,055 100%  259.61kB/s    0:00:01 (xfr#1147, to-chk=185/1342)
# val/images/human_spleen_28.png
#         404,721 100%    1.29MB/s    0:00:00 (xfr#1148, to-chk=184/1342)
# val/images/human_spleen_31.png
#         400,313 100%  568.21kB/s    0:00:00 (xfr#1149, to-chk=183/1342)
# val/images/human_testis_10.png
#         438,385 100%  390.61kB/s    0:00:01 (xfr#1150, to-chk=182/1342)
# val/images/human_testis_11.png
#         434,987 100%    1.12MB/s    0:00:00 (xfr#1151, to-chk=181/1342)
# val/images/human_tongue_01.png
#         390,876 100%  540.67kB/s    0:00:00 (xfr#1152, to-chk=180/1342)
# val/images/human_tongue_09.png
#         384,115 100%  362.78kB/s    0:00:01 (xfr#1153, to-chk=179/1342)
# val/images/human_tongue_14.png
#         384,820 100%    1.21MB/s    0:00:00 (xfr#1154, to-chk=178/1342)
# val/images/human_tongue_22.png
#         399,302 100%  634.05kB/s    0:00:00 (xfr#1155, to-chk=177/1342)
# val/images/human_tongue_28.png
#         369,476 100%  376.24kB/s    0:00:00 (xfr#1156, to-chk=176/1342)
# val/images/human_tongue_29.png
#         382,009 100%  302.07kB/s    0:00:01 (xfr#1157, to-chk=175/1342)
# val/images/human_tongue_35.png
#         383,185 100%  789.46kB/s    0:00:00 (xfr#1158, to-chk=174/1342)
# val/images/human_tongue_36.png
#         393,406 100%  473.13kB/s    0:00:00 (xfr#1159, to-chk=173/1342)
# val/images/human_tongue_39.png
#         379,772 100%  349.88kB/s    0:00:01 (xfr#1160, to-chk=172/1342)
# val/images/human_tonsile_10.png
#         428,954 100%    1.06MB/s    0:00:00 (xfr#1161, to-chk=171/1342)
# val/images/human_tonsile_11.png
#         439,545 100%  504.40kB/s    0:00:00 (xfr#1162, to-chk=170/1342)
# val/images/human_umbilical_cord_06.png
#         429,497 100%  370.85kB/s    0:00:01 (xfr#1163, to-chk=169/1342)
# val/images/human_umbilical_cord_09.png
#         429,791 100%    1.29MB/s    0:00:00 (xfr#1164, to-chk=168/1342)
# val/images/human_umbilical_cord_11.png
#         433,386 100%  669.67kB/s    0:00:00 (xfr#1165, to-chk=167/1342)
# val/images/mouse_femur_01.png
#         424,441 100%  424.25kB/s    0:00:00 (xfr#1166, to-chk=166/1342)
# val/images/mouse_femur_02.png
#         428,482 100%  317.48kB/s    0:00:01 (xfr#1167, to-chk=165/1342)
# val/images/mouse_femur_03.png
#         423,038 100%  853.56kB/s    0:00:00 (xfr#1168, to-chk=164/1342)
# val/images/mouse_heart_02.png
#         418,478 100%  550.77kB/s    0:00:00 (xfr#1169, to-chk=163/1342)
# val/images/mouse_heart_09.png
#         418,548 100%  393.77kB/s    0:00:01 (xfr#1170, to-chk=162/1342)
# val/images/mouse_heart_10.png
#         420,366 100%    1.21MB/s    0:00:00 (xfr#1171, to-chk=161/1342)
# val/images/mouse_heart_11.png
#         417,688 100%  572.09kB/s    0:00:00 (xfr#1172, to-chk=160/1342)
# val/images/mouse_heart_13.png
#         420,571 100%  411.95kB/s    0:00:00 (xfr#1173, to-chk=159/1342)
# val/images/mouse_heart_15.png
#         417,149 100%  304.69kB/s    0:00:01 (xfr#1174, to-chk=158/1342)
# val/images/mouse_heart_17.png
#         421,061 100%  736.90kB/s    0:00:00 (xfr#1175, to-chk=157/1342)
# val/images/mouse_heart_28.png
#         421,169 100%  489.06kB/s    0:00:00 (xfr#1176, to-chk=156/1342)
# val/images/mouse_kidney_04.png
#         420,437 100%  364.32kB/s    0:00:01 (xfr#1177, to-chk=155/1342)
# val/images/mouse_kidney_20.png
#         411,785 100%    1.19MB/s    0:00:00 (xfr#1178, to-chk=154/1342)
# val/images/mouse_kidney_26.png
#         420,539 100%  587.53kB/s    0:00:00 (xfr#1179, to-chk=153/1342)
# val/images/mouse_kidney_31.png
#         419,879 100%  411.68kB/s    0:00:00 (xfr#1180, to-chk=152/1342)
# val/images/mouse_kidney_34.png
#         422,561 100%  328.81kB/s    0:00:01 (xfr#1181, to-chk=151/1342)
# val/images/mouse_liver_07.png
#         414,764 100%  626.03kB/s    0:00:00 (xfr#1182, to-chk=150/1342)
# val/images/mouse_liver_12.png
#         415,390 100%  416.06kB/s    0:00:00 (xfr#1183, to-chk=149/1342)
# val/images/mouse_liver_14.png
#         412,300 100%  286.78kB/s    0:00:01 (xfr#1184, to-chk=148/1342)
# val/images/mouse_liver_17.png
#         410,066 100%  527.61kB/s    0:00:00 (xfr#1185, to-chk=147/1342)
# val/images/mouse_liver_20.png
#         417,713 100%  329.50kB/s    0:00:01 (xfr#1186, to-chk=146/1342)
# val/images/mouse_liver_32.png
#         411,302 100%    1.15MB/s    0:00:00 (xfr#1187, to-chk=145/1342)
# val/images/mouse_liver_35.png
#         412,143 100%  599.83kB/s    0:00:00 (xfr#1188, to-chk=144/1342)
# val/images/mouse_muscle_tibia_04.png
#         412,762 100%  407.98kB/s    0:00:00 (xfr#1189, to-chk=143/1342)
# val/images/mouse_muscle_tibia_09.png
#         407,431 100%  281.19kB/s    0:00:01 (xfr#1190, to-chk=142/1342)
# val/images/mouse_muscle_tibia_13.png
#         413,441 100%  697.32kB/s    0:00:00 (xfr#1191, to-chk=141/1342)
# val/images/mouse_muscle_tibia_16.png
#         408,213 100%  436.63kB/s    0:00:00 (xfr#1192, to-chk=140/1342)
# val/images/mouse_spleen_01.png
#         415,891 100%  329.66kB/s    0:00:01 (xfr#1193, to-chk=139/1342)
# val/images/mouse_spleen_02.png
#         415,540 100%  729.86kB/s    0:00:00 (xfr#1194, to-chk=138/1342)
# val/images/mouse_spleen_05.png
#         409,292 100%  398.11kB/s    0:00:01 (xfr#1195, to-chk=137/1342)
# val/images/mouse_subscapula_14.png
#         431,974 100%    1.40MB/s    0:00:00 (xfr#1196, to-chk=136/1342)
# val/images/mouse_subscapula_15.png
#         429,873 100%  700.83kB/s    0:00:00 (xfr#1197, to-chk=135/1342)
# val/images/mouse_subscapula_28.png
#         433,368 100%  470.76kB/s    0:00:00 (xfr#1198, to-chk=134/1342)
# val/images/mouse_subscapula_36.png
#         428,312 100%  362.77kB/s    0:00:01 (xfr#1199, to-chk=133/1342)
# val/labels/
# val/labels/human_bladder_03.txt
#          21,056 100%  193.99kB/s    0:00:00 (xfr#1200, to-chk=132/1342)
# val/labels/human_bladder_07.txt
#          47,602 100%  153.93kB/s    0:00:00 (xfr#1201, to-chk=131/1342)
# val/labels/human_bladder_08.txt
#          29,970 100%   77.84kB/s    0:00:00 (xfr#1202, to-chk=130/1342)
# val/labels/human_bladder_10.txt
#          17,868 100%   41.15kB/s    0:00:00 (xfr#1203, to-chk=129/1342)
# val/labels/human_brain_4.txt
#           7,884 100%   15.62kB/s    0:00:00 (xfr#1204, to-chk=128/1342)
# val/labels/human_cardia_2.txt
#          42,452 100%   72.48kB/s    0:00:00 (xfr#1205, to-chk=127/1342)
# val/labels/human_cardia_4.txt
#          10,056 100%   15.64kB/s    0:00:00 (xfr#1206, to-chk=126/1342)
# val/labels/human_cardia_7.txt
#          15,932 100%   13.67kB/s    0:00:01 (xfr#1207, to-chk=125/1342)
# val/labels/human_cerebellum_12.txt
#           4,708 100%    3.74kB/s    0:00:01 (xfr#1208, to-chk=124/1342)
# val/labels/human_cerebellum_3.txt
#           8,426 100%    6.34kB/s    0:00:01 (xfr#1209, to-chk=123/1342)
# val/labels/human_epiglottis_5.txt
#             976 100%    0.70kB/s    0:00:01 (xfr#1210, to-chk=122/1342)
# val/labels/human_jejunum_07.txt
#         101,330 100%   37.08kB/s    0:00:02 (xfr#1211, to-chk=121/1342)
# val/labels/human_jejunum_10.txt
#         105,286 100%    1.21MB/s    0:00:00 (xfr#1212, to-chk=120/1342)
# val/labels/human_kidney_01.txt
#          98,336 100%  342.97kB/s    0:00:00 (xfr#1213, to-chk=119/1342)
# val/labels/human_kidney_04.txt
#          78,454 100%  172.17kB/s    0:00:00 (xfr#1214, to-chk=118/1342)
# val/labels/human_kidney_05.txt
#         120,428 100%  188.17kB/s    0:00:00 (xfr#1215, to-chk=117/1342)
# val/labels/human_kidney_09.txt
#         130,882 100%  156.06kB/s    0:00:00 (xfr#1216, to-chk=116/1342)
# val/labels/human_kidney_10.txt
#         109,196 100%  104.44kB/s    0:00:01 (xfr#1217, to-chk=115/1342)
# val/labels/human_kidney_11.txt
#          79,254 100%  439.75kB/s    0:00:00 (xfr#1218, to-chk=114/1342)
# val/labels/human_liver_05.txt
#          42,218 100%  170.37kB/s    0:00:00 (xfr#1219, to-chk=113/1342)
# val/labels/human_liver_07.txt
#          28,124 100%   70.60kB/s    0:00:00 (xfr#1220, to-chk=112/1342)
# val/labels/human_liver_14.txt
#          36,038 100%   76.18kB/s    0:00:00 (xfr#1221, to-chk=111/1342)
# val/labels/human_liver_25.txt
#          28,440 100%   51.82kB/s    0:00:00 (xfr#1222, to-chk=110/1342)
# val/labels/human_liver_30.txt
#          24,608 100%   37.09kB/s    0:00:00 (xfr#1223, to-chk=109/1342)
# val/labels/human_liver_35.txt
#          37,532 100%   40.01kB/s    0:00:00 (xfr#1224, to-chk=108/1342)
# val/labels/human_liver_37.txt
#          31,998 100%   23.34kB/s    0:00:01 (xfr#1225, to-chk=107/1342)
# val/labels/human_liver_39.txt
#          24,316 100%   17.02kB/s    0:00:01 (xfr#1226, to-chk=106/1342)
# val/labels/human_lung_1.txt
#          21,418 100%   14.39kB/s    0:00:01 (xfr#1227, to-chk=105/1342)
# val/labels/human_melanoma_06.txt
#          39,774 100%    6.68MB/s    0:00:00 (xfr#1228, to-chk=104/1342)
# val/labels/human_melanoma_07.txt
#          34,140 100%  111.88kB/s    0:00:00 (xfr#1229, to-chk=103/1342)
# val/labels/human_melanoma_10.txt
#          28,418 100%   74.20kB/s    0:00:00 (xfr#1230, to-chk=102/1342)
# val/labels/human_muscle_2.txt
#           8,478 100%   12.00kB/s    0:00:00 (xfr#1231, to-chk=101/1342)
# val/labels/human_oesophagus_07.txt
#          91,822 100%  104.39kB/s    0:00:00 (xfr#1232, to-chk=100/1342)
# val/labels/human_oesophagus_13.txt
#          55,882 100%   59.58kB/s    0:00:00 (xfr#1233, to-chk=99/1342)
# val/labels/human_oesophagus_14.txt
#          22,608 100%   16.59kB/s    0:00:01 (xfr#1234, to-chk=98/1342)
# val/labels/human_oesophagus_16.txt
#          15,470 100%   10.71kB/s    0:00:01 (xfr#1235, to-chk=97/1342)
# val/labels/human_oesophagus_24.txt
#          25,482 100%    0.00kB/s    0:00:00 (xfr#1236, to-chk=96/1342)
# val/labels/human_oesophagus_26.txt
#          62,796 100%    1.20MB/s    0:00:00 (xfr#1237, to-chk=95/1342)
# val/labels/human_oesophagus_27.txt
#          16,330 100%   37.70kB/s    0:00:00 (xfr#1238, to-chk=94/1342)
# val/labels/human_oesophagus_28.txt
#          65,414 100%   80.25kB/s    0:00:00 (xfr#1239, to-chk=93/1342)
# val/labels/human_oesophagus_32.txt
#          29,126 100%   32.92kB/s    0:00:00 (xfr#1240, to-chk=92/1342)
# val/labels/human_pancreas_01.txt
#          59,938 100%   62.20kB/s    0:00:00 (xfr#1241, to-chk=91/1342)
# val/labels/human_pancreas_05.txt
#          50,292 100%   34.37kB/s    0:00:01 (xfr#1242, to-chk=90/1342)
# val/labels/human_pancreas_10.txt
#          39,472 100%  174.42kB/s    0:00:00 (xfr#1243, to-chk=89/1342)
# val/labels/human_pancreas_11.txt
#          53,128 100%  187.30kB/s    0:00:00 (xfr#1244, to-chk=88/1342)
# val/labels/human_pancreas_13.txt
#          47,058 100%   72.03kB/s    0:00:00 (xfr#1245, to-chk=87/1342)
# val/labels/human_pancreas_17.txt
#          55,212 100%   44.45kB/s    0:00:01 (xfr#1246, to-chk=86/1342)
# val/labels/human_pancreas_22.txt
#          33,700 100%  506.31kB/s    0:00:00 (xfr#1247, to-chk=85/1342)
# val/labels/human_pancreas_29.txt
#          59,280 100%  143.29kB/s    0:00:00 (xfr#1248, to-chk=84/1342)
# val/labels/human_pancreas_33.txt
#          52,784 100%  107.61kB/s    0:00:00 (xfr#1249, to-chk=83/1342)
# val/labels/human_pancreas_36.txt
#          26,828 100%   32.34kB/s    0:00:00 (xfr#1250, to-chk=82/1342)
# val/labels/human_pancreas_38.txt
#          45,216 100%   42.62kB/s    0:00:01 (xfr#1251, to-chk=81/1342)
# val/labels/human_pancreas_40.txt
#          38,520 100%  214.96kB/s    0:00:00 (xfr#1252, to-chk=80/1342)
# val/labels/human_peritoneum_10.txt
#          13,564 100%   54.96kB/s    0:00:00 (xfr#1253, to-chk=79/1342)
# val/labels/human_peritoneum_4.txt
#          61,894 100%  151.11kB/s    0:00:00 (xfr#1254, to-chk=78/1342)
# val/labels/human_placenta_03.txt
#          27,058 100%   43.97kB/s    0:00:00 (xfr#1255, to-chk=77/1342)
# val/labels/human_placenta_11.txt
#          48,762 100%   46.23kB/s    0:00:01 (xfr#1256, to-chk=76/1342)
# val/labels/human_placenta_20.txt
#          55,398 100%  151.54kB/s    0:00:00 (xfr#1257, to-chk=75/1342)
# val/labels/human_placenta_21.txt
#          38,922 100%   91.81kB/s    0:00:00 (xfr#1258, to-chk=74/1342)
# val/labels/human_placenta_23.txt
#          53,470 100%  109.70kB/s    0:00:00 (xfr#1259, to-chk=73/1342)
# val/labels/human_placenta_27.txt
#          56,052 100%   68.59kB/s    0:00:00 (xfr#1260, to-chk=72/1342)
# val/labels/human_placenta_30.txt
#          44,788 100%   44.22kB/s    0:00:00 (xfr#1261, to-chk=71/1342)
# val/labels/human_placenta_32.txt
#          64,908 100%   54.22kB/s    0:00:01 (xfr#1262, to-chk=70/1342)
# val/labels/human_placenta_35.txt
#          65,296 100%    1.13MB/s    0:00:00 (xfr#1263, to-chk=69/1342)
# val/labels/human_placenta_36.txt
#          71,272 100%  100.29kB/s    0:00:00 (xfr#1264, to-chk=68/1342)
# val/labels/human_placenta_37.txt
#          39,470 100%   49.93kB/s    0:00:00 (xfr#1265, to-chk=67/1342)
# val/labels/human_placenta_38.txt
#          41,026 100%   45.37kB/s    0:00:00 (xfr#1266, to-chk=66/1342)
# val/labels/human_pylorus_7.txt
#          36,252 100%   35.72kB/s    0:00:00 (xfr#1267, to-chk=65/1342)
# val/labels/human_rectum_2.txt
#          16,282 100%   15.24kB/s    0:00:01 (xfr#1268, to-chk=64/1342)
# val/labels/human_rectum_3.txt
#          26,032 100%   22.40kB/s    0:00:01 (xfr#1269, to-chk=63/1342)
# val/labels/human_rectum_9.txt
#          27,460 100%   22.33kB/s    0:00:01 (xfr#1270, to-chk=62/1342)
# val/labels/human_salivory_03.txt
#          70,776 100%   34.13kB/s    0:00:02 (xfr#1271, to-chk=61/1342)
# val/labels/human_salivory_10.txt
#          52,304 100%  751.15kB/s    0:00:00 (xfr#1272, to-chk=60/1342)
# val/labels/human_salivory_14.txt
#          88,886 100%  236.52kB/s    0:00:00 (xfr#1273, to-chk=59/1342)
# val/labels/human_salivory_15.txt
#          40,892 100%   93.52kB/s    0:00:00 (xfr#1274, to-chk=58/1342)
# val/labels/human_salivory_17.txt
#          84,088 100%  150.67kB/s    0:00:00 (xfr#1275, to-chk=57/1342)
# val/labels/human_salivory_28.txt
#          52,414 100%   64.47kB/s    0:00:00 (xfr#1276, to-chk=56/1342)
# val/labels/human_salivory_31.txt
#         102,174 100%   83.43kB/s    0:00:01 (xfr#1277, to-chk=55/1342)
# val/labels/human_salivory_36.txt
#          70,644 100%  346.67kB/s    0:00:00 (xfr#1278, to-chk=54/1342)
# val/labels/human_salivory_44.txt
#          77,560 100%  144.00kB/s    0:00:00 (xfr#1279, to-chk=53/1342)
# val/labels/human_spleen_07.txt
#          33,074 100%   53.92kB/s    0:00:00 (xfr#1280, to-chk=52/1342)
# val/labels/human_spleen_28.txt
#         138,462 100%  108.78kB/s    0:00:01 (xfr#1281, to-chk=51/1342)
# val/labels/human_spleen_31.txt
#         124,226 100%    1.20MB/s    0:00:00 (xfr#1282, to-chk=50/1342)
# val/labels/human_testis_10.txt
#          30,736 100%   89.60kB/s    0:00:00 (xfr#1283, to-chk=49/1342)
# val/labels/human_testis_11.txt
#          14,144 100%   32.42kB/s    0:00:00 (xfr#1284, to-chk=48/1342)
# val/labels/human_tongue_01.txt
#          16,064 100%   32.89kB/s    0:00:00 (xfr#1285, to-chk=47/1342)
# val/labels/human_tongue_09.txt
#          21,844 100%   38.16kB/s    0:00:00 (xfr#1286, to-chk=46/1342)
# val/labels/human_tongue_14.txt
#          23,344 100%   37.07kB/s    0:00:00 (xfr#1287, to-chk=45/1342)
# val/labels/human_tongue_22.txt
#          10,422 100%   15.17kB/s    0:00:00 (xfr#1288, to-chk=44/1342)
# val/labels/human_tongue_28.txt
#          81,430 100%   54.92kB/s    0:00:01 (xfr#1289, to-chk=43/1342)
# val/labels/human_tongue_29.txt
#          38,380 100%  215.40kB/s    0:00:00 (xfr#1290, to-chk=42/1342)
# val/labels/human_tongue_35.txt
#          44,008 100%  121.06kB/s    0:00:00 (xfr#1291, to-chk=41/1342)
# val/labels/human_tongue_36.txt
#          18,452 100%   34.65kB/s    0:00:00 (xfr#1292, to-chk=40/1342)
# val/labels/human_tongue_39.txt
#          26,392 100%   43.54kB/s    0:00:00 (xfr#1293, to-chk=39/1342)
# val/labels/human_tonsile_10.txt
#          22,924 100%   33.92kB/s    0:00:00 (xfr#1294, to-chk=38/1342)
# val/labels/human_tonsile_11.txt
#          57,664 100%   75.59kB/s    0:00:00 (xfr#1295, to-chk=37/1342)
# val/labels/human_umbilical_cord_06.txt
#           3,578 100%    4.36kB/s    0:00:00 (xfr#1296, to-chk=36/1342)
# val/labels/human_umbilical_cord_09.txt
#           3,436 100%    3.91kB/s    0:00:00 (xfr#1297, to-chk=35/1342)
# val/labels/human_umbilical_cord_11.txt
#          11,394 100%   11.93kB/s    0:00:00 (xfr#1298, to-chk=34/1342)
# val/labels/mouse_femur_01.txt
#          63,426 100%   42.84kB/s    0:00:01 (xfr#1299, to-chk=33/1342)
# val/labels/mouse_femur_02.txt
#         114,778 100%  112.09kB/s    0:00:01 (xfr#1300, to-chk=32/1342)
# val/labels/mouse_femur_03.txt
#          34,260 100%  348.51kB/s    0:00:00 (xfr#1301, to-chk=31/1342)
# val/labels/mouse_heart_02.txt
#          15,742 100%   79.24kB/s    0:00:00 (xfr#1302, to-chk=30/1342)
# val/labels/mouse_heart_09.txt
#          11,968 100%   30.36kB/s    0:00:00 (xfr#1303, to-chk=29/1342)
# val/labels/mouse_heart_10.txt
#           9,816 100%   20.57kB/s    0:00:00 (xfr#1304, to-chk=28/1342)
# val/labels/mouse_heart_11.txt
#           7,224 100%   13.06kB/s    0:00:00 (xfr#1305, to-chk=27/1342)
# val/labels/mouse_heart_13.txt
#           9,736 100%   15.87kB/s    0:00:00 (xfr#1306, to-chk=26/1342)
# val/labels/mouse_heart_15.txt
#          16,796 100%   23.74kB/s    0:00:00 (xfr#1307, to-chk=25/1342)
# val/labels/mouse_heart_17.txt
#          12,102 100%   13.58kB/s    0:00:00 (xfr#1308, to-chk=24/1342)
# val/labels/mouse_heart_28.txt
#          17,536 100%   18.30kB/s    0:00:00 (xfr#1309, to-chk=23/1342)
# val/labels/mouse_kidney_04.txt
#          24,916 100%   23.85kB/s    0:00:01 (xfr#1310, to-chk=22/1342)
# val/labels/mouse_kidney_20.txt
#          22,100 100%   19.02kB/s    0:00:01 (xfr#1311, to-chk=21/1342)
# val/labels/mouse_kidney_26.txt
#          33,608 100%  820.31kB/s    0:00:00 (xfr#1312, to-chk=20/1342)
# val/labels/mouse_kidney_31.txt
#          27,518 100%  139.96kB/s    0:00:00 (xfr#1313, to-chk=19/1342)
# val/labels/mouse_kidney_34.txt
#          34,262 100%   62.54kB/s    0:00:00 (xfr#1314, to-chk=18/1342)
# val/labels/mouse_liver_07.txt
#          19,372 100%   31.48kB/s    0:00:00 (xfr#1315, to-chk=17/1342)
# val/labels/mouse_liver_12.txt
#          19,592 100%   21.35kB/s    0:00:00 (xfr#1316, to-chk=16/1342)
# val/labels/mouse_liver_14.txt
#          19,660 100%   20.23kB/s    0:00:00 (xfr#1317, to-chk=15/1342)
# val/labels/mouse_liver_17.txt
#          12,664 100%   12.26kB/s    0:00:01 (xfr#1318, to-chk=14/1342)
# val/labels/mouse_liver_20.txt
#          14,560 100%   11.49kB/s    0:00:01 (xfr#1319, to-chk=13/1342)
# val/labels/mouse_liver_32.txt
#          25,078 100%   18.94kB/s    0:00:01 (xfr#1320, to-chk=12/1342)
# val/labels/mouse_liver_35.txt
#          12,354 100%    8.93kB/s    0:00:01 (xfr#1321, to-chk=11/1342)
# val/labels/mouse_muscle_tibia_04.txt
#           2,530 100%    0.00kB/s    0:00:00 (xfr#1322, to-chk=10/1342)
# val/labels/mouse_muscle_tibia_09.txt
#           7,182 100%  109.59kB/s    0:00:00 (xfr#1323, to-chk=9/1342)
# val/labels/mouse_muscle_tibia_13.txt
#             596 100%    4.81kB/s    0:00:00 (xfr#1324, to-chk=8/1342)
# val/labels/mouse_muscle_tibia_16.txt
#           6,876 100%   18.05kB/s    0:00:00 (xfr#1325, to-chk=7/1342)
# val/labels/mouse_spleen_01.txt
#         155,488 100%   72.72kB/s    0:00:02 (xfr#1326, to-chk=6/1342)
# val/labels/mouse_spleen_02.txt
#         188,586 100%  386.90kB/s    0:00:00 (xfr#1327, to-chk=5/1342)
# val/labels/mouse_spleen_05.txt
#         149,890 100%  171.00kB/s    0:00:00 (xfr#1328, to-chk=4/1342)
# val/labels/mouse_subscapula_14.txt
#           4,424 100%    4.61kB/s    0:00:00 (xfr#1329, to-chk=3/1342)
# val/labels/mouse_subscapula_15.txt
#           2,510 100%    2.48kB/s    0:00:00 (xfr#1330, to-chk=2/1342)
# val/labels/mouse_subscapula_28.txt
#           7,316 100%    6.86kB/s    0:00:01 (xfr#1331, to-chk=1/1342)
# val/labels/mouse_subscapula_36.txt
#           9,546 100%    8.21kB/s    0:00:01 (xfr#1332, to-chk=0/1342)

# sent 312,864,893 bytes  received 25,399 bytes  789,130.62 bytes/sec
# total size is 312,700,041  speedup is 1.00
#    ⚠ Warning: rsync returned non-zero exit code: 5888
# 3. Detecting available splits...
#    ✓ Found train split
#    ✓ Found val split
#    ✓ Found test split

# 4. Creating data.yaml...
#    ✓ Created: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/data.yaml

# 5. data.yaml contents:
# ----------------------------------------------------------------------
# path: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols
# train: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/train/images
# val: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/val/images
# test: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/test/images
# nc: 1
# names:
# - Nuclei

# ----------------------------------------------------------------------


# ======================================================================
# DATA.YAML VALIDATION
# ======================================================================
# File: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/data.yaml

# rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1338) [sender=3.2.7]
# ✓ train: /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/train/images
#          399 images
# ✓ val  : /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/val/images
#          133 images
# ✓ test : /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset_on_vols/test/images
#          133 images

# Classes: 1
# Names: ['Nuclei']
# ======================================================================

# ✓ All validations passed!

# ======================================================================
# COPY COMPLETE
# ======================================================================


# COMMAND ----------

# DBTITLE 1,check for required data/configs/paths
# Verify YOLO training requirements from UC Vols are met
status = check_yolo_environment(verbose=True, create_missing=True)

if not status['ready']:
    raise RuntimeError("Environment not ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Specify & Organize Training Paths 

# COMMAND ----------

# DBTITLE 1,Update settings
from ultralytics import settings

# Update setting specirically for datasets_dir
settings.update({"datasets_dir": f"{YOLO_DATA_UCVol_path}"})
print(settings) 

# COMMAND ----------

# DBTITLE 1,Additional Paths for PyTorch Training
# Config project structure directory under UC
PROJECT_training_runs = f'{PROJECT_PATH}/training_runs_sgc/'
os.makedirs(PROJECT_training_runs, exist_ok=True)

PROJECT_yolo_model = f'{PROJECT_PATH}/yolo_model_sgc/'
os.makedirs(PROJECT_yolo_model, exist_ok=True)

# for cache related to ultralytics
os.environ['ULTRALYTICS_CACHE_DIR'] = PROJECT_yolo_model


## ephemeral project location on VM, required for Appending operation during training.
# tmp_project_location = f"/local_disk0/tmp/nuinsseg/" # not consistently accessible via severless but much faster on classic compute cf /tmp/
# os.makedirs(tmp_project_location, exist_ok=True)

# Serverless provides writable temp space
tmp_project_location = os.path.join(tempfile.gettempdir(), "nuinsseg")
print(f"Temp project location: {tmp_project_location}")
os.makedirs(tmp_project_location, exist_ok=True)


# Make tmp_project_location available to callbacks
import builtins
builtins.tmp_project_location = tmp_project_location

# COMMAND ----------

# DBTITLE 1,path permissions check!
!ls -lah {tmp_project_location}

# COMMAND ----------

# DBTITLE 1,check gpu status
# MAGIC %sh nvidia-smi

# COMMAND ----------

# DBTITLE 1,check cuda availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# COMMAND ----------

# DBTITLE 1,Load model and check device
model = YOLO(f"{PROJECT_yolo_model}/yolo11n-seg.pt")
print(f"Model device: {next(model.parameters()).device}")

# If not on CUDA, move to CUDA
if next(model.parameters()).device.type != 'cuda':
    model.to('cuda')
    print(f"Moved model to: {next(model.parameters()).device}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transfer Learning: Run on Serverless GPU with MLflow Integration

# COMMAND ----------

# DBTITLE 1,Training Configuration + MLflow Callback
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Clean up any active runs
if mlflow.active_run():
    print(f"Ending active run: {mlflow.active_run().info.run_id}")
    mlflow.end_run()

if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)

# Checkpoint logging configuration 
CHECKPOINT_LOG_FREQUENCY = 10 #1 #10 (for larger n_epochs training) # you can specify to use e.g. n_epochs/divisor
configure_checkpoint_logging(frequency=CHECKPOINT_LOG_FREQUENCY, log_best=True, log_final=True, log_first=True)

# Get experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if experiment is None:    
    print(f"Creating experiment: {experiment_name}")
    experiment_id = mlflow.create_experiment(name=experiment_name)
else:
    print(f"Reusing experiment_name: {experiment_name} | experiment_id: {experiment.experiment_id}")
    experiment_id = experiment.experiment_id

# COMMAND ----------

# DBTITLE 1,Training with callback and MLflow run + minimal plots/saving
n_epochs = 50 #10  # For quick testing use e.g. 5-10; at least 50 epochs for better inference performance | Adjust as needed | 50 epochs + batch=8 : 1hr50mins
batch_sz = 8  # Adjust based on GPU memory

try:
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"yolo_training_{timestamp}",
        log_system_metrics=True,
    ) as run:

        # Get organized paths
        run_paths = get_organized_paths(run.info.run_id, timestamp, PROJECT_PATH)
        
        # Create all directories
        for path_key, path_value in run_paths.items():
            if path_key != 'base':
                os.makedirs(path_value, exist_ok=True)
        
        print(f"Organized paths:")
        print(f"  Base: {run_paths['base']}")
        print(f"  Train: {run_paths['train']}")
        print(f"  Weights: {run_paths['train_weights']}")
        
        # Make run_paths available globally for inference
        builtins.current_run_paths = run_paths
        builtins.YOLO_DATA_UCVol_path = YOLO_DATA_UCVol_path
        
        # Load model
        model = YOLO(f"{PROJECT_PATH}/yolo_model_sgc/yolo11n-seg.pt")
        model.to("cuda")
        
        # Log parameters
        mlflow.log_params({
            "model_name": "yolo11n-seg",
            "epochs": n_epochs,
            "batch_size": batch_sz,
            "image_size": 1024,
            "optimizer": "adam",
            "checkpoint_log_frequency": CHECKPOINT_LOG_FREQUENCY,
            "training_timestamp": timestamp,
            "artifacts_base_path": run_paths['base'],
        })
        
        # Register callback
        model.add_callback("on_fit_epoch_end", mlflow_epoch_logger)
        
        # TRAIN
        print(f"\nStarting training: yolo_training_{timestamp}")
        print(f"Epochs: {n_epochs}, Batch: {batch_sz}")
        print(f"{'='*70}\n")
        
        model.train(            
            data=os.path.join(YOLO_DATA_UCVol_path, "data.yaml"),
            epochs=n_epochs, 
            batch=batch_sz, 
            imgsz=1024,
            workers=0, ## A10 SGC has no multithreaded workers
            project=tmp_project_location,
            name="yolo_training",
            exist_ok=True,
            optimizer="adam",
            device=0,          # Explicit instead of selecting idle GPU e.g. device = -1
            val=True,
            plots=False,       # Disable plots during training
            save=True,
            save_period=-1,    # Only save last/best wrt YOLO framework -- MLflow logging is done separately + callbacks
            # save_weights=True, # Save weights for each epoch            
        )
        
        # Copy artifacts to UC Volumes
        copy_training_artifacts(tmp_project_location, run_paths)
        
        # Log artifacts to MLflow
        data_yaml_path = os.path.join(YOLO_DATA_UCVol_path, "data.yaml")
        log_training_artifacts_to_mlflow(run_paths, data_yaml_path)
        
        # Finalize training run with metrics and summary
        finalize_training_run(run, timestamp, n_epochs, batch_sz, run_paths)
        
        # Store for inference use
        global current_run_paths
        current_run_paths = run_paths
        
        # Reset callback state for next run
        for attr in ['best_fitness', 'best_epoch', 'checkpoints_logged']:
            if hasattr(mlflow_epoch_logger, attr):
                delattr(mlflow_epoch_logger, attr)

except Exception as e:
    print(f"\n{'='*70}\nTRAINING FAILED\n{'='*70}")
    print(f"Error: {e}\n{'='*70}")
    
    # Log error to MLflow if possible
    try:
        mlflow.log_param("training_error", str(e))
        mlflow.set_tag("training_status", "failed")
    except:
        pass
    
    # Clean up callback state
    for attr in ['best_fitness', 'best_epoch', 'checkpoints_logged']:
        if hasattr(mlflow_epoch_logger, attr):
            delattr(mlflow_epoch_logger, attr)
    
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** 
# MAGIC Slower training via SGC with comprehensive MLFlow tracking/checkpointing/plots-saving (may not be required for quick testing)

# COMMAND ----------

# DBTITLE 1,Get run_id
# run_id = "<manually add from previous run>"

run_id = run.info.run_id
print(f"Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Post-Training: Validation (optional) + Test Inference + Visualization
# MAGIC With the model trained and checkpoints saved, we can now:
# MAGIC 1. Load the best model from the training run
# MAGIC 2. Run inference on (validation and) test sets
# MAGIC 3. Where labels exist for datasets, we can    
# MAGIC   a. Calculate comprehensive metrics (mAP, precision, recall, fitness);   
# MAGIC   b. Compare predictions against ground truth  
# MAGIC 4. Visualize predictions with customizable overlays
# MAGIC 5. Generate summary reports.  
# MAGIC All inference results are automatically logged to MLflow and saved to Unity Catalog volumes.

# COMMAND ----------

run_id

# COMMAND ----------

# MAGIC %md
# MAGIC **!!! NB:** We may have to restart the compute, run the relevant setups and paths (without retraining) for Inferencing due to A10 CUDA OOM; alternatively, select `device='cpu'` in the input args.  

# COMMAND ----------

# DBTITLE 1,Run {Val} or Test Inference + Metrics
## Clear cache before starting
clear_all_caches()
torch.cuda.empty_cache()
gpu_status() ## need at least ~5-10gigs

# Run inference on test split with validation
test_summary = run_inference_with_metrics(
    run_id=run_id,
    split='test',
    has_labels=True, ## in this dataset test split has labels ## alternatively has_labels=False, # test -- will not run validation
    validate_config=True,  # Validates data.yaml before running
    log_to_mlflow=True,
    skip_existing_params=True,    
    # debug=True,  # ← Enable debug output # test without debug
    device='cuda', # 'auto' | 'cpu' in case of cuda OOM    
    batch_size=8 # reduce if necessary 
)

# Inspect the output
# inspect_inference_output(test_summary)

# COMMAND ----------

# DBTITLE 1,Test Inference Summary
print_inference_summary(test_summary, include_paths=True, include_performance=True)

# COMMAND ----------

# DBTITLE 1,Visualize Test Results - Boxes Only
## Clear cache before starting
clear_all_caches()
torch.cuda.empty_cache()
gpu_status() 

fig = visualize_inference_results(
    test_summary,
    num_samples=25, # update as appropriate
    show_boxes=True,
    show_masks=False,
    show_labels=False,
    show_conf=False,
    box_color=(255, 255, 255),
    figsize=(10,10),
    save_figure=True,    
)

# COMMAND ----------

# DBTITLE 1,test | no labels
## Clear cache before starting
clear_all_caches()
torch.cuda.empty_cache()
gpu_status() 

# Test predict mode (no labels)
test_predict_summary = run_inference_with_metrics(
    run_id=run_id,
    split='test',
    has_labels=False,  # Force predict mode
    validate_config=True,
    log_to_mlflow=True,
    skip_existing_params=True,
    # debug=True,  
    device='cpu', # slower per image but avoids CUDA OOM
    batch_size=8
)

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------
# MAGIC ### Summary
# MAGIC ------------------------------------------------------------------ 
# MAGIC In this notebook we demonstrated how to leverage the YOLO instance segmentation model on a custom nuclei dataset preprocessed in YOLO format and saved on Unity Catalog volumes.
# MAGIC
# MAGIC Depending on the number of training epochs and other parameters, the inference results may yield better cell nuclei segmentations. The logged metrics and visualizations can help guide the focus of finetuning.
# MAGIC
# MAGIC Future work and extensions: Distributed training (multi-GPU), as well as logging the finetuned YOLO model as a MLflow Custom Pyfunc for Unity Catalog registration, and/or serving as an endpoint for downstream applications.
# MAGIC
# MAGIC #### What We Built
# MAGIC
# MAGIC We enhanced MLflow tracking and checkpointing for YOLO training workflows within Databricks with the following features:   
# MAGIC
# MAGIC **Native Databricks Managed MLflow Integration:**
# MAGIC We used MLflow's tracking API to log parameters, metrics, and artifacts (including model checkpoints) directly from within the training loop via custom callbacks, leveraging Databricks' built-in MLflow support for seamless experiment management and centralized artifact storage.   
# MAGIC
# MAGIC **Custom Epoch-Level Callback for Comprehensive Tracking:**
# MAGIC A custom `mlflow_epoch_logger` callback was implemented to capture and log training/validation metrics after each epoch, providing detailed visibility into model performance throughout training. This goes beyond YOLO's default CSV-based logging by providing structured, queryable metrics in MLflow.   
# MAGIC
# MAGIC **Configurable Checkpoint Logging:**
# MAGIC The callback supports flexible checkpoint logging strategies:
# MAGIC - Log first epoch (baseline)
# MAGIC - Log final epoch (completion)
# MAGIC - Log best model (based on fitness metric)
# MAGIC - Log periodic checkpoints (configurable frequency, e.g., every 10 epochs)
# MAGIC This provides fine-grained model versioning and recovery options while balancing storage efficiency.   
# MAGIC
# MAGIC **`Best Model` Tracking:**
# MAGIC The callback monitors validation fitness (weighted combination of mAP50 and mAP50-95) and automatically logs the best-performing model checkpoint to MLflow, ensuring easy access to the optimal model for deployment or further analysis.   
# MAGIC
# MAGIC <!-- **Resume Training Support:**
# MAGIC Helper functions (`setup_resume()`, `get_resume_checkpoint()`) enable resuming training from any saved checkpoint, restoring both model state and callback tracking variables (best fitness, best epoch, checkpoint count) for seamless continuation of interrupted training runs.    -->
# MAGIC
# MAGIC **Organized Artifact Management:**
# MAGIC Training artifacts are organized in a structured directory hierarchy on Unity Catalog volumes:
# MAGIC ```
# MAGIC /Volumes/catalog/schema/volume/ 
# MAGIC └── run_YYYYMMDD_HHMMSS_<run_id>/ 
# MAGIC ├── train/ 
# MAGIC │ ├── weights/ 
# MAGIC │ ├── checkpoints/ 
# MAGIC │ └── [plots, logs, results.csv] 
# MAGIC └── val/
# MAGIC └── test/
# MAGIC ```
# MAGIC #### Overall
# MAGIC This approach provides a **production-ready, reproducible, and collaborative** deep learning workflow for YOLO training within the Databricks ecosystem. While it adds some overhead compared to vanilla YOLO, it delivers significant value through:
# MAGIC 1. **Comprehensive experiment tracking** - All metrics, parameters, and artifacts in one place
# MAGIC  : within MLflow UI and UC Volumes path.   
# MAGIC 2. **Reproducibility** - Easy to recreate any training run from logged artifacts
# MAGIC 3. **Collaboration** - Centralized storage and tracking accessible to entire team
# MAGIC 4. **Governance** - Unity Catalog integration for access control and lineage
# MAGIC 5. **Flexibility** - Compare experiments, deploy best models. 
# MAGIC <!-- 5. **Flexibility** - Resume training (helperfuncs included), compare experiments, deploy best models.  -->
# MAGIC
# MAGIC The trade-off between performance overhead and operational benefits makes this approach well-suited for **production ML workflows** where reproducibility, collaboration, and governance are priorities.
