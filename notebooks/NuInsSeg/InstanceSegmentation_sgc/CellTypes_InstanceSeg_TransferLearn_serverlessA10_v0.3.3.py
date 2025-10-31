# Databricks notebook source
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
# MAGIC ### Applying `YOLO_v11` Instance Segmentation within Databricks 
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
# MAGIC
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
dbutils.widgets.text("CATALOG_NAME", "<your_catalog_name>","Catalog Name")
dbutils.widgets.text("SCHEMA_NAME","<your_schema_name>","Schema Name")
dbutils.widgets.text("VOLUME_NAME","<your_project_name>","Volume Name")

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
    project_path=f'/Volumes/{CATLOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}/NuInsSeg',
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
# MAGIC ### We will first illustrate the `Default` local workspace paths used by Ultralytics.
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
# MAGIC #### Quick Inference with default YOLO framework + Workspace datapath best weights 

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
# MAGIC #### Data Preparation: Copy YOLO Images to Unity Catalog Volumes
# MAGIC To make sure we have sufficient space and paths to organize our training outputs, MLflow tracking, and logging, we will copy our workspace YOLO formatted data to UC Volumes. 
# MAGIC _The [NuInsSeg data](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images), while not huge in size, takes about 5-10 mins for the copying to UC Volumes. Ideally the data is downloaded to UC Vols and the preprocessed versions are updated in UC via medallion ETL. For the simplicity of this example we make them available via the workspace path as a start._

# COMMAND ----------

# DBTITLE 1,YOLO_DATA_UCVol_path
## Specify the UC Volumes destination path for the YOLO dataset
YOLO_DATA_UCVol_path = f'{PROJECT_PATH}/yolo_dataset_on_vols'

# COMMAND ----------

# DBTITLE 1,copy datasets to UC Vols (if needed)
## Execute the copy ~8mins | leave commented out if already done else uncomment & run
data_yaml_path = copy_to_uc_volumes_with_yaml(WS_PROJ_DIR, YOLO_DATA_UCVol_path)

##NB: rsync will attempt to make copy -- you may see initial errors/warnings. 

# COMMAND ----------

# DBTITLE 1,check for required data/configs/paths
# Verify YOLO training requirements from UC Vols are met
status = check_yolo_environment(verbose=True, create_missing=True)

if not status['ready']:
    raise RuntimeError("Environment not ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Specify & Organize Training Paths 

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
# MAGIC #### Transfer Learning: Run on Serverless GPU with MLflow Integration

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
# MAGIC #### Post-Training: Validation (optional) + Test Inference + Visualization
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
# MAGIC
# MAGIC ## Summary
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
# MAGIC
# MAGIC ```
# MAGIC
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
