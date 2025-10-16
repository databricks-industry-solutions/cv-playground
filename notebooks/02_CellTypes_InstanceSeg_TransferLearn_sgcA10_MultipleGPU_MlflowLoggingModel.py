# Databricks notebook source
# MAGIC %md
# MAGIC > - This notebook is an extension example of how to run YOLO Instance Segmentation on a custom dataset with **remote multiple serverless gpu (A10) compute**. 
# MAGIC > - For single SGC gpu run with more granular details (like custom callback functions per checkpoint and more controlled logging practice), see the other **"01_CellTypes_InstanceSeg_TransferLearn_sgcA10"** notebook by `may.merkletan@databricks.com`.  
# MAGIC > - The example solution will be part of the assets within the forthcoming [databricks-industry-solutions/cv-playground](https://github.com/databricks-industry-solutions/cv-playground) that will show case other CV-related solutions on Databricks.   
# MAGIC > - Developed and last tested [`2025Oct12`] using remote multiple `sgc_A10` and `env_v4` by `yang.yang@databricks.com`
# MAGIC `@distributed(gpus=8, gpu_type='A10', remote=True)`
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup Widgets for User to setup their own I/O path for the project
## replace with your specific catalog and schema etc. names in the widgets panel above.
dbutils.widgets.text("CATALOG_NAME", "yyang", "Catalog Name")
dbutils.widgets.text("SCHEMA_NAME", "computer_vision", "Schema Name")
dbutils.widgets.text("VOLUME_NAME", "projects", "Volume Name")

# COMMAND ----------

# MAGIC %md
# MAGIC ---    

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this notebook walks you through:  
# MAGIC **[1] YOLO continued training on serverless compute (SGC) using multipe remote GPUs    
# MAGIC [2] Log Model itself and the related paramters/metrics/artifacts with [Databricks managed MLflow](https://www.databricks.com/product/managed-mlflow)**
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Pinned Dependencies
import serverless_gpu
%pip install -U mlflow>=3.0
%pip install threadpoolctl==3.1.0
%pip install ultralytics==8.3.204
%pip install nvidia-ml-py==13.580.82 # for later mlflow GPU monitoring
%pip install pyrsmi==0.2.0 # for later mlflow AMD GPU monitoring if you have AMD

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,check dependencies version
import importlib.metadata as meta

versions = {}
packages = [
    'ultralytics',
    'torch',
    'mlflow',
    'scikit-learn',
    'matplotlib',
    'nvidia-ml-py',
    'threadpoolctl'
]

for pkg in packages:
    try:
        versions[pkg] = meta.version(pkg)
    except Exception:
        try:
            mod = __import__(pkg.replace('-', '_'))
            versions[pkg] = getattr(mod, '__version__', 'Not found')
        except Exception:
            versions[pkg] = 'Not found'

for pkg, ver in versions.items():
    print(f"{pkg:15}: {ver}")


### check wrt pinned depdendencies
# ultralytics    : 8.3.200
# torch          : 2.6.0+cu124
# mlflow         : 2.21.3
# scikit-learn   : 1.7.2
# matplotlib     : 3.10.7
# nvidia-ml-py3  : 7.352.0
# threadpoolctl  : 3.6.0    

# COMMAND ----------

# DBTITLE 1,Load Library
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from ultralytics import YOLO
from serverless_gpu import distributed
import mlflow


import os
from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from ultralytics.utils import RANK, LOCAL_RANK

# COMMAND ----------

# MAGIC %cat /tmp/Ultralytics/settings.json

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,PATH NAMES
#: these variables inherited from widgets panel on the top.
CATALOG_NAME = dbutils.widgets.get("CATALOG_NAME")
SCHEMA_NAME = dbutils.widgets.get("SCHEMA_NAME")
VOLUME_NAME = dbutils.widgets.get("VOLUME_NAME")

## Volumes path prefix
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/"
# volume
PROJECTS_DIR = f"{VOLUME_PATH}/{VOLUME_NAME}"
# folder under volume
PROJECT_PATH = f"{PROJECTS_DIR}/NuInsSeg"
# subfolder
YOLO_DATA_DIR = f"{PROJECT_PATH}/yolo_dataset" # can update this to change the path to your own data 

# Get the current working directory
nb_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
current_path = f"/Workspace{nb_context}"
# print(f"Current path: {current_path}")
WS_PROJ_DIR = '/'.join(current_path.split('/')[:-1]) 

WORKSPACE_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_WORKSPACE_PATH = f"/Users/{WORKSPACE_PATH}"


### we need to defined experiment_name when starting mflow...
project_name = "yolo_CellTypesNuclei_InstanceSeg_scg"
# experiment_name = f"{USER_WORKSPACE_PATH}/mlflow_experiments/yolo/{project_name}"
experiment_name = f"{WS_PROJ_DIR}/{project_name}"
# os.makedirs(experiment_name, exist_ok=True)  # won't work on serverless
mlflow.set_experiment(experiment_name)
print(f"Setting experiment name to be {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Create Catalog, Schema, and Volume if not exists
# Create catalog if not exists
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")

# Create schema if not exists
spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}"
)

# Create volume if not exists
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.{VOLUME_NAME}"
)

# COMMAND ----------

# DBTITLE 1,check paths
# List of paths to check/create (Volumes)
paths_to_check = [
    PROJECTS_DIR,
    PROJECT_PATH,    
    YOLO_DATA_DIR
]

def path_exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

for path in paths_to_check:
    if not path_exists(f"{path}"):
        print(f"{path}", path_exists(f"{path}"))
        dbutils.fs.mkdirs(f"{path}")
    else:
        print(f"{path}", path_exists(f"{path}"))    

## Alternatively
# display(dbutils.fs.ls(f"{PROJECTS_DIR}"))     

# COMMAND ----------

PROJECT_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------------

# COMMAND ----------

# DBTITLE 1,preprocessed DATA in YOLO format
# MAGIC %ls -lah {WS_PROJ_DIR}/datasets/ 

# COMMAND ----------

# DBTITLE 1,data.yaml specifying data paths
# MAGIC %cat data.yaml

# COMMAND ----------

# DBTITLE 1,specify the data.yaml path under the wksp

data_yaml_path = f"{WS_PROJ_DIR}/data.yaml"

# COMMAND ----------

# MAGIC %md
# MAGIC > Note: **For better governance, we prefer to have datasets sit in the UC volume.**

# COMMAND ----------

# MAGIC %md 
# MAGIC # Test with Multiple-GPU Training

# COMMAND ----------

# DBTITLE 1,Load related library
from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Note: To infer the signature, we will need to load a model (off-the-shell), predict on a few images and get the output.**
# MAGIC
# MAGIC The format of the prediction output matters, not the quality at this time. We will not log anything specific.

# COMMAND ----------

# DBTITLE 1,load the best model
model = YOLO(f"yolo11n-seg.pt")

# COMMAND ----------

model.train(
                # task="detect",
                batch=8, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70). Note, with multi-GPU, only integer works. Others modes all throw errors.
                device=[-1], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
                data=data_yaml_path,
                epochs=2,
                # project=f'{tmp_project_location}', # local VM ephermal location
                # project=f'{volume_project_location}', # volume path still wont work
                #
                save=True,
                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                project=WS_PROJ_DIR,
                exist_ok=True,
                #
                fliplr=1,
                flipud=1,
                perspective=0.001,
                degrees=.45
            )

# COMMAND ----------

# DBTITLE 1,input and output
example_image_path = [f"{WS_PROJ_DIR}/datasets/test/images/human_bladder_01.png", f"{WS_PROJ_DIR}/datasets/test/images/human_brain_9.png"]
predictions = model.predict(example_image_path)

# COMMAND ----------

predictions

# COMMAND ----------

# DBTITLE 1,to_pandas so easy to infer signature later
prediction = predictions[0]
pred_df = prediction.to_df().to_pandas()

# COMMAND ----------

pred_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual ModelSignature Limitations
# MAGIC
# MAGIC **Key Limitation: Manual ModelSignature can only define flat schemas**
# MAGIC
# MAGIC When using MLflow's `ModelSignature` with manual `ColSpec` definitions, you are limited to **flat, primitive data types only**:
# MAGIC
# MAGIC ### Supported Types in Manual ColSpec:
# MAGIC - `boolean`, `integer`, `long`, `float`, `double`, `string`, `binary`, `datetime`
# MAGIC
# MAGIC ### **NOT Supported** in Manual ColSpec:
# MAGIC - ❌ Nested structures like `{"box": {"x1": 10.5, "y1": 15.2}}`
# MAGIC - ❌ Array types like `Array(double)` or `[1.0, 2.0, 3.0]`
# MAGIC - ❌ Complex objects with multiple levels of nesting
# MAGIC
# MAGIC ### Comparison:
# MAGIC
# MAGIC **`infer_signature()` (Automatic)**:
# MAGIC - ✅ Automatically detects nested structures from actual data
# MAGIC - ✅ Supports complex types like `Object` and `Array`
# MAGIC - ✅ Can handle DataFrames with dictionary columns
# MAGIC - ✅ Creates schemas like: `'box': {x1: double, x2: double, y1: double, y2: double}`
# MAGIC
# MAGIC **Manual `ModelSignature` with `ColSpec`**:
# MAGIC - ❌ Limited to flat, primitive columns only
# MAGIC - ❌ Cannot create nested `Object` or `Array` types
# MAGIC - ✅ Gives precise control over schema definition
# MAGIC - ✅ Better for production models with predictable, flat outputs
# MAGIC
# MAGIC ### Best Practice:
# MAGIC **For complex nested data**: Use `infer_signature()` with sample data
# MAGIC **For production models**: Flatten your data structure and use manual `ColSpec` for precise control
# MAGIC
# MAGIC ### Here:
# MAGIC **Here we will use `infer_signature()` since `pred_df` has nested structure.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Infer Signature Here

# COMMAND ----------

signature = infer_signature(example_image_path, pred_df)

# COMMAND ----------

signature

# COMMAND ----------

# DBTITLE 1,(skip) Manual ModelSignature - No Exact Match
# # Create manual ModelSignature that exactly matches the inferred signature
# from mlflow.types.schema import Schema, ColSpec
# from mlflow.models.signature import ModelSignature
# from mlflow.types import DataType

# # Manual definition that exactly matches the inferred signature structure
# input_schema_exact = Schema([
#     ColSpec(DataType.string, "image_source")
# ])

# output_schema_exact = Schema([
#     ColSpec(DataType.string, "name"),
#     ColSpec(DataType.long, "class"),
#     ColSpec(DataType.double, "confidence"),
#     ColSpec(DataType.double, "x1"),
#     ColSpec(DataType.double, "x2"),
#     ColSpec(DataType.double, "y1"),
#     ColSpec(DataType.double, "y2"),
#     ColSpec(DataType.string, "segments_x"),
#     ColSpec(DataType.string, "segments_y")
# ])

# # Create the exact manual signature
# exact_manual_signature = ModelSignature(inputs=input_schema_exact, outputs=output_schema_exact)

# print("Original inferred signature:")
# print(signature)
# print("\n" + "="*60 + "\n")
# print("Manual signature (exact match):")
# print(exact_manual_signature)
# print("\n" + "="*60 + "\n")
# print("Verification - Are they identical?")
# print(f"Input schemas match: {signature.inputs == exact_manual_signature.inputs}")
# print(f"Output schemas match: {signature.outputs == exact_manual_signature.outputs}")
# print(f"Signatures are identical: {signature == exact_manual_signature}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Helper Functions

# COMMAND ----------

# DBTITLE 1,helper functions
def setup():
    """Initialize the distributed training process group"""
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Fallback for single GPU
        rank = 0
        world_size = 1
        local_rank = 0

    # Initialize process group
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return rank, world_size, device
  
def cleanup():
    """Clean up the distributed training process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

# COMMAND ----------

import serverless_gpu
from serverless_gpu import distributed

# COMMAND ----------

# DBTITLE 1,Set MLflow Experiment Parameters
# We set the experiment details here
import os

os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")

os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")

experiment = mlflow.set_experiment(experiment_name)


# COMMAND ----------

data_yaml_path

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training kickoff with wksp I/O

# COMMAND ----------

# DBTITLE 1,cleaned version - v2a with 8 GPUs vs prior 8 to test more nodes scalability
settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

print('data_yaml_path is:', data_yaml_path)

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)


@distributed(gpus=8, gpu_type='A10', remote=True)
#: -----------worker func: this function is visible to each GPU device.-------------------
def train_fn(world_size = None, parent_run_id = None):
    try:
        from ultralytics.utils import RANK, LOCAL_RANK

        # Setup distributed training
        rank, world_size, device = setup()

        print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
        print(f"Rank: {RANK}, World Size: {world_size}, Device: {LOCAL_RANK}")

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")


        ############################
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # use 1 for synchronization operation, debugging model prefers this.
        os.environ["NCCL_DEBUG"] = "INFO" # "WARN" # for more debugging info on the NCCL side.
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
        os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
        # We set the experiment details here
        experiment = mlflow.set_experiment(experiment_name)
        print('data_yaml_path is:', data_yaml_path)
        
        #
        # with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run():
            model = YOLO(f"yolo11n-seg.pt")
            model.train(
                # task="detect",
                batch=8, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70). Note, with multi-GPU, only integer works. Others modes all throw errors.
                device=[LOCAL_RANK], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
                data=data_yaml_path,
                epochs=50,
                # project=f'{tmp_project_location}', # local VM ephermal location
                # project=f'{volume_project_location}', # volume path still wont work
                #
                save=True,
                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                project=WS_PROJ_DIR,
                exist_ok=True,
                #
                fliplr=1,
                flipud=1,
                perspective=0.001,
                degrees=.45
            )
            success = None
            if RANK in (0, -1):
                success = model.val()
                if success:
                    model.export() # ref: https://docs.ultralytics.com/modes/export/#introduction
            

        active_run_id = mlflow.last_active_run().info.run_id
        print("For YOLO autologging, active_run_id is: ", active_run_id)

        # after training is done.
        if not dist.is_initialized():
        # import torch.distributed as dist
            dist.init_process_group("nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"------After training, we have: RANK:{global_rank=} -- LOCAL_RANK:{local_rank=} -- world_size: {world_size=}------")

        if global_rank == 0:
            with mlflow.start_run(run_id=active_run_id) as run:
                mlflow.log_artifact(data_yaml_path, "input_data_yaml")
                # mlflow.log_dict(data, "data.yaml")
                mlflow.log_params({"rank":global_rank})
                mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature) # this succeeded
                #: TODO: we can log more stuff here
        
        return "finished" # can return any picklable object
    
    finally:
        # clean up
        cleanup()


train_fn.distributed(world_size = None, parent_run_id = None) # now can program can run without specifying manually the parameters of world_size and parent_run_id. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### If you need to wipe the ass
# MAGIC
# MAGIC **Warning: If a job/Experiment run (Yes, it is both) stuck forever, you would like to wipe your ass by stopping the run, otherwise the # of GPUs are draining our money. See below.**

# COMMAND ----------

# MAGIC %md
# MAGIC - In some serverless GPU workflows, canceling the notebook/job may leave the MLflow run status as “RUNNING” unless you explicitly terminate the run; use one of the methods above to mark it “KILLED” or “FAILED”. 
# MAGIC 5
# MAGIC
# MAGIC - Run status changes show up in the MLflow Experiments UI and MLflow system tables (for example, system.mlflow.runs_latest). 
# MAGIC 1
# MAGIC
# MAGIC - Terminating an MLflow run does not stop the underlying compute; use Jobs API runs/cancel to stop a job run if the workload is still executing. 
# MAGIC 2

# COMMAND ----------

# DBTITLE 1,Uncomment if you want to manually cancel the above Experiment Run
# From outside (programmatically mark an existing MLflow run terminated)
from mlflow import MlflowClient
client = MlflowClient()
client.set_terminated("6a1b4d610dbc42c497db73d835fe98b0", status="KILLED")  # or "FAILED"/"FINISHED"

# COMMAND ----------

# DBTITLE 1,Uncomment if you want to manually cancel the job
#: Job is job, experiment is experiment. killing way is different...

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()

# # Cancel a single run
# w.jobs.cancel_run(run_id="571169593303805") #6a1b4d610dbc42c497db73d835fe98b0

# # Or cancel all active runs for a job
# # w.jobs.cancel_all_runs(job_id=<job_id>)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training kick off with volume I/O

# COMMAND ----------

# DBTITLE 1,move .yaml and datasets folder to volume for better governance
# Copy data.yaml to UC volume
if not path_exists(f"{YOLO_DATA_DIR}/data.yaml"):
  dbutils.fs.cp(f"file:{WS_PROJ_DIR}/data.yaml", f"{YOLO_DATA_DIR}/data.yaml")

# Copy datasets folder to UC volume (recursive)
if not path_exists(f"{YOLO_DATA_DIR}/datasets"):
  dbutils.fs.cp(f"file:{WS_PROJ_DIR}/datasets", f"{YOLO_DATA_DIR}/datasets", recurse=True)

# COMMAND ----------

data_yaml_path = f"{YOLO_DATA_DIR}/data.yaml"

# COMMAND ----------

PROJECT_PATH

# COMMAND ----------

# DBTITLE 1,run with switching project location to use volume and load data from volume
settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

print('data_yaml_path is:', data_yaml_path)

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)


@distributed(gpus=8, gpu_type='A10', remote=True)
#: -----------worker func: this function is visible to each GPU device.-------------------
def train_fn(world_size = None, parent_run_id = None):
    try:
        from ultralytics.utils import RANK, LOCAL_RANK

        # Setup distributed training
        rank, world_size, device = setup()

        print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
        print(f"Rank: {RANK}, World Size: {world_size}, Device: {LOCAL_RANK}")

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")


        ############################
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # use 1 for synchronization operation, debugging model prefers this.
        os.environ["NCCL_DEBUG"] = "INFO" # "WARN" # for more debugging info on the NCCL side.
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
        os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
        # We set the experiment details here
        experiment = mlflow.set_experiment(experiment_name)
        print('data_yaml_path is:', data_yaml_path)
        
        #
        # with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run():
            model = YOLO(f"yolo11n-seg.pt")
            model.train(
                # task="detect",
                batch=8, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70). Note, with multi-GPU, only integer works. Others modes all throw errors.
                device=[LOCAL_RANK], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
                data=data_yaml_path,
                epochs=50,
                # project=f'{tmp_project_location}', # local VM ephermal location
                # project=f'{volume_project_location}', # volume path still wont work
                #
                save=True,
                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                project=PROJECT_PATH,
                exist_ok=True,
                #
                fliplr=1,
                flipud=1,
                perspective=0.001,
                degrees=.45
            )
            success = None
            if RANK in (0, -1):
                success = model.val()
                if success:
                    model.export() # ref: https://docs.ultralytics.com/modes/export/#introduction
            

        active_run_id = mlflow.last_active_run().info.run_id
        print("For YOLO autologging, active_run_id is: ", active_run_id)

        # after training is done.
        if not dist.is_initialized():
        # import torch.distributed as dist
            dist.init_process_group("nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"------After training, we have: RANK:{global_rank=} -- LOCAL_RANK:{local_rank=} -- world_size: {world_size=}------")

        if global_rank == 0:
            with mlflow.start_run(run_id=active_run_id) as run:
                mlflow.log_artifact(data_yaml_path, "input_data_yaml")
                # mlflow.log_dict(data, "data.yaml")
                mlflow.log_params({"rank":global_rank})
                mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature) # this succeeded
                #: TODO: we can log more stuff here
        
        return "finished" # can return any picklable object
    
    finally:
        # clean up
        cleanup()


train_fn.distributed(world_size = None, parent_run_id = None) # now can program can run without specifying manually the parameters of world_size and parent_run_id. 

# COMMAND ----------

# MAGIC %md
# MAGIC ________

# COMMAND ----------

# DBTITLE 1,todo
# MAGIC %md
# MAGIC # Issues Report
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Node 0 Log and Error analysis
# MAGIC
# MAGIC Node 0 stuck forever at trying to sending data to each working node. We have in total 8 A10 GPU nodes (1 GPU per node).
# MAGIC
# MAGIC Node 0 log is below
# MAGIC ```
# MAGIC ip-10-153-159-148:6132:6132 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
# MAGIC ip-10-153-159-148:6132:6132 [0] NCCL INFO Bootstrap: Using eth0:10.153.159.148<0>
# MAGIC ip-10-153-159-148:6132:6132 [0] NCCL INFO cudaDriverVersion 12040
# MAGIC ip-10-153-159-148:6132:6132 [0] NCCL INFO NCCL version 2.26.2+cuda12.2
# MAGIC ip-10-153-159-148:6132:6132 [0] NCCL INFO Comm config Blocking set to 1
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v10)
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.15.0
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Using Libfabric version 2.1
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Using CUDA driver version 12040 with runtime 12060
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Configuring AWS-specific options
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Setting provider_filter to efa
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Setting NCCL_NVLSTREE_MAX_CHUNKSIZE to 512KiB
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Setting NCCL_NVLS_CHUNKSIZE to 512KiB
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI Internode latency set at 75.0 us
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/OFI No eligible providers were found
# MAGIC
# MAGIC [2025-10-14 21:34:17] ip-10-153-159-148:6132:6289 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):263 NCCL WARN NET/OFI Unable to find a protocol that worked.  Failing initialization.
# MAGIC
# MAGIC [2025-10-14 21:34:17] ip-10-153-159-148:6132:6289 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):354 NCCL WARN NET/OFI aws-ofi-nccl initialization failed
# MAGIC
# MAGIC [2025-10-14 21:34:17] ip-10-153-159-148:6132:6289 [0] ncclResult_t nccl_net_ofi_init_v2(ncclDebugLogger_t):155 NCCL WARN NET/OFI Initializing plugin failed
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/IB : No device found.
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/IB : Using [RO]; OOB eth0:10.153.159.148<0>
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO NET/Socket : Using [0]eth0:10.153.159.148<0>
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so. 
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO Using network Socket
# MAGIC ip-10-153-159-148:6132:6289 [0] NCCL INFO ncclCommInitRankConfig comm 0x39a74630 rank 0 nranks 8 cudaDev 0 nvmlDev 0 busId 1e0 commId 0x260da5686b91fcc7 - Init START
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis of Node 0 NCCL and OFI Log: Key Insights and Causes
# MAGIC
# MAGIC **Main takeaway:**  
# MAGIC The log from your GPU node shows that the NCCL (NVIDIA Collective Communications Library) initialization attempted to use advanced network transports (Libfabric, aws-ofi-nccl) but failed to find a suitable provider, falling back to basic socket networking. This will likely result in **reduced communication performance** across nodes, especially in a distributed or multi-GPU setup.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Detailed Explanation
# MAGIC
# MAGIC ### 1. NCCL Initialization Sequence
# MAGIC
# MAGIC - The process starts with NCCL being directed to use the `eth0` network interface, which is standard for AWS EC2 instances.
# MAGIC - CUDA driver (version 12040) and NCCL (version 2.26.2+cuda12.2) are detected.
# MAGIC - The AWS libfabric plugin (`aws-ofi-nccl`) loads but immediately logs warnings about missing symbols for various plugin versions (`ncclCollNetPlugin_v10`, etc.).
# MAGIC
# MAGIC ### 2. OFI/Libfabric Plugin Failure
# MAGIC
# MAGIC - The logs indicate:  
# MAGIC   - **"No eligible providers were found"** following configuration for the EFA (Elastic Fabric Adapter) provider.
# MAGIC   - **NCCL WARN** entries confirm:  
# MAGIC     - No working protocol was found.
# MAGIC     - The `aws-ofi-nccl` initialization failed.
# MAGIC     - The plugin could not be initialized.
# MAGIC
# MAGIC **This usually means:**
# MAGIC - The system either lacks EFA hardware support, or
# MAGIC - The EFA software stack is not present, correctly installed, or compatible with this environment, or
# MAGIC - The correct permissions or environment variables for EFA use are not in place.
# MAGIC
# MAGIC ### 3. Fallback to Basic Networking
# MAGIC
# MAGIC - After plugin failure, NCCL attempts to use InfiniBand (`NET/IB`) but finds no devices.
# MAGIC - It ultimately proceeds using Ethernet sockets over `eth0`, with lines like:
# MAGIC
# MAGIC   ```
# MAGIC   NCCL INFO Using network Socket
# MAGIC   ```
# MAGIC
# MAGIC - The fallback is successful—NCCL will still work, but with significantly lower throughput and higher latency compared to EFA.
# MAGIC
# MAGIC ### 4. Profiler Plugin Missing (Minor Issue)
# MAGIC
# MAGIC - A message about `libnccl-profiler.so` not being found is displayed, but this is only an auxiliary plugin for profiling.
# MAGIC
# MAGIC ### 5. Communication Rank Setup
# MAGIC
# MAGIC - The last entries show that NCCL continues to initialize communicator ranks, so basic operation will proceed, just without advanced network acceleration.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Key Learnings & Troubleshooting
# MAGIC
# MAGIC **This log teaches:**
# MAGIC - NCCL is configured to optimize for EFA/libfabric, which is optimal in AWS for high-performance distributed deep learning.
# MAGIC - The failure means you will not benefit from these optimizations, which can impact your multi-node or multi-GPU training speed.
# MAGIC - Fallback to sockets is a safe path, but not ideal for large-scale distributed training workloads.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ### What to Check or Fix
# MAGIC
# MAGIC - **Are you intending to use EFA?**  
# MAGIC   Ensure your instance type and AMI (OS image) support EFA, and that the EFA drivers and libraries (`libfabric`, `aws-ofi-nccl`, etc.) are properly installed and loaded.
# MAGIC - **If using EFA:**  
# MAGIC   - Check `fi_info` command output for EFA support.
# MAGIC   - Ensure security groups, instance placement, and VPC settings allow EFA.
# MAGIC   - Make sure environment variables like `FI_PROVIDER=efa` and `NCCL_NET_GDR_LEVEL=PHB` are set when needed.
# MAGIC - **If not using EFA:**  
# MAGIC   You can ignore the warnings, but expect reduced communication performance.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ### Reference Table: Fallback Summary
# MAGIC
# MAGIC | Step                 | Status              | Impact                    |
# MAGIC |----------------------|---------------------|---------------------------|
# MAGIC | EFA/libfabric plugin | Failed (no provider) | No network acceleration    |
# MAGIC | InfiniBand           | Not found           | No Infiniband support      |
# MAGIC | Socket (Ethernet)    | Active              | Lower speed, higher latency|
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC **Summary:**  
# MAGIC Your system is falling back from EFA/infiniband to basic ethernet sockets for network communication. Distributed performance will be limited. Check system, instance, and network configuration if EFA is intended; otherwise, operation will continue, but not at optimal speed.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Node 1-7 Log(s) and Error analysis
# MAGIC
# MAGIC Since other worker nodes all have the very similar logs, here I just put Node 3's log below, since it has exited with errors.
# MAGIC
# MAGIC ```
# MAGIC /databricks/python3/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
# MAGIC   import pynvml  # type: ignore[import]
# MAGIC /databricks/python3/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
# MAGIC   import pynvml  # type: ignore[import]
# MAGIC Warning: serverless_gpu is in Beta. The API is subject to change.
# MAGIC   Error =>  [Errno 2] No such file or directory: 'rocminfo'
# MAGIC 2025/10/14 21:32:49 INFO mlflow.system_metrics.system_metrics_monitor: Started monitoring system metrics.
# MAGIC Rank: 3, World Size: 8, Device: cuda:0
# MAGIC Rank: 3, World Size: 8, Device: 0
# MAGIC PyTorch version: 2.7.1+cu126
# MAGIC CUDA available: True
# MAGIC CUDA device count: 1
# MAGIC Current CUDA device: 0
# MAGIC data_yaml_path is: /Workspace/Users/yang.yang@databricks.com/SGC_YOLO/data.yaml
# MAGIC 2025/10/14 21:32:49 INFO mlflow.system_metrics.system_metrics_monitor: Started monitoring system metrics.
# MAGIC
# MAGIC [KDownloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt to 'yolo11n-seg.pt': 100% ━━━━━━━━━━━━ 5.9MB 83.9MB/s 0.1s
# MAGIC [KDownloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt to 'yolo11n-seg.pt': 100% ━━━━━━━━━━━━ 5.9MB 83.8MB/s 0.1s
# MAGIC
# MAGIC [KDownloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 15.6MB/s 0.0s
# MAGIC [KDownloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 15.5MB/s 0.0s
# MAGIC 🏃 View run jobTaskRun-571169593303805 at: https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/3873581079937048/runs/6a1b4d610dbc42c497db73d835fe98b0
# MAGIC 🧪 View experiment at: https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/3873581079937048
# MAGIC 2025/10/14 21:32:51 INFO mlflow.system_metrics.system_metrics_monitor: Stopping system metrics monitoring...
# MAGIC 2025/10/14 21:32:51 INFO mlflow.system_metrics.system_metrics_monitor: Successfully terminated system metrics monitoring!
# MAGIC 2025/10/14 21:32:51 INFO mlflow.system_metrics.system_metrics_monitor: Stopping system metrics monitoring...
# MAGIC 2025/10/14 21:32:51 INFO mlflow.system_metrics.system_metrics_monitor: Successfully terminated system metrics monitoring!
# MAGIC [rank3]: [rank3]: Traceback (most recent call last):
# MAGIC [rank3]: [rank3]:   File "/Workspace/Users/yang.yang@databricks.com/.serverless_gpu/pkls/train_fn/80d01670-0268-479b-b7d6-0bfc4429de91/_air.py", line 10, in <module>
# MAGIC [rank3]: [rank3]:     _deserialize_and_run("train_fn", "80d01670-0268-479b-b7d6-0bfc4429de91", "/Workspace/Users/yang.yang@databricks.com/.serverless_gpu/pkls")
# MAGIC [rank3]: [rank3]:   File "/databricks/python3/lib/python3.12/site-packages/serverless_gpu/script.py", line 155, in _deserialize_and_run
# MAGIC [rank3]: [rank3]:     raise output
# MAGIC [rank3]: [rank3]:   File "/databricks/python3/lib/python3.12/site-packages/serverless_gpu/script.py", line 149, in _deserialize_and_run
# MAGIC [rank3]: [rank3]:     output = func(*args, **kwargs)
# MAGIC [rank3]: [rank3]:              ^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]:   File "/databricks/python/lib/python3.12/site-packages/serverless_gpu/launcher.py", line 170, in wrapped_fun
# MAGIC [rank3]: [rank3]:     return func(*args, **kwargs)
# MAGIC [rank3]: [rank3]:            ^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]:   File "/home/spark-1b7371fa-22bd-444f-939b-6d/.ipykernel/9980/command-3873581079936889-965044199", line 42, in train_fn
# MAGIC [rank3]: [rank3]:   File "/.pythonenv/lib/python3.12/site-packages/ultralytics/engine/model.py", line 795, in train
# MAGIC [rank3]: [rank3]:     self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
# MAGIC [rank3]: [rank3]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]:   File "/.pythonenv/lib/python3.12/site-packages/ultralytics/models/yolo/segment/train.py", line 42, in __init__
# MAGIC [rank3]: [rank3]:     super().__init__(cfg, overrides, _callbacks)
# MAGIC [rank3]: [rank3]:   File "/.pythonenv/lib/python3.12/site-packages/ultralytics/models/yolo/detect/train.py", line 65, in __init__
# MAGIC [rank3]: [rank3]:     super().__init__(cfg, overrides, _callbacks)
# MAGIC [rank3]: [rank3]:   File "/.pythonenv/lib/python3.12/site-packages/ultralytics/engine/trainer.py", line 157, in __init__
# MAGIC [rank3]: [rank3]:     with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
# MAGIC [rank3]: [rank3]:   File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
# MAGIC [rank3]: [rank3]:     next(self.gen)
# MAGIC [rank3]: [rank3]:   File "/.pythonenv/lib/python3.12/site-packages/ultralytics/utils/torch_utils.py", line 68, in torch_distributed_zero_first
# MAGIC [rank3]: [rank3]:     dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
# MAGIC [rank3]: [rank3]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]:   File "/databricks/python3/lib/python3.12/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
# MAGIC [rank3]: [rank3]:     return func(*args, **kwargs)
# MAGIC [rank3]: [rank3]:            ^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]:   File "/databricks/python3/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 4635, in barrier
# MAGIC [rank3]: [rank3]:     work = group.barrier(opts=opts)
# MAGIC [rank3]: [rank3]:            ^^^^^^^^^^^^^^^^^^^^^^^^
# MAGIC [rank3]: [rank3]: torch.distributed.DistBackendError: [3] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: failed to recv, got 0 bytes
# MAGIC [rank3]: [rank3]: Exception raised from recvBytes at /pytorch/torch/csrc/distributed/c10d/Utils.hpp:678 (most recent call first):
# MAGIC [rank3]: [rank3]: frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7ff3d2b785e8 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libc10.so)
# MAGIC [rank3]: [rank3]: frame #1: <unknown function> + 0x5ba8bfe (0x7ff3bbefabfe in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #2: <unknown function> + 0x5bab100 (0x7ff3bbefd100 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #3: c10d::TCPStore::doWait(c10::ArrayRef<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::chrono::duration<long, std::ratio<1l, 1000l> >) + 0x28b (0x7ff3bbef7cab in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #4: c10d::TCPStore::doGet(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x33 (0x7ff3bbef7fa3 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #5: c10d::TCPStore::get(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xab (0x7ff3bbef908b in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #6: c10d::PrefixStore::get(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x2f (0x7ff3bbea794f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #7: c10d::PrefixStore::get(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x2f (0x7ff3bbea794f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #8: c10d::PrefixStore::get(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x2f (0x7ff3bbea794f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #9: c10d::ProcessGroupNCCL::broadcastUniqueNCCLID(ncclUniqueId*, bool, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int) + 0x16d (0x7ff37d5ede7d in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
# MAGIC [rank3]: [rank3]: frame #10: c10d::ProcessGroupNCCL::initNCCLComm(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, c10::Device&, c10d::OpType, int, bool) + 0x16bf (0x7ff37d5fdf6f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
# MAGIC [rank3]: [rank3]: frame #11: <unknown function> + 0x11ecf83 (0x7ff37d602f83 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
# MAGIC [rank3]: [rank3]: frame #12: c10d::ProcessGroupNCCL::allreduce_impl(at::Tensor&, char const*, c10d::AllreduceOptions const&) + 0xee (0x7ff37d60377e in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
# MAGIC [rank3]: [rank3]: frame #13: c10d::ProcessGroupNCCL::barrier(c10d::BarrierOptions const&) + 0x683 (0x7ff37d6133a3 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so)
# MAGIC [rank3]: [rank3]: frame #14: <unknown function> + 0x5b4363f (0x7ff3bbe9563f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #15: <unknown function> + 0x5b52cfe (0x7ff3bbea4cfe in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #16: <unknown function> + 0x52125b0 (0x7ff3bb5645b0 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #17: <unknown function> + 0x5b61a02 (0x7ff3bbeb3a02 in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #18: <unknown function> + 0x5b6293d (0x7ff3bbeb493d in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so)
# MAGIC [rank3]: [rank3]: frame #19: <unknown function> + 0xc6ea8f (0x7ff3cade9a8f in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_python.so)
# MAGIC [rank3]: [rank3]: frame #20: <unknown function> + 0x37f2dd (0x7ff3ca4fa2dd in /databricks/python3/lib/python3.12/site-packages/torch/lib/libtorch_python.so)
# MAGIC [rank3]: [rank3]: frame #21: /databricks/python3/bin/python() [0x581a6f]
# MAGIC [rank3]: [rank3]: frame #22: _PyObject_MakeTpCall + 0x13e (0x5493be in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #23: _PyEval_EvalFrameDefault + 0xadf (0x5d68bf in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #24: /databricks/python3/bin/python() [0x5554a6]
# MAGIC [rank3]: [rank3]: frame #25: /databricks/python3/bin/python() [0x5d3afc]
# MAGIC [rank3]: [rank3]: frame #26: _PyEval_EvalFrameDefault + 0x212e (0x5d7f0e in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #27: /databricks/python3/bin/python() [0x54cf04]
# MAGIC [rank3]: [rank3]: frame #28: PyObject_Vectorcall + 0x35 (0x549cf5 in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #29: _PyEval_EvalFrameDefault + 0xadf (0x5d68bf in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #30: _PyObject_Call_Prepend + 0x18a (0x54ac0a in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #31: /databricks/python3/bin/python() [0x59da4f]
# MAGIC [rank3]: [rank3]: frame #32: /databricks/python3/bin/python() [0x599513]
# MAGIC [rank3]: [rank3]: frame #33: _PyObject_MakeTpCall + 0x13e (0x5493be in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #34: _PyEval_EvalFrameDefault + 0xadf (0x5d68bf in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #35: PyEval_EvalCode + 0x15b (0x5d4dab in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #36: /databricks/python3/bin/python() [0x607fc2]
# MAGIC [rank3]: [rank3]: frame #37: /databricks/python3/bin/python() [0x6b4393]
# MAGIC [rank3]: [rank3]: frame #38: _PyRun_SimpleFileObject + 0x1aa (0x6b40fa in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #39: _PyRun_AnyFileObject + 0x4f (0x6b3f2f in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #40: Py_RunMain + 0x3b5 (0x6bbf45 in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #41: Py_BytesMain + 0x2d (0x6bba2d in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: frame #42: <unknown function> + 0x2a1ca (0x7ff445c3c1ca in /usr/lib/x86_64-linux-gnu/libc.so.6)
# MAGIC [rank3]: [rank3]: frame #43: __libc_start_main + 0x8b (0x7ff445c3c28b in /usr/lib/x86_64-linux-gnu/libc.so.6)
# MAGIC [rank3]: [rank3]: frame #44: _start + 0x25 (0x656a35 in /databricks/python3/bin/python)
# MAGIC [rank3]: [rank3]: . This may indicate a possible application crash on rank 0 or a network set up issue.
# MAGIC ERROR:__main__:Rank 3 crashed with exit code 1.
# MAGIC ERROR:__main__:Global rank 3 (PID 6138) exited with code 1
# MAGIC Warning: serverless_gpu is in Beta. The API is subject to change.
# MAGIC Global rank 3 (PID 6138) exited with code 1
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis of Node 3 Log: Distributed Training Communication Failure
# MAGIC
# MAGIC **Main takeaway:**  
# MAGIC The log from Node 3 reveals a **critical distributed training failure** where rank 3 cannot communicate with rank 0 (the master node) through NCCL. This is causing your 8-node training job to crash due to a breakdown in the key-value store communication system that coordinates distributed processes.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Key Findings from the Log
# MAGIC
# MAGIC ### 1. **Distributed Training Setup Information**
# MAGIC - **Environment:** Databricks serverless GPU (Beta feature)[1]
# MAGIC - **Framework:** PyTorch 2.7.1 with CUDA 12.6 support
# MAGIC - **Training setup:** 8 nodes total, with Node 3 being rank 3 out of 8 total ranks
# MAGIC - **Model:** YOLO11n segmentation model with Ultralytics framework
# MAGIC - **MLflow integration:** Active system metrics monitoring
# MAGIC
# MAGIC ### 2. **The Critical Error Sequence**
# MAGIC
# MAGIC The failure occurs during the **torch_distributed_zero_first** context manager execution[2][3]:
# MAGIC
# MAGIC ```
# MAGIC torch.distributed.DistBackendError: [3] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: failed to recv, got 0 bytes
# MAGIC ```
# MAGIC
# MAGIC **What this means:**
# MAGIC - Node 3 is trying to establish NCCL communication with the master node (rank 0)
# MAGIC - The key-value store (TCPStore) that coordinates distributed processes cannot retrieve the unique NCCL ID[2][3]
# MAGIC - The communication channel returned 0 bytes, indicating either network failure or the master process crashed[4]
# MAGIC
# MAGIC ### 3. **Root Cause Analysis**
# MAGIC
# MAGIC **Primary causes based on the error pattern:**
# MAGIC 1. **Master node (rank 0) failure or premature exit** - The log explicitly states "This may indicate a possible application crash on rank 0 or a network set up issue"[2][3]
# MAGIC 2. **Network connectivity issues** between nodes in the distributed setup[5][6]
# MAGIC 3. **NCCL communication breakdown** due to the network transport issues identified in your Node 0 log (EFA/libfabric failure)[2]
# MAGIC
# MAGIC ### 4. **Connection to Previous Node 0 Log**
# MAGIC
# MAGIC This failure is **directly related** to the NCCL networking issues from your Node 0 log:
# MAGIC - Node 0 fell back to basic socket networking instead of EFA
# MAGIC - The degraded network performance likely caused communication timeouts
# MAGIC - Without proper high-speed interconnects, the distributed coordination became unstable[6][7]
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Technical Breakdown
# MAGIC
# MAGIC ### **The torch_distributed_zero_first Context**
# MAGIC This Ultralytics utility ensures only rank 0 downloads datasets while other ranks wait[8][9]. The failure occurs when:
# MAGIC 1. Rank 3 calls `dist.barrier()` to synchronize with other processes
# MAGIC 2. The barrier operation requires NCCL communicator setup
# MAGIC 3. NCCL tries to get the unique ID from rank 0's key-value store
# MAGIC 4. The TCP connection fails with "failed to recv, got 0 bytes"
# MAGIC
# MAGIC ### **Impact on Training**
# MAGIC - **Complete job termination:** All 8 nodes crash when any single node fails[10][11]
# MAGIC - **No fault tolerance:** Distributed training requires all processes to participate[10]
# MAGIC - **Resource waste:** All allocated serverless GPU resources become unusable
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Troubleshooting Recommendations
# MAGIC
# MAGIC ### **Immediate Actions**
# MAGIC 1. **Check Node 0 status:** Verify if the master node is still running and accessible
# MAGIC 2. **Network diagnostics:** Test connectivity between all 8 nodes
# MAGIC 3. **NCCL debugging:** Add `NCCL_DEBUG=INFO` to get detailed communication logs[5][12]
# MAGIC
# MAGIC ### **Configuration Fixes**
# MAGIC 1. **Force socket transport:** Set `NCCL_SOCKET_IFNAME=eth0` explicitly[7][5]
# MAGIC 2. **Disable P2P if needed:** Add `NCCL_P2P_DISABLE=1` for compatibility[7]
# MAGIC 3. **Increase timeouts:** Consider setting longer NCCL timeout values
# MAGIC
# MAGIC ### **Alternative Approaches**
# MAGIC 1. **Use Gloo backend:** Switch from NCCL to Gloo for CPU-based communication[10]
# MAGIC 2. **Reduce node count:** Test with fewer nodes to isolate the issue
# MAGIC 3. **Check Databricks limits:** Verify serverless GPU multi-node constraints[1]
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Key Learnings
# MAGIC
# MAGIC **This log teaches you:**
# MAGIC 1. **Distributed training fragility:** Any single node failure kills the entire job[10][11]
# MAGIC 2. **Network dependency:** Multi-node training heavily relies on stable, high-performance networking
# MAGIC 3. **NCCL limitations:** Without proper network acceleration (EFA), NCCL becomes unreliable at scale
# MAGIC 4. **Databricks serverless constraints:** The Beta nature of serverless GPU may have stability limitations[1]
# MAGIC 5. **Coordination complexity:** The key-value store mechanism is a single point of failure
# MAGIC
# MAGIC **Most importantly:** Your Node 0's network fallback to basic sockets (from the previous log) is likely the root cause of this Node 3 communication failure. The degraded network performance cannot support reliable 8-node coordination.
# MAGIC
# MAGIC Sources
# MAGIC [1] Serverless GPU compute | Databricks on AWS https://docs.databricks.com/aws/en/compute/serverless/gpu
# MAGIC [2] torch.distributed.DistBackendError: NCCL error in: ../torch/csrc ... https://github.com/pytorch/pytorch/issues/111187
# MAGIC [3] distributed training: using GPU 0 to perform barrier as devices used ... https://github.com/hiyouga/LLaMA-Factory/issues/5769
# MAGIC [4] recv() returns 0 - sockets - Stack Overflow https://stackoverflow.com/questions/10526382/recv-returns-0
# MAGIC [5] torch.distributed.DistBackendError: NCCL error - PyTorch Forums https://discuss.pytorch.org/t/torch-distributed-distbackenderror-nccl-error/191509
# MAGIC [6] torch.distributed.DistBackendError: NCCL error · Issue #715 - GitHub https://github.com/aws/aws-ofi-nccl/issues/715
# MAGIC [7] Distributed training with TorchDistributor | Databricks on AWS https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor
# MAGIC [8] SettingsManager with torch_distributed_zero_fitst(RANK) #6532 https://github.com/ultralytics/ultralytics/issues/6532
# MAGIC [9] YoloV7 - Multi-GPU constantly gives RunTime Error - Stack Overflow https://stackoverflow.com/questions/77382185/yolov7-multi-gpu-constantly-gives-runtime-error
# MAGIC [10] How to handle exception in DistributedDataParallel? - distributed https://discuss.pytorch.org/t/how-to-handle-exception-in-distributeddataparallel/42026
# MAGIC [11] Distributed Training quits if any worker node fail... - 120383 https://community.databricks.com/t5/machine-learning/distributed-training-quits-if-any-worker-node-fails/td-p/120383
# MAGIC [12] Troubleshooting — NCCL 2.28.3 documentation - NVIDIA Docs https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html
# MAGIC [13] Distributed training got stuck every few seconds - PyTorch Forums https://discuss.pytorch.org/t/distributed-training-got-stuck-every-few-seconds/132483
# MAGIC [14] NCCL AlltoAll hang at first step, guess caused by zero byte send/recv https://github.com/NVIDIA/nccl/issues/696
# MAGIC [15] How to solve the famous `unhandled cuda error, NCCL version 2.7.8 ... https://stackoverflow.com/questions/66807131/how-to-solve-the-famous-unhandled-cuda-error-nccl-version-2-7-8-error
# MAGIC [16] Can not realize distributed training across machines with DDP https://discuss.pytorch.org/t/can-not-realize-distributed-training-across-machines-with-ddp/144519
# MAGIC [17] Distributed communication package - torch.distributed - PyTorch https://docs.pytorch.org/docs/stable/distributed.html
# MAGIC [18] NCCL failing with A100 GPUs, works fine with V100 GPUs - distributed https://discuss.pytorch.org/t/nccl-failing-with-a100-gpus-works-fine-with-v100-gpus/201388
# MAGIC [19] python - I want to use the distributed package in PyTorch for point-to ... https://stackoverflow.com/questions/79335746/i-want-to-use-the-distributed-package-in-pytorch-for-point-to-point-communicatio
# MAGIC [20] Error: Some NCCL operations have failed or timed out https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out
# MAGIC [21] NCCL error when running distributed training - PyTorch Forums https://discuss.pytorch.org/t/nccl-error-when-running-distributed-training/129301
# MAGIC [22] Encountered NCCL communicator error while using multi GPU training https://forums.developer.nvidia.com/t/encountered-nccl-communicator-error-while-using-multi-gpu-training/286563
# MAGIC [23] distributed training not starting · Issue #65121 - GitHub https://github.com/pytorch/pytorch/issues/65121
# MAGIC [24] Torch distributed not working on two machines [nccl backend] https://discuss.pytorch.org/t/torch-distributed-not-working-on-two-machines-nccl-backend/87659
# MAGIC [25] Unable to make nccl work - Container - NVIDIA Developer Forums https://forums.developer.nvidia.com/t/unable-to-make-nccl-work/276722
# MAGIC [26] Multi-GPU Training error: torch.distributed.run · Issue #849 - GitHub https://github.com/ultralytics/ultralytics/issues/849
# MAGIC [27] Multi-node multi-GPU training wont run after loading images #12429 https://github.com/ultralytics/yolov5/issues/12429
# MAGIC [28] torch.distributed.DistBackendError · Issue #20284 - GitHub https://github.com/ultralytics/ultralytics/issues/20284
# MAGIC [29] Serverless GPU compute - Azure Databricks - Microsoft Learn https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/gpu
# MAGIC [30] Pack ERROR mismatch - vision - PyTorch Forums https://discuss.pytorch.org/t/pack-error-mismatch/170294
# MAGIC [31] Normal then slow then crashing training - YOLO - Ultralytics https://community.ultralytics.com/t/normal-then-slow-then-crashing-training/1203
# MAGIC [32] ML Training Tip Of The Week #1: Optimizing GPU ... - 86677 https://community.databricks.com/t5/technical-blog/ml-training-tip-of-the-week-1-optimizing-gpu-utilization-in/ba-p/86677
# MAGIC [33] Error when fine-tuning on multi-gpu - Hugging Face Forums https://discuss.huggingface.co/t/error-when-fine-tuning-on-multi-gpu/141196
# MAGIC [34] Reference for ultralytics/utils/torch_utils.py https://docs.ultralytics.com/reference/utils/torch_utils/
# MAGIC [35] Troubleshooting Common YOLO Issues - Ultralytics YOLO Docs https://docs.ultralytics.com/guides/yolo-common-issues/
# MAGIC [36] Job Serverless Issues : r/databricks - Reddit https://www.reddit.com/r/databricks/comments/1j3ktal/job_serverless_issues/
# MAGIC [37] Yolov5 to yolov11 - YOLO - Ultralytics https://community.ultralytics.com/t/yolov5-to-yolov11/671
# MAGIC [38] Source code for transformers.trainer_pt_utils - Hugging Face https://huggingface.co/transformers/v4.4.2/_modules/transformers/trainer_pt_utils.html
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Overall Analysis by taking into consider that we have loaded a Big image dataset for training, additional inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis: Large Image Dataset Loading Impact on Multi-Node Training
# MAGIC
# MAGIC **Main takeaway:**  
# MAGIC Loading large image datasets in multi-node YOLO training significantly **amplifies the distributed coordination issues** you're experiencing. The combination of massive dataset loading, NCCL communication problems, and the `torch_distributed_zero_first` synchronization mechanism creates a perfect storm for training failures.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## How Large Datasets Compound Your Issues
# MAGIC
# MAGIC ### 1. **Dataset Loading Coordination Problem**
# MAGIC
# MAGIC **The torch_distributed_zero_first Context:**
# MAGIC - This Ultralytics mechanism ensures only rank 0 downloads/processes the dataset while other ranks wait at a barrier[1][2]
# MAGIC - **With large datasets:** The master node (rank 0) takes significantly longer to process images
# MAGIC - **Other nodes:** Must wait indefinitely at the NCCL barrier until rank 0 completes[3][4]
# MAGIC
# MAGIC **Your specific failure pattern:**
# MAGIC ```
# MAGIC with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
# MAGIC     dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
# MAGIC ```
# MAGIC Node 3 fails here because it's waiting for rank 0's dataset processing to complete, but rank 0 likely crashed or timed out during the large dataset loading phase[3][4].
# MAGIC
# MAGIC ### 2. **Memory and I/O Amplification Issues**
# MAGIC
# MAGIC **Multiple Memory Loads:**
# MAGIC - Each of your 8 nodes loads the **entire dataset into memory** simultaneously[5][6]
# MAGIC - Large image datasets can consume **gigabytes per node**
# MAGIC - **Memory pressure** increases dramatically with dataset size[7][8]
# MAGIC
# MAGIC **Network I/O Bottlenecks:**
# MAGIC - Dataset downloading/loading creates **massive network traffic**
# MAGIC - Your nodes are already using degraded socket networking (from Node 0 log)
# MAGIC - **Concurrent large file transfers** across 8 nodes overwhelm the network infrastructure[9]
# MAGIC
# MAGIC ### 3. **NCCL Timeout Cascading Effects**
# MAGIC
# MAGIC **Documented timeout patterns with large datasets:**
# MAGIC - NCCL operations regularly time out during large dataset initialization[9][4][10]
# MAGIC - **Default timeout (600 seconds)** is often insufficient for big datasets[9][11]
# MAGIC - **Progressive slowdown:** Initial images load quickly, then dramatically slow down[7]
# MAGIC
# MAGIC **Your timeline correlation:**
# MAGIC ```
# MAGIC [2025-10-14 21:34:17] - NCCL communication fails
# MAGIC ```
# MAGIC This timing suggests rank 0 was likely still processing your large dataset when the 10-minute default timeout expired.
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Specific YOLO/Ultralytics Issues with Large Datasets
# MAGIC
# MAGIC ### **Cache Loading Problems**
# MAGIC - YOLO can cache entire datasets in RAM for faster training (`cache=ram`)[4]
# MAGIC - **With large datasets:** Cache loading becomes extremely slow and memory-intensive[7]
# MAGIC - **Multi-node coordination:** All nodes attempt caching simultaneously, causing resource contention[3]
# MAGIC
# MAGIC ### **Progressive Loading Degradation**
# MAGIC Common pattern observed in large dataset scenarios[4][7]:
# MAGIC 1. **Fast initial loading:** First few hundred images process quickly
# MAGIC 2. **Dramatic slowdown:** Loading rate drops from 143.54it/s to 1.25s/it
# MAGIC 3. **Eventual timeout:** NCCL operations time out waiting for coordination
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Technical Root Cause Analysis
# MAGIC
# MAGIC ### **The Perfect Storm Scenario:**
# MAGIC
# MAGIC 1. **Rank 0 starts dataset processing** - Downloads/caches large image dataset
# MAGIC 2. **Ranks 1-7 wait at barrier** - All other nodes pause execution
# MAGIC 3. **Network degradation** - Your socket-only networking (no EFA) struggles with dataset I/O
# MAGIC 4. **Memory pressure builds** - Large dataset loading consumes increasing RAM
# MAGIC 5. **NCCL timeout triggers** - Default 600-second timeout expires
# MAGIC 6. **Cascade failure** - All nodes crash when coordination breaks down
# MAGIC
# MAGIC ### **Why This Particularly Affects YOLO Training:**
# MAGIC
# MAGIC **Ultralytics-specific factors:**
# MAGIC - Heavy reliance on `torch_distributed_zero_first` for dataset coordination[1]
# MAGIC - Aggressive caching strategies that amplify memory usage[4]
# MAGIC - Image preprocessing and augmentation during loading phase
# MAGIC - Multi-resolution image handling increasing I/O complexity
# MAGIC
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Immediate Mitigation Strategies
# MAGIC
# MAGIC ### **1. Dataset Loading Optimizations**
# MAGIC ```python
# MAGIC # Disable caching for large datasets
# MAGIC cache=False  # Instead of cache='ram'
# MAGIC
# MAGIC # Reduce image resolution temporarily
# MAGIC imgsz=416  # Instead of 640
# MAGIC
# MAGIC # Limit dataset size for testing
# MAGIC # Use subset of data to isolate networking issues
# MAGIC ```
# MAGIC
# MAGIC ### **2. NCCL Timeout Adjustments**
# MAGIC ```bash
# MAGIC export NCCL_TIMEOUT=3600  # 1 hour instead of 10 minutes
# MAGIC export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# MAGIC ```
# MAGIC
# MAGIC ### **3. Alternative Loading Strategies**
# MAGIC - **Pre-process datasets** offline to avoid runtime coordination[2]
# MAGIC - **Use streaming datasets** instead of loading everything into memory[12]
# MAGIC - **Implement progressive loading** with checkpointing between dataset chunks
# MAGIC
# MAGIC ### **4. Reduce Node Count Temporarily**
# MAGIC - Test with 2-4 nodes first to isolate the dataset size vs. node count interaction
# MAGIC - Scale up gradually once stability is achieved
# MAGIC
# MAGIC ***
# MAGIC
# MAGIC ## Key Learnings
# MAGIC
# MAGIC **Critical insights from your scenario:**
# MAGIC
# MAGIC 1. **Dataset size is a multiplier** for distributed training complexity - not just an additive factor
# MAGIC 2. **Network transport matters more** with large datasets - your EFA failure becomes critical
# MAGIC 3. **Coordination mechanisms fail** when any single step (dataset loading) takes too long
# MAGIC 4. **Memory scaling is non-linear** - 8 nodes × large dataset ≠ manageable memory usage
# MAGIC 5. **YOLO's optimization features** (caching, preprocessing) become liabilities at scale
# MAGIC
# MAGIC **The large dataset is likely the trigger** that exposed your underlying network infrastructure problems. Without proper high-speed networking (EFA), the combination of dataset I/O and distributed coordination becomes unsustainable.
# MAGIC
# MAGIC Sources
# MAGIC [1] SettingsManager with torch_distributed_zero_fitst(RANK) #6532 https://github.com/ultralytics/ultralytics/issues/6532
# MAGIC [2] utils/datasets.py · akhaliq/Kapao at main - Hugging Face https://huggingface.co/spaces/akhaliq/Kapao/blob/main/utils/datasets.py
# MAGIC [3] DDP: multi node training · Issue #6286 - GitHub https://github.com/ultralytics/ultralytics/issues/6286
# MAGIC [4] NCCL timeout problem on DPP · Issue #7481 · ultralytics/yolov5 https://github.com/ultralytics/yolov5/issues/7481
# MAGIC [5] Distributed training: how to avoid loading dataset in memory N times https://discuss.pytorch.org/t/distributed-training-how-to-avoid-loading-dataset-in-memory-n-times/46940
# MAGIC [6] How to share a cache among multiple subprocesses when using ... https://discuss.pytorch.org/t/how-to-share-a-cache-among-multiple-subprocesses-when-using-pytorch-ddp-training/139969
# MAGIC [7] Cache dataset slowdown/not loading for large dataset on multi-gpu https://github.com/Project-MONAI/MONAI/issues/1589
# MAGIC [8] Lightgbm Trainer for distribute training use too much memory - Ray https://discuss.ray.io/t/lightgbm-trainer-for-distribute-training-use-too-much-memory/21556
# MAGIC [9] NCCL Timeout Bug During Dataset Building with (Large) Datasets https://github.com/azavea/raster-vision/issues/2276
# MAGIC [10] Error waiting on exit barrier - distributed - PyTorch Forums https://discuss.pytorch.org/t/error-waiting-on-exit-barrier/197541
# MAGIC [11] NCCL Timeout Accelerate Load From Checkpoint https://discuss.huggingface.co/t/nccl-timeout-accelerate-load-from-checkpoint/33953
# MAGIC [12] If the dataset is too big to fit into your RAM, but you still wish to train ... https://www.reddit.com/r/deeplearning/comments/z8otan/if_the_dataset_is_too_big_to_fit_into_your_ram/
# MAGIC [13] Multi-GPU Training with YOLOv5 - Ultralytics YOLO Docs https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/
# MAGIC [14] Multi node training of YOLOv8 (2 machine with 4GPU each) #7038 https://github.com/ultralytics/ultralytics/issues/7038
# MAGIC [15] Error: Some NCCL operations have failed or timed out https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out
# MAGIC [16] Reference for ultralytics/utils/torch_utils.py https://docs.ultralytics.com/reference/utils/torch_utils/
# MAGIC [17] Model Training with Ultralytics YOLO https://docs.ultralytics.com/modes/train/
# MAGIC [18] ultralytics/yolov5/utils/datasets.py - Hugging Face https://huggingface.co/spaces/nakamura196/yolov5-ndl-layout/blob/447b47ec77e6ea46fef0abba2594b11de7874676/ultralytics/yolov5/utils/datasets.py
# MAGIC [19] Machine Learning Best Practices and Tips for Model Training https://docs.ultralytics.com/guides/model-training-tips/
# MAGIC [20] Some NCCL operations have failed or timed out. Due to the ... https://discuss.huggingface.co/t/some-nccl-operations-have-failed-or-timed-out-due-to-the-asynchronous-nature-of-cuda-kernels/26877
# MAGIC [21] Datasets — Torchvision 0.16 documentation https://docs.pytorch.org/vision/0.16/datasets.html
# MAGIC [22] YOLO v11 training multi-GPU DDP Errors - Stack Overflow https://stackoverflow.com/questions/79372969/yolo-v11-training-multi-gpu-ddp-errors
# MAGIC [23] How to fix Nvidia NCCL's "watchdog timeout" error - LinkedIn https://www.linkedin.com/posts/agam-jn_new-blog-alert-understanding-multi-gpu-activity-7321357828789530624-JAa2
# MAGIC [24] torchvision.datasets.country211 - PyTorch https://docs.pytorch.org/vision/2.0/_modules/torchvision/datasets/country211.html
# MAGIC [25] Distributed Training: Definition & How it Works - Ultralytics https://www.ultralytics.com/glossary/distributed-training
# MAGIC [26] Multi node PyTorch Distributed Training Guide For People In A Hurry https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide
# MAGIC [27] Support of very large dataset? - Hugging Face Forums https://discuss.huggingface.co/t/support-of-very-large-dataset/6872
# MAGIC [28] Cache problem while runing on multiple nodes with GPU #30859 https://github.com/huggingface/transformers/issues/30859
# MAGIC [29] Help Needed with DDP Training on Multiple Nodes Unexpected ... https://discuss.pytorch.org/t/help-needed-with-ddp-training-on-multiple-nodes-unexpected-low-accuracy-with-resnet50-and-moco-v3/207469
# MAGIC [30] NCCL timed out when using the torch.distributed.run - PyTorch Forums https://discuss.pytorch.org/t/nccl-timed-out-when-using-the-torch-distributed-run/153276
# MAGIC [31] Using XGBoost External Memory Version https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
# MAGIC [32] Running a PyTorch dataloader/Dataset on multiple distributed CPUs https://stackoverflow.com/questions/59552122/running-a-pytorch-dataloader-dataset-on-multiple-distributed-cpus
# MAGIC [33] An NCCL Error Occurs When a Training Job Fails to Be Executed https://support.huaweicloud.com/intl/en-us/trouble-modelarts/modelarts_trouble_0001.html
# MAGIC [34] Why do I run out of memory when training with a large dataset, but ... https://stackoverflow.com/questions/77029483/why-do-i-run-out-of-memory-when-training-with-a-large-dataset-but-have-no-probl
# MAGIC [35] `torch.distributed.barrier` used in multi-node distributed data-parallel ... https://discuss.pytorch.org/t/torch-distributed-barrier-used-in-multi-node-distributed-data-parallel-training/89711

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional Timeout Related Remedy
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You’re likely hitting the multi‑GPU rendezvous “barrier” during Serverless GPU (SGC) distributed launches, where workers wait for all peers to come up before proceeding. Extending the wait window helps avoid premature timeouts on slower pod startup or transient network delays. Databricks docs explicitly recommend adding retries or increasing timeout for multi‑node launches to avoid barrier‑timeout issues.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What “barrier” refers to in SGC
# MAGIC
# MAGIC * The **barrier** is the synchronization point during distributed training where all ranks must join before the job advances (often called “rendezvous”). If one or more ranks start slowly, you may see messages like “Timed out after N seconds waiting for clients.” Extending the timeout reduces spurious failures on multi‑GPU/multi‑node jobs.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Increase the barrier wait time (recommended options)
# MAGIC
# MAGIC * Set the **SGC_ENV_SYNC_TIMEOUT_SECONDS** environment variable to increase the environment sync/rendezvous timeout (default is 900 seconds = 15 minutes). For example, set to 1800 for 30 minutes before launching the distributed job.
# MAGIC
# MAGIC * Optionally set **SGC_JOB_TIMEOUT_SECONDS** to a larger value (or leave at 0 for no job‑level timeout) to ensure the overall run isn’t terminated before slow ranks finish rendezvous and start training.
# MAGIC
# MAGIC * If you explicitly call **torch.distributed.init_process_group** in your code (common for DDP/FSDP), pass a larger `timeout` (e.g., `timedelta(minutes=20)`) to expand the rendezvous window at the framework level. This aligns the framework’s expectation with SGC’s extended sync time window.
# MAGIC
# MAGIC * When scheduling multi‑node runs, **add retries and/or longer timeouts** per Databricks best practices to avoid barrier timeouts, especially when capacity is constrained or pods start unevenly across nodes.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Example: increase barrier timeout in a notebook
# MAGIC
# MAGIC ```python
# MAGIC # Increase SGC rendezvous/environment sync timeout to 30 minutes
# MAGIC import os
# MAGIC os.environ["SGC_ENV_SYNC_TIMEOUT_SECONDS"] = "1800"  # default is 900 (15 minutes)
# MAGIC os.environ["SGC_JOB_TIMEOUT_SECONDS"] = "0"          # 0 = no job-level timeout
# MAGIC
# MAGIC # If using PyTorch DDP/FSDP directly, also raise framework rendezvous timeout:
# MAGIC from datetime import timedelta
# MAGIC import torch.distributed as dist
# MAGIC
# MAGIC # Example init (adjust backend/world_size/etc. as needed)
# MAGIC dist.init_process_group(
# MAGIC     backend="nccl",
# MAGIC     timeout=timedelta(minutes=20)
# MAGIC )
# MAGIC
# MAGIC # ...rest of your training launch (serverless_gpu @distributed, ray, etc.)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Where to set these in SGC
# MAGIC
# MAGIC * In a **Serverless GPU notebook**, you can set environment variables in the Environment side panel before connecting, or set `os.environ[...]` in the notebook before invoking the **serverless_gpu** `@distributed` function. This is consistent with the SGC setup workflow.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Practical tips
# MAGIC
# MAGIC * Prefer increasing timeouts on **multi‑node A10** runs (H100 currently supports single‑node only), where node‑to‑node skew is most likely. Add retries per docs to avoid transient rendezvous failures.
# MAGIC
# MAGIC * If you see repeated “N−1 of N clients joined” messages, bump both SGC and framework timeouts and retry; this pattern is consistent with rendezvous skew discussed by the platform team.
# MAGIC
# MAGIC * Keep an eye on logs streamed by SGC; log chunking/encoding issues were recently discussed—ensure your environment uses the latest GPU environment version (v4) for fixes and improvements.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Quick checklist
# MAGIC
# MAGIC * Increase SGC rendezvous timeout: set **SGC_ENV_SYNC_TIMEOUT_SECONDS** to 1800–3600 for large jobs.
# MAGIC
# MAGIC * Ensure job isn’t killed early: adjust **SGC_JOB_TIMEOUT_SECONDS** or leave at 0.
# MAGIC
# MAGIC * Align framework rendezvous: pass a larger `timeout` to `torch.distributed.init_process_group` (if you call it directly).
# MAGIC
# MAGIC * Follow docs guidance: add retries/longer timeouts for multi‑node launches to avoid barrier timeouts.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Have you been seeing these timeouts on multi‑node A10 runs or single‑node H100? If you can share the GPU type and whether you’re using PyTorch DDP/FSDP or Ray, I can suggest exact timeout values and where to set them for your setup.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
