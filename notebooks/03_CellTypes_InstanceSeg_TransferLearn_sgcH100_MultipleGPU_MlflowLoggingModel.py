# Databricks notebook source
# MAGIC %md
# MAGIC > - This notebook is an extension example of how to run YOLO Instance Segmentation on a custom dataset with **remote/local single-node multiple 8x H100 SGC compute**.  
# MAGIC > - The example solution will be part of the assets within the forthcoming [databricks-industry-solutions/cv-playground](https://github.com/databricks-industry-solutions/cv-playground) that will show case other CV-related solutions on Databricks.   
# MAGIC > - Developed and last tested [`2025Oct12`] using 1 node with 8 H100 GPUs `sgc_8xH100` and `env_v4` by `yang.yang@databricks.com`
# MAGIC `@distributed(gpus=8, gpu_type='H100', remote=False)`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **What is the different of setting `remote=False` vs `remote=True`**
# MAGIC
# MAGIC set remote=False if your notebook is already attached to an H100 single-node and you want to use those 8 local GPUs; use remote=True if you want to launch onto remote H100 capacity from a lighter notebook environment.
# MAGIC
# MAGIC Guidance
# MAGIC remote=False runs on the GPUs attached to your notebook’s compute. For H100, notebooks can attach a single-node with 8x H100 and you can use those locally. 
# MAGIC 1
# MAGIC 2
# MAGIC
# MAGIC remote=True offloads to remote GPUs. On H100, this should launch a single-node 8x H100 (multi-node on H100 is not supported), and it’s required if your notebook isn’t attached to H100 or you want clean separation between the notebook and training workers. 
# MAGIC 2
# MAGIC 3
# MAGIC
# MAGIC If you set gpus > 8 with H100, it implies multi-node and will fail because multi-node on H100 isn’t supported yet. 
# MAGIC 2
# MAGIC
# MAGIC Heads-up: there’s a known issue where remote=True can sometimes return “H100 multi-node is not permitted.” If you hit it, switch to remote=False and file an incident via go/sgc/incident. 
# MAGIC 4

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

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())

# COMMAND ----------

# DBTITLE 1,remote=True
settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

print('data_yaml_path is:', data_yaml_path)

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)


@distributed(gpus=8, gpu_type='H100', remote=True) # use `remote=False` if you want to train on the local SGC node your notebook is running on; use `True` option, if you want to send the job to another node so that you have isolation from the notebook node.
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

# DBTITLE 1,remote=False
settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

print('data_yaml_path is:', data_yaml_path)

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)


@distributed(gpus=8, gpu_type='H100', remote=False) # use `remote=False` if you want to train on the local SGC node your notebook is running on; use `True` option, if you want to send the job to another node so that you have isolation from the notebook node.
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
