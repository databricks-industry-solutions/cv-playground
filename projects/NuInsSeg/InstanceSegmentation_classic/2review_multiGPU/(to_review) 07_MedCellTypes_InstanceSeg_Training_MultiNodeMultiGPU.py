# Databricks notebook source
## TODO -- paths need to be updated + test on e2 wrt IP blocking on f-e-east ws

# COMMAND ----------

# MAGIC %md
# MAGIC # MedCellTypes Instance Segmentation: 
# MAGIC
# MAGIC ##YOLO Training on MultiNode-MultiGPU Cluster on Databricks
# MAGIC
# MAGIC What is Instance Segmentation?
# MAGIC Instance segmentation goes beyond basic object detection, which draws bounding boxes around objects, and semantic segmentation, which labels each pixel in an image with a class but does not differentiate between individual objects of the same class. Instead, instance segmentation uniquely identifies each object instance, even when they overlap. For example, in an image with multiple cars, instance segmentation will not only recognize all of them as 'car' but will also create a separate, pixel-perfect mask for each individual car, distinguishing them from one another and the background. This capability is crucial in scenarios where counting individual objects or analyzing their specific shapes is important.
# MAGIC
# MAGIC Instance Segmentation vs. Related Tasks
# MAGIC While related, instance segmentation differs significantly from other computer vision tasks:
# MAGIC
# MAGIC - Object Detection: Object detection focuses on identifying and localizing objects within an image by drawing bounding boxes around them. It tells you what and where objects are, but not their exact shape or boundaries.
# MAGIC - Semantic Segmentation: Semantic segmentation classifies each pixel in an image into predefined classes, such as 'sky,' 'road,' or 'car.' It provides a pixel-level understanding of the scene but does not differentiate between separate instances of the same object class. For example, all cars are labeled as 'car' pixels, but are not distinguished as individual objects.
# MAGIC - Instance Segmentation: Instance segmentation combines the strengths of both. It performs pixel-level classification like semantic segmentation, but also differentiates and segments each object instance individually, like object detection, providing a comprehensive and detailed understanding of the objects in an image.

# COMMAND ----------

# MAGIC %md
# MAGIC Below image shows the example of **Instance Segmentation**.
# MAGIC
# MAGIC Specifically, this shows the Mosaiced Image: 
# MAGIC
# MAGIC This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This aids the model's ability to generalize to different object sizes, aspect ratios, and contexts.
# MAGIC ![](https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-3.avif)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Framework Background README

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Pytorch
# MAGIC PyTorch distributed package supports Linux (stable), MacOS (stable), and Windows (prototype). By default for Linux, the Gloo and NCCL backends are built and included in PyTorch distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be included if you build PyTorch from source. (e.g. building PyTorch on a host that has MPI installed.)
# MAGIC In short, 
# MAGIC - use NCCL for GPU,
# MAGIC - use Gloo for CPU,
# MAGIC - use MPI if Gloo wont work for CPU
# MAGIC
# MAGIC ref: https://pytorch.org/docs/stable/distributed.html

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying Parallelism To Scale Your Model
# MAGIC Data Parallelism is a widely adopted single-program multiple-data training paradigm where the model is replicated on every process, every model replica computes local gradients for a different set of input data samples, gradients are averaged within the data-parallel communicator group before each optimizer step.
# MAGIC
# MAGIC Model Parallelism techniques (or Sharded Data Parallelism) are required when a model doesn’t fit in GPU, and can be combined together to form multi-dimensional (N-D) parallelism techniques.
# MAGIC
# MAGIC When deciding what parallelism techniques to choose for your model, use these common guidelines:
# MAGIC
# MAGIC - Use DistributedDataParallel (DDP), if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
# MAGIC
# MAGIC   + Use torchrun, to launch multiple pytorch processes if you are using more than one node.
# MAGIC
# MAGIC   + See also: Getting Started with Distributed Data Parallel
# MAGIC
# MAGIC - Use FullyShardedDataParallel (FSDP) when your model cannot fit on one GPU.
# MAGIC
# MAGIC   + See also: Getting Started with FSDP
# MAGIC
# MAGIC - Use Tensor Parallel (TP) and/or Pipeline Parallel (PP) if you reach scaling limitations with FSDP.
# MAGIC
# MAGIC   + Try our Tensor Parallelism Tutorial
# MAGIC
# MAGIC   + See also: TorchTitan end to end example of 3D parallelism
# MAGIC
# MAGIC   ref: https://pytorch.org/tutorials/beginner/dist_overview.html#applying-parallelism-to-scale-your-model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternatively, you can try the Microsoft variant of DistributedTorch called "DeepSpeed Distributor"
# MAGIC
# MAGIC DeepSpeed is built on top of Torch Distributor. Though DeepSpeed is more for LLM training, you can try apply this for Image DL models like YOLO
# MAGIC
# MAGIC ref: 
# MAGIC 1. https://github.com/microsoft/DeepSpeed
# MAGIC 1. https://docs.databricks.com/en/machine-learning/train-model/distributed-training/deepspeed.html
# MAGIC 2. https://community.databricks.com/t5/technical-blog/introducing-the-deepspeed-distributor-on-databricks/ba-p/59641

# COMMAND ----------

# MAGIC %md
# MAGIC ### YOLO on Databricks Ref: 
# MAGIC 1. YOLO on databricks, https://benhay.es/posts/object_detection_yolov8/
# MAGIC 2. Distributed Torch + MLflow, https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html
# MAGIC
# MAGIC ![](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Environment

# COMMAND ----------

# DBTITLE 1,newest env with .40 ultralytics
# MAGIC %pip install -U ultralytics==8.3.40 opencv-python==4.10.0.84 ray==2.39.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,debug switcher
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Enable synchronous execution for debugging,for debugging purpose, show stack trace immediately. Use it in dev mode.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # Reset to asynchronous execution for production in production

# COMMAND ----------

# MAGIC %md
# MAGIC Create your secret scope using "Databricks Secret Vault", for example,
# MAGIC 1. Create your secret scope first for a specific workspace profile: `databricks secrets create-scope yyang_secret_scope`
# MAGIC 2. Put your secret key and value: `databricks secrets put-secret yyang_secret_scope pat`, here `pat` is your key
# MAGIC     - then input the value following the prompt or editor edit/save
# MAGIC 3. (optional) you can also save other key:value pair like databricks_host and workspace_id. `databricks secrets put-secret yyang_secret_scope db_host`
# MAGIC
# MAGIC
# MAGIC Now you are done.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Ref: https://learn.microsoft.com/en-us/azure/databricks/security/secrets/

# COMMAND ----------

# MAGIC %md
# MAGIC ### At this moment, you could provide Databrics credentials in two ways
# MAGIC 1. Service Principal and OAuth M2M (for production)
# MAGIC 2. Personal Access Token (for development)
# MAGIC
# MAGIC Note:
# MAGIC - Choose either 1. or 2., uncomment the corresponding cell of os.environ variable setup below.
# MAGIC - After setting up the os.environ variables, remember redact the values, e.g., 'xxx'
# MAGIC

# COMMAND ----------

# DBTITLE 1,SP and oauth M2M env vars
import os

# Set Databricks workspace URL
os.environ['DATABRICKS_HOST'] = db_host = 'https://e2-demo-field-eng.cloud.databricks.com/'

# Set Service Principal credentials
os.environ['DATABRICKS_CLIENT_ID'] = 'xxx'
os.environ['DATABRICKS_CLIENT_SECRET'] = 'xxx'


# Print environment variables to verify
print(os.environ['DATABRICKS_HOST'])
print(os.environ['DATABRICKS_CLIENT_ID'])
# print(os.environ['DATABRICKS_TENANT_ID'])

# COMMAND ----------

# DBTITLE 1,PAT env vars
# # Set Databricks workspace URL and token
# os.environ['DATABRICKS_HOST'] = 'https://e2-demo-field-eng.cloud.databricks.com/'
# os.environ['DATABRICKS_TOKEN'] = 'xxx'

# print(os.environ['DATABRICKS_HOST'])
# print(os.environ['DATABRICKS_TOKEN']) # anything from vault would be redacted print.

# COMMAND ----------

# MAGIC %md
# MAGIC **We are using databricks sdk secretAPI for register the secret info.**   
# MAGIC Ref: https://databricks-sdk-py.readthedocs.io/en/latest/workspace/workspace/secrets.html

# COMMAND ----------

# DBTITLE 1,Creating a Secret Scope in Databricks
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import DatabricksError

# Initialize the client
client = WorkspaceClient()

# Define the scope name
scope_name = "cv_yolo_sp_scope"

# Create the secret scope
try:
  client.secrets.create_scope(scope=scope_name)
  print(f"Secret scope '{scope_name}' created successfully!")
except DatabricksError as e:
  print(f"Error creating secret scope: {e}")


# COMMAND ----------

# DBTITLE 1,store secrets in databricks scope
# Define the secrets
secrets = {
    "client_id": os.environ['DATABRICKS_CLIENT_ID'],
    "client_secret": os.environ['DATABRICKS_CLIENT_SECRET']
}

# Write secrets to the scope
for key, value in secrets.items():
    client.secrets.put_secret(scope = scope_name, key=key, string_value=value)

print("Secret added successfully!")

# COMMAND ----------

# DBTITLE 1,list all secrets in specified scope
client.secrets.list_secrets(scope = scope_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Secret Values back from Vault

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import DatabricksError

# Initialize the client
client = WorkspaceClient()

# Define the scope name
scope_name = "cv_yolo_sp_scope"

#
secrets = client.secrets.list_secrets(scope=scope_name)
print(secrets)

# COMMAND ----------

# DBTITLE 1,this api will reveal secret value so you can check
##: uncomment if you want to see the values of your secrets.
# client_id = client.secrets.get_secret(scope=scope_name, key="client_id").value
# client_secret = client.secrets.get_secret(scope=scope_name, key="client_secret").value
# print(client_id)
# print(client_secret)

# COMMAND ----------

# DBTITLE 1,this api will redact the secret value for privacy
client_id = dbutils.secrets.get(scope=scope_name, key="client_id")
client_secret = dbutils.secrets.get(scope=scope_name, key="client_secret")
print(client_id)
print(client_secret)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check package versions
# MAGIC Testing are done under pinned version so we need to be sure

# COMMAND ----------

import ultralytics
print(ultralytics.__version__)

# COMMAND ----------

import ray
ray.__version__

# COMMAND ----------

from ultralytics.utils.checks import check_yolo, check_python, check_latest_pypi_version, check_version, check_requirements

print("check_yolo", check_yolo())
print("check_python", check_python())
print("check_latest_pypi_version", check_latest_pypi_version())
print("check_version", check_version())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Customized PyFunc

# COMMAND ----------

from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

input_schema = Schema(
    [
        ColSpec("string", "image_source"),
    ]
)
output_schema = Schema([ColSpec("string","class_name"),
                        ColSpec("integer","class_num"),
                        ColSpec("double","confidence")]
                       )

signature = ModelSignature(inputs=input_schema, 
                           outputs=output_schema)

# settings.update({"mlflow":False}) # Specifically, it disables the integration with MLflow. By setting the mlflow key to False, you are instructing the ultralytics library not to use MLflow for logging or tracking experiments.

# ultralytics level setting with MLflow
settings.update({"mlflow":True}) # if you do want to autolog.
# # Config MLflow
mlflow.autolog(disable=True)
mlflow.end_run()

# COMMAND ----------

# os.environ["OMP_NUM_THREADS"] = "12"  # OpenMP threads

# COMMAND ----------

# DBTITLE 1,new version added .predict
############################################################################
## Create YOLOC class to capture model results d a predict() method ##
############################################################################

class YOLOC(mlflow.pyfunc.PythonModel):
  def __init__(self, point_file):
    self.point_file=point_file

  def load_context(self, context):
    from ultralytics import YOLO
    self.model = YOLO(context.artifacts['best_point'])

  def predict(self, context, model_input):
    # ref: https://docs.ultralytics.com/modes/predict/
    return self.model(model_input)
  

#: use model() vs model.predict()
# The model.predict() method in YOLO supports various arguments such as conf, iou, imgsz, device, and more. These arguments allow you to customize the inference process, setting parameters like confidence thresholds, image size, and the device used for computation. Detailed descriptions of these arguments can be found in the inference arguments section.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup I/O Path

# COMMAND ----------

# MAGIC %sql
# MAGIC -- #: in normal situation, please uncomment below line, here we have 2000 limit and already reached.
# MAGIC -- /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset
# MAGIC create catalog if not exists mmt;
# MAGIC create schema if not exists mmt.cv;
# MAGIC create Volume if not exists mmt.cv.projects;

# COMMAND ----------

import os
# Config project structure directory under UC
project_location = '/Volumes/mmt/cv/projects/'
os.makedirs(f'{project_location}/training_runs/', exist_ok=True)
os.makedirs(f'{project_location}/data/', exist_ok=True)
os.makedirs(f'{project_location}/raw_model/', exist_ok=True)

# for cache related to ultralytics
os.environ['ULTRALYTICS_CACHE_DIR'] = f'{project_location}/raw_model/'


# volume folder in UC.
volume_project_location = f'{project_location}/training_results/'
os.makedirs(volume_project_location, exist_ok=True)

# or more traditional way, setup folder under DBFS.
dbfs_project_location = '/dbfs/tmp/cv_project_location/Nuclei_Instance/'
os.makedirs(dbfs_project_location, exist_ok=True)

# ephemeral /tmp/ project location on VM, good for Appending operation during training.
tmp_project_location = "/tmp/training_results/"
os.makedirs(tmp_project_location, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,check the content of this .yaml file under the Volumes
# data.yaml is the YOLO data template, needed by the training step later.
%cat /Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset/data.yaml

# COMMAND ----------

# DBTITLE 1,define your yaml_path for later training section
yaml_path = "/Volumes/mmt/cv/projects/NuInsSeg/yolo_dataset/data.yaml"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Working Directory
# MAGIC root folder will be the folder containing this notebook

# COMMAND ----------

os.getcwd()

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0]

# COMMAND ----------

# DBTITLE 1,reset directory to current notebook path
os.chdir('/Workspace/' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0])
os.getcwd()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow Logged Distributed Training

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup TensorBoard for monitoring

# COMMAND ----------

# DBTITLE 1,(optional, nice to have) setup tensorboard before running the below distributed training
# MAGIC %load_ext tensorboard
# MAGIC # This sets up our tensorboard settings
# MAGIC # /tmp/training_results/train
# MAGIC # tensorboard --logdir /tmp/training_results/train
# MAGIC experiment_log_dir = f'{tmp_project_location}/train'
# MAGIC %tensorboard --logdir $experiment_log_dir
# MAGIC # This starts Tensorboard

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Setup MLflow

# COMMAND ----------

# DBTITLE 1,Setting Up Experiment Name Based on User Path
workspace_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
workspace_path = f"/Users/{workspace_path}/"
experiment_name = workspace_path + "MedCellTypes_Instance_Segmentation_Experiment_Managed"
print(f"Setting experiment name to be {experiment_name}")


# COMMAND ----------

os.getcwd()

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow experiment artifacts could be **saved under dbfs (managed) or UC volume**.
# MAGIC
# MAGIC Here we saved them into the same UC volume path we defined earlier, so easier to manage the whole project.

# COMMAND ----------

# DBTITLE 1,mlflow setup for experiment and UC artifact paths
#: mlflow settup
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.end_run()
#
# experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0] + "/MedCellTypes_Instance_Segmentation_Experiment_Managed"
# print(f"Setting experiment name to be {experiment_name}")
workspace_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
workspace_path = f"/Users/{workspace_path}/"
experiment_name = workspace_path + "MedCellTypes_Instance_Segmentation_Experiment_Managed"
print(f"Setting experiment name to be {experiment_name}")

#: Use UC Volume path to logging MLflow experiment instead of MLflow-managed artifact storage: dbfs:/databricks/mlflow-tracking/<experiment-id>.
# project_location = '/Volumes/yyang/computer_vision/Nuclei_Instance/'
ARTIFACT_PATH = f"dbfs:{project_location}artifact"
print(f"Creating experiment ARTIFACT_PATH to be {ARTIFACT_PATH}")
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name, artifact_location=ARTIFACT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MultiNode-MultiGPU version
# MAGIC
# MAGIC Multiple Nodes, GPU can be many per node (> 1).

# COMMAND ----------

# DBTITLE 1,Helper func to get total # of GPUs across multiple nodes
#: this works for both single-node and multi-node cluster scenarios (each node may have multiple GPUs)
def get_total_gpus():
    # Get the number of nodes in the cluster
    num_nodes = int(spark.conf.get("spark.databricks.clusterUsageTags.clusterTargetWorkers"))
    num_nodes = max(num_nodes, 1) # to avoid 0 issue if single-node cluster without any workers
    
    # Get the number of GPUs per node
    num_gpus_per_node = int(spark.conf.get("spark.executor.resource.gpu.amount"))

    # Calculate the total number of GPUs
    total_gpus = num_nodes * num_gpus_per_node

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of GPUs per node: {num_gpus_per_node}")
    print(f"Total number of GPUs across all nodes: {total_gpus}")
    
    return total_gpus

# Call the function to get the total number of GPUs
total_gpus = get_total_gpus()

# COMMAND ----------

# MAGIC %md
# MAGIC __Doc of YOLO `model.train` API__
# MAGIC https://docs.ultralytics.com/modes/train/#train-settings
# MAGIC
# MAGIC E.g., 
# MAGIC ```
# MAGIC model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
# MAGIC         model.train(
# MAGIC             batch=8,
# MAGIC             device=device_list,
# MAGIC             data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
# MAGIC             epochs=5,
# MAGIC             project=f'{tmp_project_location}',
# MAGIC             exist_ok=True,
# MAGIC             fliplr=1,
# MAGIC             flipud=1,
# MAGIC             perspective=0.001,
# MAGIC             degrees=.45
# MAGIC         )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC **We have two versions here, only differing in how you log the model flavour. Otherwise, the other parts are exactly the same.**
# MAGIC
# MAGIC 1. using before defined custom pyfunc PythonModel class. In this way, you have full control of your model behaviour
# MAGIC 2. default class of PyTorch model.

# COMMAND ----------

# DBTITLE 1,Version A: custom pyfunc pythonModel class
# major ultralytics distributed training debugging refs: https://github.com/ultralytics/ultralytics/issues/7038 and other related threads for issues with multi-node training on ultralytics repo.
# MLFLOW nested-runs refs: https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs.html

# update: revised from a single run to nested runs with driver creating a parent run and each GPU creating a child run under it.
    # 1. driver run is responsible for recording manually logged artifacts as well as full-length system metrics, e.g., from start to end of the whole process including overheads in the starting and ending phase.
    # 2. each GPU will record its own system metrics at the dedicated training phase, usually shorter than the parent run span.
    # 3. GPU 0 will record parameters and model metrics into its own child run "GPU_RANK_0)".
    # 4. GPU 0 will also log the model via customized pyfunc after training is finished into the driver parent run (NOT GPU_RANK_0 child run).

yaml_path = yaml_path
#: -----------worker func: this function is visible to each GPU device.-------------------
def train_fn(world_size = None, parent_run_id = None):

    import os
    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature
    from ultralytics.utils import RANK, LOCAL_RANK


    ############################
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for synchronization operation, debugging model prefers this.
    # os.environ["NCCL_DEBUG"] = "INFO" # "WARN" # for more debugging info on the NCCL side.
    if "NCCL_DEBUG" in os.environ:
        os.environ.pop('NCCL_DEBUG') # reset
    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    print(f"DATABRICKS_HOST set to {os.environ['DATABRICKS_HOST']}")
    # os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    # print(f"DATABRICKS_TOKEN set to {os.environ['DATABRICKS_TOKEN']}") # should be redacted
    os.environ['DATABRICKS_CLIENT_ID'] = client_id
    print(f"DATABRICKS_CLIENT_ID set to {os.environ['DATABRICKS_CLIENT_ID']}") # should be redacted
    os.environ['DATABRICKS_CLIENT_SECRET'] = client_secret
    print(f"DATABRICKS_CLIENT_SECRET set to {os.environ['DATABRICKS_CLIENT_SECRET']}") # should be redacted


    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    # #: from repo issue https://github.com/ultralytics/ultralytics/issues/11680
    ## conclusion: doesn't work, has error :"ValueError: Invalid CUDA 'device=0,1' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU."
    # torch.backends.cudnn.benchmark = False
    # torch.cuda.synchronize()
    print(f"------Before init_process_group, we have: {RANK=} -- {LOCAL_RANK=}------")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus, # total gpus you have across all nodes.
        rank=RANK, # this must be from 0 to world_size - 1. LOCAL_RANK wont work.
    )
    print(f"------After init_process_group, we have: {RANK=} -- {LOCAL_RANK=}------")


    #
    with mlflow.start_run(parent_run_id=parent_run_id, run_name = f"GPU_RANK_{RANK}", description="child runs", nested=True) as child_run:
        model = YOLO(f"{project_location}/raw_model/yolo11n-seg.pt") # shared location
        # model = YOLO("yolo11n")
        model.train(
            batch=16, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
            device=[LOCAL_RANK], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
            data=yaml_path,
            epochs=20,
            project=f'{tmp_project_location}', # local VM ephermal location
            # project=f'{volume_project_location}', # volume path still wont work
            exist_ok=True,
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
        

    # active_run_id = mlflow.last_active_run().info.run_id
    # print("For YOLO autologging, active_run_id is: ", active_run_id)

    # after training is done.
    if not dist.is_initialized():
      # import torch.distributed as dist
      dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    print(f"------After training, we have: RANK:{global_rank=} -- LOCAL_RANK:{local_rank=}------")

    if global_rank == 0:

        # active_run_id = mlflow.last_active_run().info.run_id
        # print("For YOLO autologging, active_run_id is: ", active_run_id)

        # # Get the list of runs in the experiment
        # runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)

        # # Extract the latest run_id
        # if not runs.empty:
        #     latest_run_id = runs.iloc[0].run_id
        #     print(f"Latest run_id: {latest_run_id}")
        # else:
        #     print("No runs found in the experiment.")


        with mlflow.start_run(run_id=parent_run_id) as run:
            mlflow.log_artifact(yaml_path, "input_data_yaml")
            # mlflow.log_dict(data, "data.yaml")
            mlflow.log_params({"rank":global_rank})
            yolo_wrapper = YOLOC(model.trainer.best)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                artifacts={'model_path': str(model.trainer.save_dir), "best_point": str(model.trainer.best)},
                python_model=yolo_wrapper,
                signature=signature)

    # clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    return "finished" # can return any picklable object


#: -------execute on the driver node to trigger multi-node training.------------
if __name__ == "__main__":
    from pyspark.ml.torch.distributor import TorchDistributor

    settings.update({"mlflow":True}) # if you do want to autolog.
    mlflow.autolog(disable = False)

    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for synchronization operation, debugging model prefers this.
    # os.environ["NCCL_DEBUG"] = "INFO" # for debugging
    if "NCCL_DEBUG" in os.environ:
        os.environ.pop('NCCL_DEBUG') # reset

    mlflow.end_run()
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    # num_gpus = int(os.environ["WORLD_SIZE"]) # this only works if driver is GPU node
    # num_gpus = torch.cuda.device_count() # this only works if driver is GPU node and This function only returns the number of GPUs available on the current node to which the process is assigned. Therefore, if you run this function on any single node within a multi-node cluster, it will only return the number of GPUs available on that particular node, not the total count across all nodes in the cluster.
    num_gpus = get_total_gpus() # from above helper function
    print("num_gpus:", num_gpus)

    # device_list = list(range(int(num_gpus/2)))
    # print("device_list:", device_list)

    with mlflow.start_run(experiment_id=experiment_id) as parent_run:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        # print("For master triggering run, active_run_id is: ", active_run_id)
        # print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All nested worker runs will be logged under the same parent run id '{active_run_id}' and name '{active_run_name}'.")

        distributor = TorchDistributor(num_processes=num_gpus, local_mode=False, use_gpu=True)      
        distributor.run(train_fn, world_size = num_gpus, parent_run_id = active_run_id)

# COMMAND ----------

# DBTITLE 1,Version B: PyTorch class
# major ultralytics distributed training debugging refs: https://github.com/ultralytics/ultralytics/issues/7038 and other related threads for issues with multi-node training on ultralytics repo.
# MLFLOW nested-runs refs: https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs.html

# update: revised from a single run to nested runs with driver creating a parent run and each GPU creating a child run under it.
    # 1. driver run is responsible for recording manually logged artifacts as well as full-length system metrics, e.g., from start to end of the whole process including overheads in the starting and ending phase.
    # 2. each GPU will record its own system metrics at the dedicated training phase, usually shorter than the parent run span.
    # 3. GPU 0 will record parameters and model metrics into its own child run "GPU_RANK_0)".
    # 4. GPU 0 will also log the model via customized pyfunc after training is finished into the driver parent run (NOT GPU_RANK_0 child run).

yaml_path = yaml_path
#: -----------worker func: this function is visible to each GPU device.-------------------
def train_fn(world_size = None, parent_run_id = None):

    import os
    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature
    from ultralytics.utils import RANK, LOCAL_RANK


    ############################
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for synchronization operation, debugging model prefers this.
    # os.environ["NCCL_DEBUG"] = "INFO" # "WARN" # for more debugging info on the NCCL side.
    if "NCCL_DEBUG" in os.environ:
        os.environ.pop('NCCL_DEBUG') # reset
    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    print(f"DATABRICKS_HOST set to {os.environ['DATABRICKS_HOST']}")
    # os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    # print(f"DATABRICKS_TOKEN set to {os.environ['DATABRICKS_TOKEN']}") # should be redacted
    os.environ['DATABRICKS_CLIENT_ID'] = client_id
    print(f"DATABRICKS_CLIENT_ID set to {os.environ['DATABRICKS_CLIENT_ID']}") # should be redacted
    os.environ['DATABRICKS_CLIENT_SECRET'] = client_secret
    print(f"DATABRICKS_CLIENT_SECRET set to {os.environ['DATABRICKS_CLIENT_SECRET']}") # should be redacted


    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    # #: from repo issue https://github.com/ultralytics/ultralytics/issues/11680
    ## conclusion: doesn't work, has error :"ValueError: Invalid CUDA 'device=0,1' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU."
    # torch.backends.cudnn.benchmark = False
    # torch.cuda.synchronize()
    print(f"------Before init_process_group, we have: {RANK=} -- {LOCAL_RANK=}------")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus, # total gpus you have across all nodes.
        rank=RANK, # this must be from 0 to world_size - 1. LOCAL_RANK wont work.
    )
    print(f"------After init_process_group, we have: {RANK=} -- {LOCAL_RANK=}------")


    #
    with mlflow.start_run(parent_run_id=parent_run_id, run_name = f"GPU_RANK_{RANK}", description="child runs", nested=True) as child_run:
        model = YOLO(f"{project_location}/raw_model/yolo11n-seg.pt") # shared location
        # model = YOLO("yolo11n")
        model.train(
            batch=16, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
            device=[LOCAL_RANK], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
            data=yaml_path,
            epochs=20,
            project=f'{tmp_project_location}', # local VM ephermal location
            # project=f'{volume_project_location}', # volume path still wont work
            exist_ok=True,
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
        

    # active_run_id = mlflow.last_active_run().info.run_id
    # print("For YOLO autologging, active_run_id is: ", active_run_id)

    # after training is done.
    if not dist.is_initialized():
      # import torch.distributed as dist
      dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    print(f"------After training, we have: RANK:{global_rank=} -- LOCAL_RANK:{local_rank=}------")

    if global_rank == 0:

        # active_run_id = mlflow.last_active_run().info.run_id
        # print("For YOLO autologging, active_run_id is: ", active_run_id)

        # # Get the list of runs in the experiment
        # runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)

        # # Extract the latest run_id
        # if not runs.empty:
        #     latest_run_id = runs.iloc[0].run_id
        #     print(f"Latest run_id: {latest_run_id}")
        # else:
        #     print("No runs found in the experiment.")


        with mlflow.start_run(run_id=parent_run_id) as run:
            mlflow.log_artifact(yaml_path, "input_data_yaml")
            mlflow.log_params({"rank":global_rank})
            mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature) # this succeeded


    # clean up
    if dist.is_initialized():
        dist.destroy_process_group()

    return "finished" # can return any picklable object


#: -------execute on the driver node to trigger multi-node training.------------
if __name__ == "__main__":
    from pyspark.ml.torch.distributor import TorchDistributor

    settings.update({"mlflow":True}) # if you do want to autolog.
    mlflow.autolog(disable = False)

    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for synchronization operation, debugging model prefers this.
    # os.environ["NCCL_DEBUG"] = "INFO" # for debugging
    if "NCCL_DEBUG" in os.environ:
        os.environ.pop('NCCL_DEBUG') # reset

    mlflow.end_run()
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    # num_gpus = int(os.environ["WORLD_SIZE"]) # this only works if driver is GPU node
    # num_gpus = torch.cuda.device_count() # this only works if driver is GPU node and This function only returns the number of GPUs available on the current node to which the process is assigned. Therefore, if you run this function on any single node within a multi-node cluster, it will only return the number of GPUs available on that particular node, not the total count across all nodes in the cluster.
    num_gpus = get_total_gpus() # from above helper function
    print("num_gpus:", num_gpus)

    # device_list = list(range(int(num_gpus/2)))
    # print("device_list:", device_list)

    with mlflow.start_run(experiment_id=experiment_id) as parent_run:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        # print("For master triggering run, active_run_id is: ", active_run_id)
        # print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All nested worker runs will be logged under the same parent run id '{active_run_id}' and name '{active_run_name}'.")

        distributor = TorchDistributor(num_processes=num_gpus, local_mode=False, use_gpu=True)      
        distributor.run(train_fn, world_size = num_gpus, parent_run_id = active_run_id)

# COMMAND ----------

experiment_name

# COMMAND ----------

# DBTITLE 1,end MLflow run
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------
