# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Computer Vision: YOLO Trained on SingleNode-MultiGPU with COCO128 Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Background

# COMMAND ----------

# MAGIC %md
# MAGIC ### SingleNode MultiGPU Setup on Databricks
# MAGIC 
# MAGIC Create a GPU Cluster (can be all-purpose or job depending on your task type). Your cluster configuration should include a single node multiple GPU note type, e.g., Standard_NC64as_T4_v3 on Azure Databricks workspace or g4dn.12xlarge on AWS workspace, which composes four T4 GPUs. You are also free to use other GPU type such like A10, V/A/H100, etc. Just make sure you have at least two GPUs on the node cause this notebook is to showcase how to parallel GPUs. Now you are ready to create your cluster:  
# MAGIC - Single node cluster with multiple GPUs, single-user.
# MAGIC - Databricks runtime version should be at least 13.0 ML+ (GPU, Scala 2.12, Spark 3.4.0). The code was tested on that version.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Pytorch
# MAGIC PyTorch distributed package supports Linux (stable), MacOS (stable), and Windows (prototype). By default for Linux, the Gloo and NCCL backends are built and included in PyTorch distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be included if you build PyTorch from source. (e.g. building PyTorch on a host that has MPI installed.)
# MAGIC In short, 
# MAGIC - use NCCL for GPU (this notebook),
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
# MAGIC - Use DistributedDataParallel (DDP) (**this notebook**), if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
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
# MAGIC ### We are ready to start! 
# MAGIC ![](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Environment

# COMMAND ----------

# DBTITLE 1,newest env
%pip install -U ultralytics==8.4.14 opencv-python==4.13.0.92 ray==2.53.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,debug switcher
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Enable synchronous execution for debugging,for debugging purpose, show stack trace immediately. Use it in dev mode.
os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # Reset to asynchronous execution for production 

# COMMAND ----------

# DBTITLE 1,Super Important for MLflow
# Infer Databricks workspace URL and token from notebook context (no hardcoded values needed)
import json
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ['DATABRICKS_HOST'] = db_host = _ctx.apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = db_token = _ctx.apiToken().get()
os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id = json.loads(_ctx.toJson())["tags"]["orgId"]



print(os.environ['DATABRICKS_HOST'])
print(os.environ['DATABRICKS_WORKSPACE_ID'])
print(os.environ['DATABRICKS_TOKEN']) # anything from vault would be redacted print.

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

############################################################################
## Create YOLOC class to capture model results d a predict() method ##
############################################################################

class YOLOC(mlflow.pyfunc.PythonModel):
  def __init__(self, point_file):
    self.point_file=point_file

  def load_context(self, context):
    from ultralytics import YOLO
    self.model = YOLO(context.artifacts['best_point'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup I/O Path

# COMMAND ----------

# MAGIC %sql
# MAGIC create catalog if not exists yyang;
# MAGIC create schema if not exists yyang.computer_vision;
# MAGIC create Volume if not exists yyang.computer_vision.yolo;

# COMMAND ----------

import os
# Config project structure directory
project_location = '/Volumes/yyang/computer_vision/yolo/'
os.makedirs(f'{project_location}/training_runs/', exist_ok=True)
# os.chdir(f'{project_location}/training_runs/')
os.makedirs(f'{project_location}/data/', exist_ok=True)
os.makedirs(f'{project_location}/raw_model/', exist_ok=True)

# for cache related to ultralytics
os.environ['ULTRALYTICS_CACHE_DIR'] = f'{project_location}/raw_model/'


# volume folder in UC.
volume_project_location = f'{project_location}/training_results/'
os.makedirs(volume_project_location, exist_ok=True)

# or more traditional way, setup folder under DBFS.
# dbfs_project_location = '/dbfs/FileStore/cv_project_location/yolo/'
dbfs_project_location = '/dbfs/tmp/cv_project_location/yolo/'
os.makedirs(dbfs_project_location, exist_ok=True)

# ephemeral /tmp/ project location on VM
tmp_project_location = "/tmp/training_results/"
os.makedirs(tmp_project_location, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset using .yaml template

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0]

# COMMAND ----------

os.chdir('/Workspace/' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 1)[0])
os.getcwd()

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L https://github.com/ultralytics/ultralytics/raw/main/ultralytics/cfg/datasets/coco8.yaml -o coco8.yaml

# COMMAND ----------

import os

with open('coco8.yaml', 'r') as file:
    data = file.read()

data = data.replace('../datasets/', f'{project_location}/data/')

with open('coco8.yaml', 'w') as file:
    file.write(data)

# COMMAND ----------

# MAGIC %sh
# MAGIC cat ./coco8.yaml

# COMMAND ----------

import yaml

with open('coco8.yaml', 'r') as file:
    data = yaml.safe_load(file)

data

# COMMAND ----------

# DBTITLE 1,Extract Zip File from URL and Save to Path
import requests, zipfile, io

response = requests.get(data['download'])
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall(data['path'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## A few minimal testing examples of Distributed Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tips from Working Mini Example below
# MAGIC Scenaio: If you have 2 minimal and current available workers, each owning 4 GPUs, while the driver node could be CPU or GPU node. The max number could be anything.
# MAGIC 
# MAGIC smaller or equaly than 8 (2*4) are supported as we have cluster config as min worker nodes = 2, max = 4. However, as of now, autoscaling is not supported for TorchDistributor on Databricks, so we couldn't go from 8 to 16 GPUs in this example. If you specify something > 8, it will hang there forever without incurring an error msg. 
# MAGIC 
# MAGIC Make sure you specify `num_processes` equal to your total # of GPUs over all the available/online nodes, e.g., 2 node * 4 GPUs/node = 8 GPUs = 8 processes.
# MAGIC 
# MAGIC Last thing, dont count any GPU from driver, they will not be used for computing.

# COMMAND ----------

# DBTITLE 1,Working mini training Example
from pyspark.ml.torch.distributor import TorchDistributor

def train():
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    # Initialize the process group for distributed training
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    
    # Set the device based on the local rank
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # Create the model and move it to the appropriate device
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[device_id])

    # Create dataset and dataloader with distributed sampler
    dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Clean up the process group
    dist.destroy_process_group()

# Instantiate TorchDistributor and run the training function
distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True) # smaller or equaly than 8 (2*4) are supported as we have min worker nodes = 2, max = 4. However,currently autoscaling is not supported for TorchDistributor on Databricks, so we couldn't go from 8 to 16 GPUs. 
distributor.run(train)

# COMMAND ----------

# DBTITLE 1,A simple multi-gpu I/O test
# (DONT RUN) this will throw error under a multi-node multi-GPU cluster setting
import os
import torch
import torch.distributed as dist
from pyspark.ml.torch.distributor import TorchDistributor


def main():

    # #Set Databricks workspace URL and token
    # os.environ['DATABRICKS_HOST'] = db_host = 'https://adb-984752964297111.11.azuredatabricks.net'
    # os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id = '984752964297111'
    # os.environ['DATABRICKS_TOKEN'] = db_token

    # Initialize distributed training
    dist.init_process_group(backend='nccl')  # Or 'gloo' if not using GPUs
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Global rank is: {rank}")
    print(f"Local rank is: {local_rank}")

    # Set the device based on the local rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # rank = dist.get_global_rank()

    # print(os.environ['DATABRICKS_HOST'])
    # print(os.environ['DATABRICKS_WORKSPACE_ID'])
    # print(os.environ['DATABRICKS_TOKEN']) # anything from vault would be redacted print.


    # output_file_path = f'/dbfs/tmp/output_{rank}.txt' # wont work, concurrency writing issue
    # output_file_path = f'{project_location}/output_{rank}.txt' # wont work, concurrency writing issue
    # output_file_path = '/dbfs/tmp/output.txt' # wont work, concurrency writing issue
    # output_file_path = f'{project_location}/output.txt' # wont work, concurrency writing issue
    output_file_path = 'output.txt'  # wont work, read-only issue as "[rank0]: OSError: [Errno 30] Read-only file system: 'output.txt'"
    output_file_path = '/tmp/output.txt' # this works, but on driver, you cannot see worker nodes' /tmp/ folder.

    if rank == 0 and os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Only rank 0 should check and create the file
    if rank == 0:
        if not os.path.exists(output_file_path):
            with open(output_file_path, 'w') as f:
                f.write('Hello from rank 0!\n')
            
            print(f"Write from {rank} finished!")

    # Barrier to ensure all processes wait until rank 0 creates the file
    dist.barrier()

    # Other ranks can now access the file
    with open(output_file_path, 'a') as f:
        f.write(f'Hello from rank {rank}!\n')
    print(f"Write from {rank} finished!")

    dist.barrier()

    if rank == 3:
        os.system(f'cp /tmp/output.txt {project_location}/output.txt')
    
    # Clean up the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    # main()
    distributor = TorchDistributor(num_processes=2*2, local_mode=False, use_gpu=True)
    distributor.run(main)

# if output_file_path = 'output.txt':
# error msg: [rank3]: torch.distributed.DistBackendError: [3] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Connection reset by peer

# if output_file_path = '/dbfs/tmp/output.txt'
# [rank0]: OSError: [Errno 95] Operation not supported

# COMMAND ----------

# MAGIC %cat {project_location}/output.txt

# COMMAND ----------

ls -ahlrt /tmp/

# COMMAND ----------

ls -ahlrt /dbfs/tmp/

# COMMAND ----------

cat /dbfs/tmp/output.txt

# COMMAND ----------

ls -ahlrt {project_location}

# COMMAND ----------

cat {project_location}/output.txt

# COMMAND ----------

# DBTITLE 1,I/O to each worker node's /tmp/
import os
import torch
import torch.distributed as dist
from pyspark.ml.torch.distributor import TorchDistributor

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')  # Or 'gloo' if not using GPUs
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Use a temporary directory on the local file system
    output_file_path = f'/tmp/output_{rank}.txt'

    # Only rank 0 should check and create the file
    if rank == 0:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        with open(output_file_path, 'w') as f:
            f.write('Hello from rank 0!\n')

    # Barrier to ensure all processes wait until rank 0 creates the file
    dist.barrier()

    # Other ranks can now access the file
    with open(output_file_path, 'a') as f:
        f.write(f'Hello from rank {rank}!\n')

    # Clean up the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
    distributor.run(main)

# COMMAND ----------

# DBTITLE 1,nothing on driver node if IO done on worker node
# MAGIC %sh
# MAGIC cat /tmp/output_0.txt

# COMMAND ----------

torch.cuda.is_available()

# COMMAND ----------

os.getenv('CUDA_VISIBLE_DEVICES')

# COMMAND ----------

# DBTITLE 1,test 1: doesn't work whatever
#: wont work either drive node is either CPU or GPU under the Multi-Node-Multi-GPU cluster setting.
# import os

# dist.init_process_group(backend='nccl')
# # Get the CUDA_VISIBLE_DEVICES environment variable
# cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

# # If CUDA_VISIBLE_DEVICES is set, it will list the GPU indices assigned to the task
# if cuda_visible_devices is not None:
#     num_gpus = len(cuda_visible_devices.split(','))
#     print(f"Number of GPUs available: {num_gpus}")
# else:
#     print("No GPUs available or CUDA_VISIBLE_DEVICES is not set.")
  
# dist.destroy_process_group()  

# COMMAND ----------

# #: this only works on the driver node.
# import subprocess

# def get_cuda_device_count():
#     try:
#         result = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if result.returncode == 0:
#             gpu_count = len(result.stdout.strip().split('\n'))
#             return gpu_count
#         else:
#             return "Error: " + result.stderr.strip()
#     except FileNotFoundError:
#         return "nvidia-smi command not found. Ensure NVIDIA drivers are installed."

# gpu_count = get_cuda_device_count()
# gpu_count

# COMMAND ----------

# DBTITLE 1,driver only query
# import torch

# if torch.cuda.is_available():
#     device_count = torch.cuda.device_count()
#     print("Number of CUDA devices:", device_count)
# else:
#     print("No CUDA devices found.")

# COMMAND ----------

# DBTITLE 1,test 2: chicken egg dilemma
# #: conclusion, wont work, chicken egg egg chicken, you still need to provide the correct # of num_processes to TorchDistributor in order to run.
# import torch.distributed as dist
# def distribute_n_get():
  
#   #: remote function for workers
#   def get_num_gpus():
#     import torch.distributed as dist
#     dist.init_process_group(backend='nccl')
#     if dist.get_rank() == 0:
#       num_processes = dist.get_world_size()

#     dist.destroy_process_group()
#     return num_processes
  
#   distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)      
#   return distributor.run(get_num_gpus)

# # Example call:
# num_gpus = distribute_n_get()

# print(f"Number of GPUs in the cluster: {num_gpus}")

# COMMAND ----------

# DBTITLE 1,test 3: get local gpu count by spark config
from pyspark.sql.functions import lower

# Get all Spark configuration settings
all_conf = spark.conf.getAll

# Convert the configuration settings to a DataFrame
conf_df = spark.createDataFrame(all_conf.items(), ["Key", "Value"])

conf_df_filter = conf_df.filter(lower(conf_df.Key).contains("gpu"))

# Display the DataFrame
display(conf_df_filter)

# COMMAND ----------

# MAGIC %cat /databricks/spark/scripts/gpu/get_gpus_resources.sh

# COMMAND ----------

# DBTITLE 1,driver only
# MAGIC %sh
# MAGIC nvidia-smi  -L

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow Logged Distributed Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup TensorBoard for monitoring

# COMMAND ----------

# DBTITLE 1,setup tensorboard before running the below distributed training
# MAGIC %load_ext tensorboard
# MAGIC # This sets up our tensorboard settings
# MAGIC # /tmp/training_results/train
# MAGIC # tensorboard --logdir /tmp/training_results/train
# MAGIC experiment_log_dir = f'{tmp_project_location}/train'
# MAGIC %tensorboard --logdir $experiment_log_dir
# MAGIC # This starts Tensorboard

# COMMAND ----------

# MAGIC %md
# MAGIC ### SingleNode-MultiGPU version

# COMMAND ----------

# DBTITLE 1,SingleNode-MultiGPU version
from pyspark.ml.torch.distributor import TorchDistributor

settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)


def train_fn(device_list = [0,1,2,3], active_run_id = None):

    from ultralytics import YOLO
    import torch
    import mlflow
    import torch.distributed as dist
    from ultralytics import settings
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature

    ############################

    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret 
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_name)
    
    #
    with mlflow.start_run(run_id=active_run_id) as run:
        model = YOLO(f"{project_location}/raw_model/yolov8n.pt")
        model.train(
            batch=8,
            device=device_list,
            data="./coco8.yaml", # ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
            epochs=5,
            project=f'{tmp_project_location}',
            exist_ok=True,
            fliplr=1,
            flipud=1,
            perspective=0.001,
            degrees=.45
        )

    # active_run_id = mlflow.last_active_run().info.run_id
    # print("For YOLO autologging, active_run_id is: ", active_run_id)

    # # after training is done.
    # if not dist.is_initialized():
    #   # import torch.distributed as dist
    #   dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    if global_rank == 0:

        active_run_id = mlflow.last_active_run().info.run_id
        print("For YOLO autologging, active_run_id is: ", active_run_id)

        # Get the list of runs in the experiment
        runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)

        # Extract the latest run_id
        if not runs.empty:
            latest_run_id = runs.iloc[0].run_id
            print(f"Latest run_id: {latest_run_id}")
        else:
            print("No runs found in the experiment.")


        with mlflow.start_run(run_id=latest_run_id) as run:
            mlflow.log_artifact("./coco8.yaml", "input_data_yaml")
            mlflow.log_dict(data, "data.yaml")
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


if __name__ == "__main__":
    mlflow.end_run()
    experiment_name = "/Users/yang.yang@databricks.com/ConvaTec Workshop Computer Vision/Experiments_YOLO_CoCo"
    os.environ['DATABRICKS_HOST'] = db_host # pending replace with db vault secret
    print(f"DATABRICKS_HOST set to {os.environ['DATABRICKS_HOST']}")
    os.environ['DATABRICKS_TOKEN'] = db_token # pending replace with db vault secret
    print(f"DATABRICKS_TOKEN set to {os.environ['DATABRICKS_TOKEN']}") # should be redacted
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")
    os.environ['DATABRICKS_WORKSPACE_ID'] = db_wksp_id  # Set the workspace ID
    print(f"DATABRICKS_WORKSPACE_ID set to {os.environ['DATABRICKS_WORKSPACE_ID']}")

    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Reset MLFLOW_RUN_ID, so we dont bump into the wrong one.
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

    num_gpus = torch.cuda.device_count()
    device_list = list(range(num_gpus))
    print("num_gpus:", num_gpus)
    print("device_list:", device_list)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        active_run_id = mlflow.last_active_run().info.run_id
        active_run_name = mlflow.last_active_run().info.run_name

        print("For master triggering run, active_run_id is: ", active_run_id)
        print("For master triggering run, active_run_name is: ", active_run_name)
        print(f"For master triggering run, active_run_id is: '{active_run_id}' and active_run_name is: '{active_run_name}'.")
        print(f"All worker runs will be logged into the same run id '{active_run_id}' and name '{active_run_name}'.")

        distributor = TorchDistributor(num_processes=num_gpus, local_mode=True, use_gpu=True)      
        distributor.run(train_fn, device_list = device_list, active_run_id = active_run_id)


# Best Practice You Are! Previously we have 2 runs, one for master and the other for all workers.
# 1. master run id for recording system metrics only
# 2. Inner run id for auto-logging and manually logging other artifacts.

# Now I have changed the code to merge outter run and inner run to be the same run id. less confusion.
