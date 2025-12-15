# Databricks notebook source
# MAGIC %md
# MAGIC **NOTES:** 
# MAGIC
# MAGIC Tested with classic MLdbr compute e.g. `16.4LTS MLdbr + (128 gigs)`
# MAGIC
# MAGIC Example Cluster Config. [`JSON` to create]:
# MAGIC ```
# MAGIC {
# MAGIC     "cluster_name": "mmt_testing4DLworkloads_gpuMLdbr_multinode",
# MAGIC     "spark_version": "16.4.x-scala2.13",
# MAGIC     "aws_attributes": {
# MAGIC         "zone_id": "auto"
# MAGIC     },
# MAGIC     "node_type_id": "g5.12xlarge",
# MAGIC     "custom_tags": {
# MAGIC         "ASQ_testing": "True",
# MAGIC         "RemoveAfte": "2025-12-31"
# MAGIC     },
# MAGIC     "autotermination_minutes": 120,
# MAGIC     "single_user_name": "may.merkletan@databricks.com",
# MAGIC     "data_security_mode": "DATA_SECURITY_MODE_AUTO",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "kind": "CLASSIC_PREVIEW",
# MAGIC     "use_ml_runtime": true,
# MAGIC     "is_single_node": false,
# MAGIC     "autoscale": {
# MAGIC         "min_workers": 2,
# MAGIC         "max_workers": 4
# MAGIC     },
# MAGIC     "apply_policy_default_values": false
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Volume in UC if it does not already exist

# COMMAND ----------

# DBTITLE 1,We specify the Catalog/Schema/Volumes Path
## update with your specific catalog and schema and volume names

CATALOG = "mmt"
SCHEMA = "pytorch"
VOLUME_NAME = "torch_data"

# COMMAND ----------

# DBTITLE 1,create the volume if not already done so
sql = f"""
CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}
COMMENT 'Volume for torch training data'
"""
spark.sql(sql)

# COMMAND ----------

# DBTITLE 1,check the create volume in UC
display(
    spark.sql(f"SHOW VOLUMES IN {CATALOG}.{SCHEMA}")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download `Tiny ImageNet` dataset
# MAGIC
# MAGIC The `Tiny ImageNet dataset` is a smaller-scale version of the ImageNet dataset, designed specifically for academic and machine learning tasks that require a more manageable training size. The original ImageNet dataset, with its 1,000 classes and high-resolution images, requires significant computational power, so `Tiny ImageNet` is often used as a benchmark for training and evaluating image classification models more quickly.
# MAGIC
# MAGIC Key features
# MAGIC - Size: The dataset contains 100,000 images, which are divided into 200 distinct object categories or classes. 
# MAGIC - Resolution: All images in the dataset are down-sampled to a resolution of `64x64` pixels, making it less computationally expensive to process than the larger ImageNet. 
# MAGIC - Structure: It is split into training, validation, and testing sets.Training set: Contains 500 images for each of the 200 classes, totaling 100,000 images.
# MAGIC - Validation set: Contains 50 images per class for a total of 10,000 images.
# MAGIC - Test set: Contains 50 images per class, for a total of 10,000 images, which are typically used for final, unbiased evaluation.
# MAGIC - Origin: Tiny   was created by Stanford University for its CS231n computer vision course and is based on a subset of the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) benchmark.

# COMMAND ----------

# DBTITLE 1,download_tiny_imagenet
import os
import urllib.request

def download_tiny_imagenet(dest_dir: str = "Volumes/") -> None:
    url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path: str = os.path.join(dest_dir, "tiny-imagenet-200.zip")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

# COMMAND ----------

# DBTITLE 1,apply function to download imgnet dataset
dest_dir=f'/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}'

download_tiny_imagenet(dest_dir=dest_dir)

# COMMAND ----------

# DBTITLE 1,show downloaded zip file
display(dbutils.fs.ls(dest_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC **NB:** **A common pattern for handling archives in Databricks:** `unzip` command-line tool cannot write directly to cloud-backed paths like `/Volumes/... (Unity Catalog Volumes)` or `DBFS`. These paths are not standard `POSIX` filesystems, and many native `OS` utilities (like `unzip`) do not support writing directly to them. We ensure compatibility by `first unzipping locally` and subsequently use `Python` (with `shutil.copytree`) to copy the extracted files to the desired Unity Catalog Volume path, which is accessible via Databricks APIs and Python file operations. 

# COMMAND ----------

# DBTITLE 1,(slow ~4hrs) unzip_imgfiles
# import shutil
# import tempfile
# import subprocess

# def unzip_imgfiles(
#     zip_path: str,
#     dest_dir: str
# ) -> None:
#     with tempfile.TemporaryDirectory() as tmpdir:
#         print(f"Unzipping {zip_path} to local temp dir {tmpdir}...")
#         subprocess.run(["unzip", "-q", zip_path, "-d", tmpdir], check=True)
#         print("Local unzip complete.")

#         # Assumes single folder in zip, like 'tiny-imagenet-200'
#         extracted_root = os.path.join(tmpdir, os.listdir(tmpdir)[0])
#         print(f"Copying {extracted_root} to {dest_dir}...")
#         shutil.copytree(extracted_root, dest_dir, dirs_exist_ok=True)
#         print("Copy complete.")


# #NB# A common pattern for handling archives in Databricks: unzip command-line tool cannot write directly to cloud-backed paths like /Volumes/... (Unity Catalog Volumes) or DBFS. These paths are not standard POSIX filesystems, and many native OS utilities (like unzip) do not support writing directly to them. We ensure compatibility by first unzipping locally and subsequently use Python (with shutil.copytree) to copy the extracted files to the desired Unity Catalog Volume path, which is accessible via Databricks APIs and Python file operations. 

# COMMAND ----------

# DBTITLE 1,(faster ~2hrs) parallel unzip_imgfiles
import os
import shutil
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

def parallel_copytree(src, dst, max_workers=8): 
    os.makedirs(dst, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(src):
            rel_path = os.path.relpath(root, src)
            dest_root = os.path.join(dst, rel_path)
            os.makedirs(dest_root, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_root, file)
                futures.append(executor.submit(shutil.copy2, src_file, dst_file))
        for f in futures:
            f.result()

def unzip_imgfiles(zip_path: str, dest_dir: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # subprocess.run(["unzip", "-q", zip_path, "-d", tmpdir], check=True) ## you can silence the process but it might be helpful to see updates
        subprocess.run(["unzip", zip_path, "-d", tmpdir], check=True)
        extracted_root = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        parallel_copytree(extracted_root, dest_dir)

# COMMAND ----------

# DBTITLE 1,unzip via tmp path before copying to UC Vols
zip_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/tiny-imagenet-200.zip"
unzip_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200"

unzip_imgfiles(zip_path, unzip_path)

# COMMAND ----------

# DBTITLE 1,check path
display(dbutils.fs.ls(unzip_path))

# COMMAND ----------

# DBTITLE 1,recursively view files in path
files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(unzip_path)
)
display(files_df.select("path"))

# COMMAND ----------

# MAGIC %md
# MAGIC ---   

# COMMAND ----------

# MAGIC %md
# MAGIC ##   Create Spark Dataframes for conversion into [Mosaic Data Shard (MDS) ](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/dataset_format.html) for subsequent [Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/)

# COMMAND ----------

# DBTITLE 1,import dependencies
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from pyspark.sql.functions import col, lit, input_file_name, regexp_extract
import os

# COMMAND ----------

# DBTITLE 1,dataset_path
# Define the path to your Tiny ImageNet dataset
# Update this path to where you've uploaded the dataset in Databricks
dataset_path = unzip_path 

# COMMAND ----------

# DBTITLE 1,define schema for sparkDF
def create_tiny_imagenet_dataframe(dataset_path):
    """
    Create a Spark DataFrame from Tiny ImageNet 200 dataset
    
    Args:
        dataset_path: Path to the extracted tiny-imagenet-200 folder
    
    Returns:
        Spark DataFrame with columns: image_path, image_data, class_id, class_name, split
    """
    
    # Read class names mapping
    words_file = f"{dataset_path}/words.txt"
    
    # Create a DataFrame for class mappings
    class_mapping = {}
    try:
        with open(words_file.replace("dbfs:", "/dbfs"), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    class_mapping[parts[0]] = parts[1]
    except:
        print(f"Warning: Could not read {words_file}")
    
    # Define schema for the final DataFrame
    schema = StructType([
        StructField("image_path", StringType(), True),
        StructField("image_data", BinaryType(), True),
        StructField("class_id", StringType(), True),
        StructField("class_name", StringType(), True),
        StructField("split", StringType(), True)
    ])
    
    # Initialize list to store DataFrames
    dataframes = []
    
    # Process training data
    train_path = f"{dataset_path}/train"
    if os.path.exists(train_path.replace("dbfs:", "/dbfs")):
        print("Processing training data...")
        
        # Read all images from train directory
        train_images = spark.read.format("binaryFile")\
            .option("pathGlobFilter", "*.JPEG")\
            .option("recursiveFileLookup", "true")\
            .load(f"{train_path}/*")
        
        # Extract class_id from file path
        train_df = train_images.select(
            col("path").alias("image_path"),
            col("content").alias("image_data"),
            regexp_extract(col("path"), r"train/([^/]+)/", 1).alias("class_id"),
            lit("train").alias("split")
        )
        
        dataframes.append(train_df)
    
    # Process validation data
    val_path = f"{dataset_path}/val"
    if os.path.exists(val_path.replace("dbfs:", "/dbfs")):
        print("Processing validation data...")
        
        # Read validation annotations
        val_annotations = {}
        val_ann_file = f"{val_path}/val_annotations.txt"
        try:
            with open(val_ann_file.replace("dbfs:", "/dbfs"), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        val_annotations[parts[0]] = parts[1]
        except:
            print(f"Warning: Could not read {val_ann_file}")
        
        # Read all validation images
        val_images = spark.read.format("binaryFile")\
            .option("pathGlobFilter", "*.JPEG")\
            .load(f"{val_path}/images/*")
        
        # Extract filename and map to class_id
        val_df = val_images.select(
            col("path").alias("image_path"),
            col("content").alias("image_data"),
            regexp_extract(col("path"), r"([^/]+)\.JPEG$", 1).alias("filename"),
            lit("val").alias("split")
        )
        
        # Create a broadcast variable for validation annotations
        val_annotations_broadcast = spark.sparkContext.broadcast(val_annotations)
        
        def map_val_class(filename):
            return val_annotations_broadcast.value.get(filename + ".JPEG", "unknown")
        
        from pyspark.sql.functions import udf
        map_val_class_udf = udf(map_val_class, StringType())
        
        val_df = val_df.withColumn("class_id", map_val_class_udf(col("filename")))\
                      .drop("filename").select("image_path", "image_data", "class_id", "split")
        
        dataframes.append(val_df)
    
    # Process test data (if available)
    test_path = f"{dataset_path}/test"
    if os.path.exists(test_path.replace("dbfs:", "/dbfs")):
        print("Processing test data...")
        
        test_images = spark.read.format("binaryFile")\
            .option("pathGlobFilter", "*.JPEG")\
            .load(f"{test_path}/images/*")
        
        test_df = test_images.select(
            col("path").alias("image_path"),
            col("content").alias("image_data"),
            lit("unknown").alias("class_id"),
            lit("test").alias("split")
        )
        
        dataframes.append(test_df)
    
    
    # Combine all DataFrames
    if dataframes:
        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = combined_df.union(df)
        
        # Create a broadcast variable for class mapping
        class_mapping_broadcast = spark.sparkContext.broadcast(class_mapping)
        
        def map_class_name(class_id):
            return class_mapping_broadcast.value.get(class_id, "unknown")
        
        map_class_name_udf = udf(map_class_name, StringType())
        
        # Add class names
        final_df = combined_df.withColumn("class_name", map_class_name_udf(col("class_id")))
        
        return final_df
    else:
        return spark.createDataFrame([], schema)


# COMMAND ----------

# DBTITLE 1,Apply the sparkDF function
# Create the DataFrame
print("Creating Tiny ImageNet DataFrame...")
tiny_imagenet_df = create_tiny_imagenet_dataframe(dataset_path)

# Show some statistics
print("\nData distribution by split:")
tiny_imagenet_df.groupBy("split").count().show()

# COMMAND ----------

# DBTITLE 1,take a peek
display(tiny_imagenet_df)

# COMMAND ----------

# DBTITLE 1,write sparkDF to UC as delta table
delta_table_path2save = f"{CATALOG}.{SCHEMA}.imagenet_tiny200_delta" 
# delta_table_path2save

tiny_imagenet_df.write.format("delta").mode("overwrite").saveAsTable(delta_table_path2save)

# COMMAND ----------

# DBTITLE 1,check for the table we just wrote to UC
## List all tables in the catalog and schema
tables_df = spark.sql(
    f"SHOW TABLES IN {CATALOG}.{SCHEMA}"
)

# Filter for tables containing e.g. 'imagenet' in the name
filtered_df = tables_df.filter(
    tables_df.tableName.contains("imagenet")
)

display(filtered_df)

## or more generally
# display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}") )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert the data to MDS and save to UC Volumes

# COMMAND ----------

# DBTITLE 1,create train-val-test split sparkDFs
# Example: Filter for specific data classes or splits
train_df = tiny_imagenet_df.filter(col("split") == "train")
test_df = tiny_imagenet_df.filter(col("split") == "test")
val_df = tiny_imagenet_df.filter(col("split") == "val")

print(f"\nTraining images: {train_df.count()}")
print(f"\nTesting images: {test_df.count()}")
print(f"Validation images: {val_df.count()}")

# COMMAND ----------

# DBTITLE 1,check
display(train_df.limit(3))
display(val_df.limit(3))
display(test_df.limit(3)) ## test set doesn't have labels wrt competition/course submission -- split validation set into val/test if need be

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### [SparkDF to MDS format conversion](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/spark_dataframe_to_mds.html):
# MAGIC The process splits the Spark DataFrame among available workers, writes out data shards (`partitions`), and merges their index files into one (using `merge_index=True`), allowing scalable conversion no matter the dataset size or Spark cluster capacity. The output is fully compatible with PyTorch and other libraries for downstream usage. You can add preprocessing (like tokenization or other transformations) as a user-defined function (`UDF`) within the conversion for advanced use cases. The overall steps involved:
# MAGIC
# MAGIC - Load and (optionally) preprocess data with Apache Spark, resulting in a Spark DataFrame.
# MAGIC
# MAGIC - Convert SparkDf to MDS: Use the function streaming.base.converters.dataframe_to_mds to save the Spark DataFrame as MDS files.    
# MAGIC   - This function writes the data to disk (either local or cloud storage) in the standardized MDS format, supporting options like compression and hashing. 
# MAGIC   - You can also integrate advanced preprocessing steps (e.g. tokenization or other transformations) using user-defined functions (UDFs) if needed.
# MAGIC
# MAGIC - Parallel Writing for Efficiency: Mosaic Streaming leverages parallel writers (MDSWriter objects), each handling a partition of your DataFrame. 
# MAGIC   - After writing, the output index files from each partition are merged using `merge_index()` to create a unified dataset index for efficient loading.
# MAGIC   - Output: The MDS dataset can include raw data, tokenized data, or multimodal fields.
# MAGIC
# MAGIC - Result: The data is now sharded and indexed in the MDS format, ready for ingestion by [PyTorch-compatible `StreamingDataset` and `StreamingDataLoader` utilities](https://docs.mosaicml.com/projects/streaming/en/v0.2.3/), optimized for high-throughput distributed machine learning workflows 
# MAGIC   - Storage: Write to any file system supported by your environment, including Unity Catalog volumes and major cloud providers
# MAGIC   
# MAGIC
# MAGIC   

# COMMAND ----------

# DBTITLE 1,*define dataframe_to_mds
import shutil
from streaming.base.converters import dataframe_to_mds

partitions = 32 ## update where appropriate
data_storage_location = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# SparkDF to MDS format conversion
def save_mds_data(df, path, subdir):
  out_path = os.path.join(data_storage_location, subdir)
  print(f"Saving data to {out_path}")
  shutil.rmtree(out_path, ignore_errors=True)

  # mds_kwargs describes output location and column schemas
  mds_kwargs = {'out': out_path, 'columns': {'image_data': 'bytes', 'class_name':'str'}, 'size_limit':'100mb'}  

  # dataframe_to_mds(...) splits the data into partitions, writes shards, and merges the indexes into a master index file.
  dataframe_to_mds(df.repartition(partitions), merge_index=True, mds_kwargs=mds_kwargs)

# COMMAND ----------

# DBTITLE 1,run save_mds_data for data_splits
split_dfs = {
    "train": train_df, # ~40mins--1hr? takes the longest to process wrt volume of files
    "val": val_df,  # ~10mins 
    "test": test_df # ~<3mins | omit if not using 
}

for split, df in split_dfs.items():
  print(f"sparkDF2mds processing {split} data...")
  save_mds_data(
        df,
        data_storage_location,
        f"imagenet_tiny200_mds_{split}"
    )

# save_mds_data(train_df, data_storage_location, "imagenet_tiny200_mds_train") 

# COMMAND ----------

# DBTITLE 1,review added folders/paths
# data_storage_location = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
display(dbutils.fs.ls(data_storage_location))

display(dbutils.fs.ls(data_storage_location + "/imagenet_tiny200_mds_train"))

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### [Optional] Approach to append new MDS shards
# MAGIC
# MAGIC For situations where larger datasets are needed through combination of different datasets, the standard MDS conversion process does not handle incremental data appending well as it is designed for single, complete dataset writes.    
# MAGIC Here we provide an example of adopting a manual approach to build larger datasets incrementally while maintaining MDS format compatibility.

# COMMAND ----------

# DBTITLE 1,simulate 20x data expansion
import os 

df0 = spark.read.table(f"{CATALOG}.{SCHEMA}.imagenet_tiny200_delta")

# crossJoin is used here to replicate each row in df 20 times, effectively expanding the dataset by a factor of 20.
expanded_df_20X = df0.crossJoin(spark.range(20)).drop("id")  # Only if you want to drop repeat_id column

expanded_df_20X.groupBy("split").count().show()

# COMMAND ----------

# DBTITLE 1,simulate 5x data expansion
expanded_df_5X = df0.crossJoin(spark.range(5)).drop("id")  # Only if you want to drop repeat_id column

train_df_add = expanded_df_5X.filter(col("split") == "train")

print(f"\nTraining images: {train_df_add.count()}")

# COMMAND ----------

# DBTITLE 1,filter data splits from expanded sparkDF
from pyspark.sql import functions as F

# Example: Filter for specific classes or splits
train_df = expanded_df_20X.filter(F.col("split") == "train")
test_df = expanded_df_20X.filter(F.col("split") == "test")
val_df = expanded_df_20X.filter(F.col("split") == "val")

print(f"\nTraining images: {train_df.count()}")
print(f"\nTesting images: {test_df.count()}")
print(f"Validation images: {val_df.count()}")

# COMMAND ----------

# DBTITLE 1,previously defined above
# import shutil
# from streaming.base.converters import dataframe_to_mds

# partitions = 32
# data_storage_location = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

## SparkDF to MDS format conversion
# def save_mds_data(df, path, subdir):
#   out_path = os.path.join(data_storage_location, subdir)
#   print(f"Saving data to {out_path}")
#   shutil.rmtree(out_path, ignore_errors=True)

#   # mds_kwargs describes output location and column schemas
#   mds_kwargs = {'out': out_path, 'columns': {'image_data': 'bytes', 'class_name':'str'}, 'size_limit':'100mb'}  

#   # dataframe_to_mds(...) splits the data into partitions, writes shards, and merges the indexes into a master index file.
#   dataframe_to_mds(df.repartition(partitions), merge_index=True, mds_kwargs=mds_kwargs)

# COMMAND ----------

# DBTITLE 1,apply to expanded data
save_mds_data(train_df, data_storage_location, "imagenet_tiny200_mds_20X_train_append")

# COMMAND ----------

# DBTITLE 1,check
display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))
len(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Safely append new data to an existing MDS dataset.    
# MAGIC
# MAGIC Append new DataFrame data to an existing MDS dataset without corrupting the existing index files.
# MAGIC
# MAGIC Step-by-step process:
# MAGIC
# MAGIC - Creates temporary directory: Uses `tmp_out_path` as a staging area within the target MDS directory
# MAGIC - Converts DataFrame to MDS: Writes the new data to the temporary folder with `merge_index=False` to avoid index conflicts
# MAGIC - Moves files safely: Transfers all generated MDS files (shards and index files) from the temporary folder to the main MDS directory
# MAGIC - Cleans up: Removes the temporary folder after successful transfer
# MAGIC
# MAGIC Key differences from regular save_mds_data:
# MAGIC
# MAGIC - Uses `merge_index=False` instead of True to prevent overwriting existing index files
# MAGIC Implements a two-stage write process (`temp → final`) for safety
# MAGIC - Allows incremental dataset building without recreating the entire MDS dataset
# MAGIC - This function is essential for the dataset expansion workflow you saw in earlier cells, where you were adding 5X and 20X replicated data to increase your training dataset size. 
# MAGIC - After using this function, you'd typically call `merge_index()` manually to consolidate all the separate index files into one unified index.

# COMMAND ----------

# DBTITLE 1,define save_mds_data_append
def save_mds_data_append(df, path, subdir):
  import os
  import shutil
  from datetime import datetime

  out_path = os.path.join(data_storage_location, subdir)
  tmp_out_path = os.path.join(out_path, "tmp")
  backup_path = os.path.join(data_storage_location, f"{subdir}_previous")
  
  print(f"Saving data to temporary folder {tmp_out_path}")

  # Remove tmp folder if it exists
  shutil.rmtree(tmp_out_path, ignore_errors=True)
  
  # [update] Create backup of existing MDS files before appending
  if os.path.exists(out_path):
    print(f"Creating backup of existing MDS files to {backup_path}")
    # Remove old backup if it exists
    shutil.rmtree(backup_path, ignore_errors=True)
    # Copy current files to backup location
    shutil.copytree(out_path, backup_path)
    print(f"Backup created successfully")

  mds_kwargs = {'out': tmp_out_path, 'columns': {'image_data': 'bytes', 'class_name':'str'}, 'size_limit':'100mb'}

  dataframe_to_mds(df, merge_index=False, mds_kwargs=mds_kwargs)

  ## Include Append logic here
  # Move data from tmp folder to out_path
  for item in os.listdir(tmp_out_path):
    s = os.path.join(tmp_out_path, item)
    d = os.path.join(out_path, item)
    if os.path.isdir(s):
      shutil.move(s, d)
    else:
      shutil.move(s, d)
  shutil.rmtree(tmp_out_path, ignore_errors=True)
  
  print(f"Data appended successfully. Backup available at {backup_path}")

# COMMAND ----------

# DBTITLE 1,append mds
save_mds_data_append(train_df_add, data_storage_location, "imagenet_tiny200_mds_20X_train_append")

# COMMAND ----------

# DBTITLE 1,check n_files
display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))

len(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### `Manually` consolidate all the separate index files into one unified index.

# COMMAND ----------

# DBTITLE 1,merge_index
import os
from streaming.base.util import merge_index

## 2025Dec -- code requires removal of json prior to merge and reindexing
index_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/index.json"
if os.path.exists(index_path):
    print(f"Removing existing index file: {index_path}")
    os.remove(index_path)

merge_index(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append")

# COMMAND ----------

# DBTITLE 1,check
display(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))
len(dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append"))

# COMMAND ----------

# DBTITLE 1,summarize_mosaic_index_with_shards
import json
import pandas as pd
from datetime import datetime
import os

def summarize_mosaic_index_with_shards(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Get index file timestamp
    index_file_timestamp = datetime.fromtimestamp(os.path.getmtime(path))
    
    # Get directory of index file to resolve relative paths
    index_dir = os.path.dirname(path)

    shards = data.get("shards", []) or []
    n_shards = len(shards)
    total_samples = sum((s.get("samples") or 0) for s in shards)
    total_raw_bytes = sum((s.get("raw_data", {}).get("bytes") or 0) for s in shards)
    total_zip_bytes = sum((s.get("zip_data", {}).get("bytes") or 0) for s in shards)

    # Per-shard details with timestamps
    shard_details = []
    for idx, s in enumerate(shards):
        # Get zip file timestamp
        zip_basename = (s.get("zip_data", {}) or {}).get("basename")
        zip_timestamp = None
        if zip_basename:
            zip_path = os.path.join(index_dir, zip_basename)
            if os.path.exists(zip_path):
                zip_timestamp = datetime.fromtimestamp(os.path.getmtime(zip_path))

        shard_details.append({
            "shard_index": idx,
            "samples": s.get("samples"),
            "raw_bytes": (s.get("raw_data", {}) or {}).get("bytes"),
            "zip_bytes": (s.get("zip_data", {}) or {}).get("bytes"),
            "zip_basename": zip_basename,
            "zip_timestamp": zip_timestamp,
            "compression": s.get("compression"),
            "format": s.get("format"),
            "version": s.get("version"),
        })

    return {
        "num_shards": n_shards,
        "total_samples": total_samples,
        "total_raw_bytes": total_raw_bytes,
        "total_zip_bytes": total_zip_bytes,
        "root_version": data.get("version"),
        "index_file_timestamp": index_file_timestamp,
        "shards": shard_details,
    }

# COMMAND ----------

# DBTITLE 1,get summaries
info = summarize_mosaic_index_with_shards(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/index.json")

# Print high-level summary
print("=== Mosaic Index Summary ===")
print(f"Index file timestamp: {info['index_file_timestamp']}")
print(f"Number of shards: {info['num_shards']}")
print(f"Total samples: {info['total_samples']:,}")
print(f"Total raw bytes: {info['total_raw_bytes']:,}")
print(f"Total zip bytes: {info['total_zip_bytes']:,}")
print(f"Root version: {info['root_version']}")

# COMMAND ----------

# DBTITLE 1,shard details
# Create DataFrame for shard details
shards_df = pd.DataFrame(info['shards'])
display(shards_df)

print(shards_df.samples.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local Path performance analysis

# COMMAND ----------

# DBTITLE 1,test_streaming_performance [1]
import time
import shutil
import os
from streaming import StreamingDataset
from torch.utils.data import DataLoader

def test_streaming_performance(remote_path, local_path, test_name, batch_size=32, num_batches=100):
    """Test streaming performance for MDS data"""
    
    print(f"\n{'='*50}")
    print(f"Testing: {test_name}")
    print(f"Remote: {remote_path}")
    print(f"Local: {local_path}")
    print(f"{'='*50}")
    
    # Clean up local cache
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    # Initialize dataset and dataloader
    start_time = time.time()
    dataset = StreamingDataset(
        remote=remote_path,
        local=local_path,
        batch_size=batch_size,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )
    
    init_time = time.time() - start_time
    print(f"Dataset initialization: {init_time:.2f}s")
    
    # Test data loading
    batch_times = []
    total_samples = 0
    first_batch_time = None
    
    overall_start = time.time()
    dataloader_iter = iter(dataloader)
    
    for i in range(num_batches):
        batch_start = time.time()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
            
        # Get batch size (handle dict or tensor)
        if isinstance(batch, dict):
            batch_size_actual = len(next(iter(batch.values())))
        else:
            batch_size_actual = len(batch)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_samples += batch_size_actual
        
        if i == 0:
            first_batch_time = batch_time
            print(f"First batch load time: {batch_time:.3f}s")
        
        if i % 20 == 0:
            print(f"Batch {i} load time: {batch_time:.3f}s")
    
    total_loading_time = time.time() - overall_start
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    throughput = total_samples / total_loading_time if total_loading_time > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total batches: {len(batch_times)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total loading time: {total_loading_time:.3f}s")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'test_name': test_name,
        'init_time': init_time,
        'first_batch_time': first_batch_time,
        'avg_batch_time': avg_batch_time,
        'total_loading_time': total_loading_time,
        'throughput': throughput,
        'total_samples': total_samples
    }

def compare_local_paths(remote_path, batch_size=32, num_batches=100):
    """Compare /tmp vs /local_disk0/tmp performance"""
    
    print("Comparing Local Path Performance for MDS Data")
    print("="*60)
    
    # Test configurations
    results = {}
    
    # Test /tmp
    results['tmp'] = test_streaming_performance(
        remote_path=remote_path,
        local_path='/tmp/streaming_cache',
        test_name='/tmp',
        batch_size=batch_size,
        num_batches=num_batches
    )
    
    # Brief pause
    time.sleep(2)
    
    # Test /local_disk0/tmp
    results['local_disk0'] = test_streaming_performance(
        remote_path=remote_path,
        local_path='/local_disk0/tmp/streaming_cache',
        test_name='/local_disk0/tmp',
        batch_size=batch_size,
        num_batches=num_batches
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"{'Metric':<25} {'/tmp':<15} {'/local_disk0/tmp':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Compare metrics
    metrics = [
        ('Init Time (s)', 'init_time'),
        ('First Batch Time (s)', 'first_batch_time'),
        ('Avg Batch Time (s)', 'avg_batch_time'),
        ('Total Loading Time (s)', 'total_loading_time'),
        ('Throughput (samples/s)', 'throughput')
    ]
    
    for metric_name, key in metrics:
        tmp_val = results['tmp'][key]
        local_val = results['local_disk0'][key]
        
        if key == 'throughput':
            improvement = f"{(local_val/tmp_val - 1)*100:.1f}%" if tmp_val > 0 else "N/A"
        else:
            improvement = f"{(tmp_val/local_val - 1)*100:.1f}%" if local_val > 0 else "N/A"
        
        print(f"{metric_name:<25} {tmp_val:<15.3f} {local_val:<15.3f} {improvement:<15}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if results['local_disk0']['throughput'] > results['tmp']['throughput']:
        speedup = results['local_disk0']['throughput'] / results['tmp']['throughput']
        print(f"Use /local_disk0/tmp - {speedup:.1f}x faster throughput")
    else:
        print("/tmp performed better (unexpected - check cluster config)")
    
    return results

# COMMAND ----------

# DBTITLE 1,apply compare_local_paths
# Usage

# Replace with your actual Volume path containing MDS data
REMOTE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/"

# Run the comparison
results = compare_local_paths(
    remote_path=REMOTE_PATH,
    batch_size=256,
    num_batches=5000
)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,check /local_disk0
# MAGIC %sh df -h /local_disk0

# COMMAND ----------

# DBTITLE 1,check /tmp
# MAGIC %sh df -h /tmp

# COMMAND ----------


