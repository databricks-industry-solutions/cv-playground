# Databricks notebook source
# MAGIC %md
# MAGIC Run with serverless compute or classic MLdbr compute e.g. `16.4LTS MLdbr + (64 -- 128 gigs)`

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
from streaming.base.util import merge_index

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



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Include additional metrics
import time
import shutil
import os
import mlflow
import mlflow.system_metrics
import torch
from streaming import StreamingDataset
from torch.utils.data import DataLoader

def setup_mlflow_system_metrics():
    """Properly configure MLflow system metrics"""
    try:
        # Configure system metrics before starting any runs
        mlflow.system_metrics.enable_system_metrics_logging()
        
        # You can also configure specific metrics and sampling intervals
        # This should be done at the start of your script, not per run
        print("MLflow system metrics enabled globally")
        return True
    except Exception as e:
        print(f"Failed to enable system metrics: {e}")
        return False

def log_performance_metrics_only(results, remote_path, config=None):
    """Log ONLY performance metrics - let system metrics handle resources automatically"""
    try:
        # Log parameters (dataset and test configuration)
        mlflow.log_param("remote_path", remote_path)
        mlflow.log_param("local_path_type", results['test_name'])
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        mlflow.log_param("gpu_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
        
        if config:
            for key, value in config.items():
                mlflow.log_param(key, value)
        
        # Log ONLY performance metrics (not system resources)
        performance_metrics = {
            "init_time_seconds": results.get('init_time'),
            "first_batch_time_seconds": results.get('first_batch_time'), 
            "avg_batch_time_seconds": results.get('avg_batch_time'),
            "total_loading_time_seconds": results.get('total_loading_time'),
            "throughput_samples_per_sec": results.get('throughput'),
            "total_samples_processed": results.get('total_samples'),
            "avg_gpu_transfer_time_seconds": results.get('avg_gpu_transfer_time', 0)
        }
        
        for metric_name, value in performance_metrics.items():
            if value is not None and value > 0:
                mlflow.log_metric(metric_name, value)
        
        print("Performance metrics logged to MLflow")
                
    except Exception as e:
        print(f"MLflow logging failed: {e}")

def test_streaming_performance_clean(remote_path, local_path, test_name, 
                                   batch_size=32, num_batches=100, num_workers=4, 
                                   predownload=None, cache_limit='50GB', 
                                   device='cuda', enable_gpu_transfer=True):
    """Clean streaming performance test - no manual resource monitoring"""
    
    print(f"\n{'='*50}")
    print(f"Testing: {test_name}")
    print(f"Remote: {remote_path}")
    print(f"Local: {local_path}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    print(f"Device: {device}, GPU Transfer: {enable_gpu_transfer}")
    print(f"{'='*50}")
    
    # Check GPU availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
        enable_gpu_transfer = False
    
    # Clean up local cache
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except Exception as e:
        print(f"Cache cleanup failed: {e}")
        return None
    
    # Initialize dataset and dataloader
    start_time = time.time()
    try:
        if predownload is None:
            predownload = min(batch_size * 4, 1000)
            
        dataset = StreamingDataset(
            remote=remote_path,
            local=local_path,
            batch_size=batch_size,
            shuffle=True,
            download_retry=3,
            download_timeout=300,
            predownload=predownload,
            keep_zip=True,
            cache_limit=cache_limit
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=enable_gpu_transfer,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        return None
    
    init_time = time.time() - start_time
    print(f"Dataset initialization: {init_time:.2f}s")
    
    # Test data loading with optional GPU transfer
    batch_times = []
    gpu_transfer_times = []
    total_samples = 0
    first_batch_time = None
    
    overall_start = time.time()
    
    try:
        dataloader_iter = iter(dataloader)
        
        for i in range(num_batches):
            batch_start = time.time()
            
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                print(f"Dataset exhausted at batch {i}")
                break
            except Exception as e:
                print(f"Error loading batch {i}: {e}")
                continue
            
            # Optional GPU transfer timing
            gpu_transfer_time = 0
            if enable_gpu_transfer and device == 'cuda':
                gpu_start = time.time()
                if isinstance(batch, dict):
                    batch = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v 
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [item.to(device, non_blocking=True) if hasattr(item, 'to') else item 
                            for item in batch]
                else:
                    if hasattr(batch, 'to'):
                        batch = batch.to(device, non_blocking=True)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_transfer_time = time.time() - gpu_start
                gpu_transfer_times.append(gpu_transfer_time)
            
            # Get batch size
            try:
                if isinstance(batch, dict):
                    batch_size_actual = len(next(iter(batch.values())))
                elif isinstance(batch, (list, tuple)):
                    batch_size_actual = len(batch[0]) if batch else 0
                else:
                    batch_size_actual = len(batch)
            except Exception as e:
                print(f"Error determining batch size: {e}")
                batch_size_actual = batch_size
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_samples += batch_size_actual
            
            if i == 0:
                first_batch_time = batch_time
                print(f"First batch load time: {batch_time:.3f}s")
                if gpu_transfer_time > 0:
                    print(f"First GPU transfer time: {gpu_transfer_time:.3f}s")
            
            if i % 20 == 0 and i > 0:
                gpu_info = f", GPU: {gpu_transfer_time:.3f}s" if gpu_transfer_time > 0 else ""
                print(f"Batch {i} load time: {batch_time:.3f}s{gpu_info}")
    
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None
    
    total_loading_time = time.time() - overall_start
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_gpu_transfer_time = sum(gpu_transfer_times) / len(gpu_transfer_times) if gpu_transfer_times else 0
    throughput = total_samples / total_loading_time if total_loading_time > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total batches: {len(batch_times)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total loading time: {total_loading_time:.3f}s")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    if avg_gpu_transfer_time > 0:
        print(f"  Average GPU transfer time: {avg_gpu_transfer_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'test_name': test_name,
        'init_time': init_time,
        'first_batch_time': first_batch_time,
        'avg_batch_time': avg_batch_time,
        'avg_gpu_transfer_time': avg_gpu_transfer_time,
        'total_loading_time': total_loading_time,
        'throughput': throughput,
        'total_samples': total_samples,
        'config': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'predownload': predownload,
            'cache_limit': cache_limit,
            'device': device,
            'enable_gpu_transfer': enable_gpu_transfer
        }
    }

def run_single_test_with_system_metrics(remote_path, local_path, test_name, 
                                      batch_size=32, num_batches=100, num_workers=4,
                                      device='cuda', enable_gpu_transfer=True):
    """Run a single test with proper MLflow system metrics tracking"""
    
    run_name = f"streaming_perf_{test_name}_{int(time.time())}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\nStarting MLflow run: {run_name}")
        
        # Log test configuration as parameters
        mlflow.log_param("test_name", test_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("num_batches", num_batches)
        mlflow.log_param("device", device)
        mlflow.log_param("gpu_transfer_enabled", enable_gpu_transfer)
        mlflow.log_param("local_path", local_path)
        
        # Run the actual performance test
        # System metrics will be automatically collected during this time
        result = test_streaming_performance_clean(
            remote_path=remote_path,
            local_path=local_path,
            test_name=test_name,
            batch_size=batch_size,
            num_workers=num_workers,
            num_batches=num_batches,
            device=device,
            enable_gpu_transfer=enable_gpu_transfer
        )
        
        if result:
            # Log only performance metrics - system metrics are automatic
            log_performance_metrics_only(result, remote_path, result.get('config'))
            print(f"Run completed: {run_name}")
        else:
            print(f"Run failed: {run_name}")
            
        return result

def compare_local_paths_with_proper_system_metrics(remote_path, batch_size=32, num_batches=100, 
                                                 num_workers=4, test_gpu_transfer=True):
    """Enhanced comparison with proper MLflow system metrics"""
    
    print("Comparing Local Path Performance for MDS Data")
    print("System metrics will be automatically tracked by MLflow")
    print("="*70)
    
    results = {}
    
    # Test configurations
    configs = [
        {
            'name': 'tmp_cpu', 
            'local_path': '/tmp/streaming_cache', 
            'device': 'cpu', 
            'gpu_transfer': False
        },
        {
            'name': 'local_disk0_cpu', 
            'local_path': '/local_disk0/tmp/streaming_cache', 
            'device': 'cpu', 
            'gpu_transfer': False
        }
    ]
    
    # Add GPU tests if CUDA is available and requested
    if test_gpu_transfer and torch.cuda.is_available():
        configs.extend([
            {
                'name': 'tmp_gpu', 
                'local_path': '/tmp/streaming_cache_gpu', 
                'device': 'cuda', 
                'gpu_transfer': True
            },
            {
                'name': 'local_disk0_gpu', 
                'local_path': '/local_disk0/tmp/streaming_cache_gpu', 
                'device': 'cuda', 
                'gpu_transfer': True
            }
        ])
    
    for config in configs:
        print(f"\nRunning test: {config['name']}")
        
        result = run_single_test_with_system_metrics(
            remote_path=remote_path,
            local_path=config['local_path'],
            test_name=config['name'],
            batch_size=batch_size,
            num_workers=num_workers,
            num_batches=num_batches,
            device=config['device'],
            enable_gpu_transfer=config['gpu_transfer']
        )
        
        if result:
            results[config['name']] = result
        
        # Brief pause between tests
        time.sleep(2)
    
    # Print comparison results
    if len(results) >= 2:
        print_comparison_results(results)
    
    return results

def benchmark_suite_with_proper_system_metrics(remote_path, quick_mode=False):
    """Complete benchmark suite with proper MLflow system metrics"""
    
    print("MDS Streaming Performance Benchmark Suite")
    print("System metrics automatically tracked by MLflow")
    print("="*70)
    
    # Set up experiment
    experiment_name = f"mds_streaming_benchmark_{int(time.strftime('%Y%m%d_%H%M%S'))}"
    try:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment: {experiment_name}")
        print(f"View results: mlflow ui")
        print(f"   System metrics will appear in the 'System Metrics' tab of each run")
    except Exception as e:
        print(f"MLflow experiment setup failed: {e}")
    
    if quick_mode:
        print("Running in QUICK MODE (reduced iterations)")
        num_batches = 50
    else:
        print("Running in COMPREHENSIVE MODE")
        num_batches = 100
    
    all_results = {}
    
    # 1. Basic comparison
    print("\nPhase 1: Basic Storage Location Comparison")
    basic_results = compare_local_paths_with_proper_system_metrics(
        remote_path=remote_path,
        batch_size=32,
        num_batches=num_batches,
        num_workers=4,
        test_gpu_transfer=True
    )
    all_results['basic'] = basic_results
    
    # 2. Scaling test - different batch sizes
    print("\nPhase 2: Batch Size Scaling Test")
    batch_sizes = [16, 32, 64, 128] if not quick_mode else [32, 64]
    scaling_results = {}
    
    for bs in batch_sizes:
        print(f"\nTesting batch size: {bs}")
        
        result = run_single_test_with_system_metrics(
            remote_path=remote_path,
            local_path=f'/local_disk0/tmp/streaming_cache_bs{bs}',
            test_name=f'scaling_bs{bs}',
            batch_size=bs,
            num_workers=4,
            num_batches=min(num_batches, 1000 // bs),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_gpu_transfer=torch.cuda.is_available()
        )
        
        if result:
            scaling_results[f'batch_size_{bs}'] = result
        
        time.sleep(1)
    
    all_results['scaling'] = scaling_results
    
    # 3. Worker scaling test
    print("\nPhase 3: Worker Scaling Test")
    worker_counts = [2, 4, 8, 16] if not quick_mode else [4, 8]
    worker_results = {}
    
    for workers in worker_counts:
        print(f"\nTesting {workers} workers")
        
        result = run_single_test_with_system_metrics(
            remote_path=remote_path,
            local_path=f'/local_disk0/tmp/streaming_cache_w{workers}',
            test_name=f'workers_{workers}',
            batch_size=32,
            num_workers=workers,
            num_batches=num_batches,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            enable_gpu_transfer=torch.cuda.is_available()
        )
        
        if result:
            worker_results[f'workers_{workers}'] = result
        
        time.sleep(1)
    
    all_results['worker_scaling'] = worker_results
    
    # 4. Generate comprehensive report
    generate_final_report(all_results, remote_path, experiment_name)
    
    return all_results

def print_comparison_results(results):
    """Print comprehensive comparison results"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    test_names = list(results.keys())
    print(f"{'Metric':<25}", end="")
    for name in test_names:
        print(f"{name:<20}", end="")
    print()
    print("-" * (25 + 20 * len(test_names)))
    
    metrics = [
        ('Throughput (samples/s)', 'throughput'),
        ('Avg Batch Time (s)', 'avg_batch_time'),
        ('GPU Transfer Time (s)', 'avg_gpu_transfer_time'),
        ('Init Time (s)', 'init_time')
    ]
    
    for metric_name, key in metrics:
        print(f"{metric_name:<25}", end="")
        for name in test_names:
            value = results[name].get(key, 0) or 0
            print(f"{value:<20.3f}", end="")
        print()
    
    # Find best performer
    best_result = max(results.values(), key=lambda x: x.get('throughput', 0))
    print(f"\nBest performer: {best_result['test_name']} "
          f"({best_result.get('throughput', 0):.1f} samples/sec)")

def generate_final_report(all_results, remote_path, experiment_name):
    """Generate final benchmark report"""
    
    print("\n" + "="*80)
    print("FINAL BENCHMARK REPORT")
    print("="*80)
    
    print(f"Dataset: {remote_path}")
    print(f"MLflow Experiment: {experiment_name}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // (1024**3)} GB)")
    
    # Create a summary run with overall findings
    try:
        with mlflow.start_run(run_name=f"benchmark_summary_{int(time.time())}"):
            # Log summary parameters
            mlflow.log_param("benchmark_type", "comprehensive_summary")
            mlflow.log_param("remote_path", remote_path)
            mlflow.log_param("total_test_phases", len(all_results))
            mlflow.log_param("experiment_name", experiment_name)
            
            # Find and log best configurations
            best_overall = None
            best_throughput = 0
            
            for phase_name, phase_results in all_results.items():
                if isinstance(phase_results, dict):
                    for test_name, result in phase_results.items():
                        if result and result.get('throughput', 0) > best_throughput:
                            best_throughput = result['throughput']
                            best_overall = {
                                'phase': phase_name,
                                'test': test_name,
                                'result': result
                            }
            
            if best_overall:
                mlflow.log_param("best_config_phase", best_overall['phase'])
                mlflow.log_param("best_config_test", best_overall['test'])
                mlflow.log_metric("best_throughput_samples_per_sec", best_throughput)
                
                # Log best configuration details
                best_config = best_overall['result'].get('config', {})
                for key, value in best_config.items():
                    mlflow.log_param(f"best_config_{key}", value)
                
                print("Overall best configuration logged to summary run")
    
    except Exception as e:
        print(f"Failed to create summary run: {e}")
    
    # 1. Storage Location Analysis
    if 'basic' in all_results:
        print("\nSTORAGE LOCATION ANALYSIS")
        print("-" * 40)
        
        basic = all_results['basic']
        storage_comparison = []
        
        for test_name, result in basic.items():
            if result:
                storage_comparison.append({
                    'name': test_name,
                    'throughput': result.get('throughput', 0),
                    'avg_batch_time': result.get('avg_batch_time', 0),
                    'gpu_transfer_time': result.get('avg_gpu_transfer_time', 0)
                })
        
        storage_comparison.sort(key=lambda x: x['throughput'], reverse=True)
        
        print("Ranking by throughput:")
        for i, item in enumerate(storage_comparison, 1):
            gpu_info = f" (GPU: {item['gpu_transfer_time']:.3f}s)" if item['gpu_transfer_time'] > 0 else ""
            print(f"  {i}. {item['name']}: {item['throughput']:.1f} samples/sec{gpu_info}")
    
    # 2. Batch Size Scaling Analysis
    if 'scaling' in all_results:
        print("\nBATCH SIZE SCALING ANALYSIS")
        print("-" * 40)
        
        scaling = all_results['scaling']
        batch_analysis = []
        
        for test_name, result in scaling.items():
            if result:
                bs = int(test_name.split('_')[-1])
                batch_analysis.append({
                    'batch_size': bs,
                    'throughput': result.get('throughput', 0),
                    'efficiency': result.get('throughput', 0) / bs if bs > 0 else 0
                })
        
        batch_analysis.sort(key=lambda x: x['batch_size'])
        
        print("Throughput by batch size:")
        optimal_bs = max(batch_analysis, key=lambda x: x['throughput']) if batch_analysis else None
        
        for item in batch_analysis:
            marker = " [OPTIMAL]" if optimal_bs and item['batch_size'] == optimal_bs['batch_size'] else ""
            print(f"  Batch size {item['batch_size']:3d}: {item['throughput']:8.1f} samples/sec "
                  f"({item['efficiency']:.1f} efficiency){marker}")
        
        if optimal_bs:
            print(f"\nOptimal batch size: {optimal_bs['batch_size']} "
                  f"({optimal_bs['throughput']:.1f} samples/sec)")
    
    # 3. Worker Scaling Analysis
    if 'worker_scaling' in all_results:
        print("\nWORKER SCALING ANALYSIS")
        print("-" * 40)
        
        worker_scaling = all_results['worker_scaling']
        worker_analysis = []
        
        for test_name, result in worker_scaling.items():
            if result:
                w = int(test_name.split('_')[-1])
                worker_analysis.append({
                    'workers': w,
                    'throughput': result.get('throughput', 0),
                    'efficiency': result.get('throughput', 0) / w if w > 0 else 0
                })
        
        worker_analysis.sort(key=lambda x: x['workers'])
        
        print("Throughput by worker count:")
        optimal_workers = max(worker_analysis, key=lambda x: x['throughput']) if worker_analysis else None
        
        for item in worker_analysis:
            marker = " [OPTIMAL]" if optimal_workers and item['workers'] == optimal_workers['workers'] else ""
            print(f"  {item['workers']:2d} workers: {item['throughput']:8.1f} samples/sec "
                  f"({item['efficiency']:.1f} samples/sec/worker){marker}")
        
        if optimal_workers:
            print(f"\nOptimal worker count: {optimal_workers['workers']} "
                  f"({optimal_workers['throughput']:.1f} samples/sec)")
    
    # 4. Final Recommendations
    print("\nFINAL RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    # Storage recommendation
    if 'basic' in all_results:
        basic = all_results['basic']
        local_disk_results = [r for k, r in basic.items() if 'local_disk0' in k and r]
        tmp_results = [r for k, r in basic.items() if 'tmp' in k and 'local_disk0' not in k and r]
        
        if local_disk_results and tmp_results:
            local_avg = sum(r['throughput'] for r in local_disk_results) / len(local_disk_results)
            tmp_avg = sum(r['throughput'] for r in tmp_results) / len(tmp_results)
            
            if local_avg > tmp_avg:
                speedup = local_avg / tmp_avg
                recommendations.append(f"Use /local_disk0/tmp for {speedup:.1f}x better performance")
            else:
                recommendations.append("/tmp performed unexpectedly well - verify cluster configuration")
    
    # Configuration recommendations
    if 'scaling' in all_results and all_results['scaling']:
        best_batch = max(all_results['scaling'].items(), 
                        key=lambda x: x[1].get('throughput', 0) if x[1] else 0)
        if best_batch[1]:
            bs = best_batch[0].split('_')[-1]
            recommendations.append(f"Optimal batch size: {bs}")
    
    if 'worker_scaling' in all_results and all_results['worker_scaling']:
        best_workers = max(all_results['worker_scaling'].items(), 
                          key=lambda x: x[1].get('throughput', 0) if x[1] else 0)
        if best_workers[1]:
            w = best_workers[0].split('_')[-1]
            recommendations.append(f"Optimal worker count: {w}")
    
    # GPU recommendations
    if torch.cuda.is_available() and 'basic' in all_results:
        gpu_results = [r for k, r in all_results['basic'].items() if 'gpu' in k and r]
        cpu_results = [r for k, r in all_results['basic'].items() if 'cpu' in k and r]
        
        if gpu_results and cpu_results:
            avg_gpu_transfer = sum(r.get('avg_gpu_transfer_time', 0) for r in gpu_results) / len(gpu_results)
            if avg_gpu_transfer > 0.005:  # More than 5ms
                recommendations.append(f"GPU transfer adds ~{avg_gpu_transfer:.3f}s/batch - consider optimization")
            else:
                recommendations.append("GPU transfer overhead is minimal")
    
    # Print recommendations
    for rec in recommendations:
        print(f"  {rec}")
    
    if not recommendations:
        print("  Run more comprehensive tests for detailed recommendations")
    
    # MLflow instructions
    print("\nVIEW DETAILED RESULTS:")
    print(f"   1. Run: mlflow ui")
    print(f"   2. Navigate to experiment: {experiment_name}")
    print(f"   3. Click on any run")
    print(f"   4. Go to 'System Metrics' tab to see:")
    print(f"      - CPU utilization over time")
    print(f"      - Memory usage patterns")
    print(f"      - Disk I/O activity")
    print(f"      - GPU utilization (if available)")
    print(f"      - Network I/O")
    print(f"   5. Compare runs using the 'Compare' feature")
    
    print(f"\n{'='*80}")

# Updated convenience functions
def quick_benchmark(remote_path):
    """Quick 5-minute benchmark with proper system metrics"""
    print("Quick Benchmark (5 minutes)")
    print("System metrics will be tracked automatically")
    
    # Enable system metrics globally
    setup_mlflow_system_metrics()
    
    return compare_local_paths_with_proper_system_metrics(
        remote_path=remote_path,
        batch_size=32,
        num_batches=30,
        num_workers=4,
        test_gpu_transfer=True
    )

def production_benchmark(remote_path):
    """Comprehensive production-ready benchmark with proper system metrics"""
    print("Production Benchmark (15-20 minutes)")
    print("System metrics will be tracked automatically")
    
    # Enable system metrics globally
    setup_mlflow_system_metrics()
    
    return benchmark_suite_with_proper_system_metrics(remote_path, quick_mode=False)

def development_benchmark(remote_path):
    """Development benchmark with proper system metrics and reduced iterations"""
    print("Development Benchmark (8-10 minutes)")
    print("System metrics will be tracked automatically")
    
    # Enable system metrics globally
    setup_mlflow_system_metrics()
    
    return benchmark_suite_with_proper_system_metrics(remote_path, quick_mode=True)

# Configuration helper
def configure_mlflow_for_databricks():
    """Configure MLflow settings optimized for Databricks"""
    try:
        # Set system metrics configuration
        import os
        
        # Configure system metrics sampling
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"  # Sample every 1 second
        os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_DISK_USAGE"] = "true"
        os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_GPU_USAGE"] = "true"
        os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_NETWORK_USAGE"] = "true"
        
        print("MLflow configured for Databricks with optimal system metrics settings")
        
    except Exception as e:
        print(f"MLflow configuration warning: {e}")

# Example usage for Databricks notebook
def notebook_example():
    """Example usage in Databricks notebook"""
    
    # Configure and enable system metrics
    configure_mlflow_for_databricks()
    setup_mlflow_system_metrics()
    
    # Set your dataset path
    # remote_path = "s3://your-bucket/your-mds-dataset"
    # Replace with your actual Volume path containing MDS data
    REMOTE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/"

    # Run quick benchmark
    print("Running quick benchmark...")
    results = quick_benchmark(REMOTE_PATH)
    
    # Display results
    if results:
        print("\nQuick Results Summary:")
        for test_name, result in results.items():
            if result:
                print(f"  {test_name}: {result.get('throughput', 0):.1f} samples/sec")
    
    print("\nView detailed system metrics in MLflow UI!")
    return results

# Sample data creation helper for testing
def create_sample_mds_dataset(local_path, num_samples=200):
    """Create sample MDS dataset for testing"""
    
    print(f"Creating sample MDS dataset: {local_path}")
    
    try:
        from streaming import MDSWriter
        import numpy as np
        
        os.makedirs(local_path, exist_ok=True)
        
        columns = {
            'data': 'bytes',
            'label': 'int'
        }
        
        with MDSWriter(out=local_path, columns=columns) as writer:
            for i in range(num_samples):
                # Create sample data
                sample_data = np.random.randint(0, 255, (100,), dtype=np.uint8).tobytes()
                
                sample = {
                    'data': sample_data,
                    'label': i % 10
                }
                writer.write(sample)
        
        print(f"Created {num_samples} samples")
        return local_path
        
    except Exception as e:
        print(f"Failed to create sample dataset: {e}")
        return None

# Main execution
def main():
    """Main execution with proper system metrics setup"""
    
    print("MDS Streaming Benchmark with System Metrics")
    print("="*60)
    
    # 1. Enable system metrics ONCE at the start
    print("1. Setting up MLflow system metrics...")
    system_metrics_enabled = setup_mlflow_system_metrics()
    
    if not system_metrics_enabled:
        print("System metrics setup failed - continuing without system metrics")
    
    # 2. Create sample dataset
    print("\n2. Creating sample dataset...")
    sample_path = "/local_disk0/tmp/mds_benchmark_sample"
    
    dataset_path = create_sample_mds_dataset(sample_path, num_samples=100)
    if not dataset_path:
        print("Failed to create sample dataset")
        return
    
    # 3. Run benchmark
    print("\n3. Running benchmark...")
    results = compare_local_paths_with_proper_system_metrics(
        remote_path=dataset_path,
        batch_size=16,
        num_batches=20  # Small for demo
    )
    
    if results:
        print("\nBenchmark completed successfully!")
        print("Check MLflow UI for system metrics in 'System Metrics' tab")
    else:
        print("\nBenchmark failed")

# if __name__ == "__main__":
#     main()

# COMMAND ----------



# COMMAND ----------

## Main execution
# if __name__ == "__main__":
# Configure MLflow for optimal system metrics collection
configure_mlflow_for_databricks()

# Enable system metrics globally
setup_mlflow_system_metrics()

# Example remote path - replace with your actual MDS dataset
# REMOTE_PATH = "s3://your-bucket/mds-data"
REMOTE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/"

print("MDS Streaming Performance Benchmark Tool")
print("System metrics will be automatically tracked in MLflow")
print("="*60)
print("Choose benchmark type:")
print("1. Quick (5 min) - Basic comparison")
print("2. Development (10 min) - Key scenarios")  
print("3. Production (20 min) - Comprehensive analysis")

# choice = input("Enter choice (1-3): ").strip()

# try:
#     if choice == "1":
#         results = quick_benchmark(REMOTE_PATH)
#     elif choice == "2":
#         results = development_benchmark(REMOTE_PATH)
#     elif choice == "3":
#         results = production_benchmark(REMOTE_PATH)
#     else:
#         print("Invalid choice, running quick benchmark...")
#         results = quick_benchmark(REMOTE_PATH)
    
#     print("\n✅ Benchmark completed successfully!")
#     print("🔗 To view system metrics:")
#     print("   1. Run: mlflow ui")
#     print("   2. Open any run from the experiment")
#     print("   3. Click the 'System Metrics' tab")
#     print("   4. View real-time resource utilization graphs")
    
# except Exception as e:
#     print(f"❌ Benchmark failed: {e}")
#     print("Check your remote path and ensure MDS dataset is accessible")


# COMMAND ----------

# DBTITLE 1,quick
quick_benchmark(REMOTE_PATH)

# COMMAND ----------

# DBTITLE 1,dev
development_benchmark(REMOTE_PATH)

# COMMAND ----------

# DBTITLE 1,prod
production_benchmark(REMOTE_PATH)

# COMMAND ----------

notebook_example()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,add mlflow system metrics
import time
import shutil
import os
import mlflow
import torch
from streaming import StreamingDataset
from torch.utils.data import DataLoader

def setup_mlflow_experiment_properly():
    """Properly set up MLflow experiment for Databricks"""
    try:
        # Get or create experiment
        experiment_name = f"/Users/{mlflow.get_experiment_by_name('Default').name}/mds_benchmark_{int(time.time())}"
        
        # Try to get existing experiment or create new one
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment: {experiment_name}")
        except:
            # Fallback to default experiment
            experiment_id = mlflow.set_experiment("Default")
            experiment_name = "Default"
            print("Using Default experiment")
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set: {experiment_name}")
        return experiment_name
        
    except Exception as e:
        print(f"MLflow experiment setup failed: {e}")
        print("Trying alternative setup...")
        
        # Alternative setup for Databricks
        try:
            # Use a simple experiment name
            experiment_name = "mds_benchmark"
            mlflow.set_experiment(experiment_name)
            print(f"Using simple experiment name: {experiment_name}")
            return experiment_name
        except Exception as e2:
            print(f"Alternative setup also failed: {e2}")
            print("Will use default experiment")
            return "Default"

def configure_system_metrics_environment():
    """Configure environment variables for MLflow system metrics"""
    import os
    
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_DISK_USAGE"] = "true" 
    os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_GPU_USAGE"] = "true"
    os.environ["MLFLOW_SYSTEM_METRICS_MONITOR_NETWORK_USAGE"] = "true"
    
    print("System metrics environment configured")

def enable_system_metrics_properly():
    """Enable MLflow system metrics with proper configuration"""
    try:
        configure_system_metrics_environment()
        
        import mlflow.system_metrics
        mlflow.system_metrics.enable_system_metrics_logging()
        
        print("MLflow system metrics enabled successfully")
        return True
        
    except Exception as e:
        print(f"Failed to enable system metrics: {e}")
        return False

def log_only_performance_metrics(results, remote_path, config=None):
    """Log ONLY performance metrics"""
    try:
        # Parameters
        mlflow.log_param("remote_path", remote_path)
        mlflow.log_param("local_path_type", results['test_name'])
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        
        if config:
            for key, value in config.items():
                mlflow.log_param(f"config_{key}", value)
        
        # Performance metrics ONLY
        performance_metrics = {
            "throughput_samples_per_sec": results.get('throughput'),
            "avg_batch_time_seconds": results.get('avg_batch_time'),
            "init_time_seconds": results.get('init_time'),
            "total_samples": results.get('total_samples')
        }
        
        for metric_name, value in performance_metrics.items():
            if value is not None and value >= 0:
                mlflow.log_metric(metric_name, value)
        
        print("Performance metrics logged")
                
    except Exception as e:
        print(f"MLflow logging failed: {e}")

def test_streaming_performance(remote_path, local_path, test_name, 
                             batch_size=32, num_batches=20, num_workers=4):
    """Streaming performance test"""
    
    print(f"\n{'='*50}")
    print(f"Testing: {test_name}")
    print(f"Local: {local_path}")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    print(f"{'='*50}")
    
    # Clean up local cache
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except Exception as e:
        print(f"Cache cleanup failed: {e}")
        return None
    
    # Initialize dataset
    start_time = time.time()
    try:
        dataset = StreamingDataset(
            remote=remote_path,
            local=local_path,
            batch_size=batch_size,
            shuffle=True,
            download_retry=3,
            download_timeout=60
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        return None
    
    init_time = time.time() - start_time
    print(f"Dataset initialization: {init_time:.2f}s")
    
    # Run data loading test
    batch_times = []
    total_samples = 0
    
    overall_start = time.time()
    
    try:
        dataloader_iter = iter(dataloader)
        
        for i in range(num_batches):
            batch_start = time.time()
            
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                print(f"Dataset exhausted at batch {i}")
                break
            except Exception as e:
                print(f"Error loading batch {i}: {e}")
                continue
            
            # Get batch size
            try:
                if isinstance(batch, dict):
                    batch_size_actual = len(next(iter(batch.values())))
                else:
                    batch_size_actual = len(batch)
            except:
                batch_size_actual = batch_size
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            total_samples += batch_size_actual
            
            if i % 5 == 0:
                print(f"Batch {i}: {batch_time:.3f}s")
    
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None
    
    total_loading_time = time.time() - overall_start
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    throughput = total_samples / total_loading_time if total_loading_time > 0 else 0
    
    print(f"\nResults:")
    print(f"  Batches processed: {len(batch_times)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'test_name': test_name,
        'init_time': init_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput,
        'total_samples': total_samples,
        'config': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_batches': num_batches
        }
    }

def run_benchmark_with_proper_mlflow_setup(remote_path, batch_size=16, num_batches=15):
    """Run benchmark with proper MLflow experiment handling"""
    
    print("MDS Storage Performance Comparison")
    print("="*60)
    
    # Set up experiment properly
    experiment_name = setup_mlflow_experiment_properly()
    
    results = {}
    
    # Test configurations
    configs = [
        {
            'name': 'tmp_storage',
            'local_path': '/tmp/streaming_cache_test'
        },
        {
            'name': 'local_disk0_storage',
            'local_path': '/local_disk0/tmp/streaming_cache_test'
        }
    ]
    
    for config in configs:
        print(f"\nRunning test: {config['name']}")
        
        # Create MLflow run with proper error handling
        run_name = f"{config['name']}_{int(time.time())}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                print(f"Started MLflow run: {run_name}")
                
                # Log test configuration
                mlflow.log_param("test_type", "storage_comparison")
                mlflow.log_param("storage_location", config['name'])
                mlflow.log_param("remote_path", remote_path)
                
                # Run the test
                result = test_streaming_performance(
                    remote_path=remote_path,
                    local_path=config['local_path'],
                    test_name=config['name'],
                    batch_size=batch_size,
                    num_batches=num_batches,
                    num_workers=4
                )
                
                if result:
                    results[config['name']] = result
                    log_only_performance_metrics(result, remote_path, result.get('config'))
                    print(f"Test completed: {config['name']} - {result['throughput']:.1f} samples/sec")
                else:
                    print(f"Test failed: {config['name']}")
                    
        except Exception as e:
            print(f"MLflow run failed for {config['name']}: {e}")
            print("Continuing with next test...")
        
        time.sleep(2)
    
    # Print comparison
    if len(results) >= 2:
        print(f"\nCOMPARISON RESULTS")
        print("-" * 40)
        
        for name, result in results.items():
            print(f"{name:20}: {result['throughput']:8.1f} samples/sec")
        
        if len(results) == 2:
            result_list = list(results.values())
            if result_list[1]['throughput'] > result_list[0]['throughput']:
                speedup = result_list[1]['throughput'] / result_list[0]['throughput']
                print(f"\nlocal_disk0 is {speedup:.1f}x faster than tmp")
    
    print(f"\nMLflow experiment: {experiment_name}")
    print("View results with: mlflow ui")
    
    return results

def create_sample_dataset(local_path, num_samples=50):
    """Create a small sample MDS dataset"""
    
    print(f"Creating sample dataset: {local_path}")
    
    try:
        from streaming import MDSWriter
        import numpy as np
        
        os.makedirs(local_path, exist_ok=True)
        
        columns = {
            'data': 'bytes',
            'label': 'int'
        }
        
        with MDSWriter(out=local_path, columns=columns) as writer:
            for i in range(num_samples):
                sample_data = np.random.randint(0, 255, (32,), dtype=np.uint8).tobytes()
                sample = {
                    'data': sample_data,
                    'label': i % 5
                }
                writer.write(sample)
        
        print(f"Created {num_samples} samples")
        return local_path
        
    except Exception as e:
        print(f"Failed to create sample dataset: {e}")
        return None

def main():
    """Main function with proper error handling"""
    
    print("MDS Streaming Benchmark with MLflow")
    print("="*50)
    
    try:
        # 1. Enable system metrics
        print("1. Enabling system metrics...")
        enable_system_metrics_properly()
        
        # 2. Create sample dataset
        print("\n2. Creating sample dataset...")
        sample_path = "/local_disk0/tmp/benchmark_sample"
        dataset_path = create_sample_dataset(sample_path, num_samples=30)
        
        if not dataset_path:
            print("Failed to create sample dataset")
            return
        
        # 3. Run benchmark
        print("\n3. Running benchmark...")
        results = run_benchmark_with_proper_mlflow_setup(
            remote_path=dataset_path,
            batch_size=8,
            num_batches=10
        )
        
        if results:
            print("\nBenchmark completed successfully!")
        else:
            print("Benchmark completed with some failures")
            
    except Exception as e:
        print(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()

# Simplified notebook version
def notebook_benchmark(remote_path):
    """Simplified version for notebook use"""
    
    try:
        # Setup
        enable_system_metrics_properly()
        setup_mlflow_experiment_properly()
        
        # Run benchmark
        results = run_benchmark_with_proper_mlflow_setup(
            remote_path=remote_path,
            batch_size=16,
            num_batches=15
        )
        
        return results
        
    except Exception as e:
        print(f"Notebook benchmark failed: {e}")
        return None

# if __name__ == "__main__":
#     main()

# COMMAND ----------

# REMOTE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/imagenet_tiny200_mds_20X_train_append/"

# Enable system metrics and run benchmark
enable_system_metrics_properly()
results = notebook_benchmark(REMOTE_PATH) #"/path/to/your/mds/dataset"

# COMMAND ----------

# https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/31162332f4b94991b820cf52b3fba289/runs?o=1444828305810485&searchFilter=tags.mlflow.databricks.notebook.commandID%3D%221758900523935_8456987813194478406_d38d9dfec8324b7f90e8ad8c1292b015%22&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D&compareRunsMode=CHART
