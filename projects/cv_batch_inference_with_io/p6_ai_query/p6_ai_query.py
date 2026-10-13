# Databricks notebook source
# DBTITLE 1,P6 · ai_query -> GPU serving endpoint (inference-only) + distributed CPU write
# MAGIC %md
# MAGIC End-to-end on **serverless CPU (Spark)**, inference offloaded to the **GPU endpoint**
# MAGIC `bench-vit-serving` (deployed in step 0). Three distributed stages:
# MAGIC 1. read raw images -> **base64 table**,
# MAGIC 2. `ai_query` each row -> endpoint -> `{label, score}`,
# MAGIC 3. distributed `mapInPandas` annotate + write (decodes the base64 it already has).

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("endpoint", "bench-vit-serving")
dbutils.widgets.text("dataset", "coco")
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("file_glob", "*.jpg")
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("num_partitions", "128")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, io, base64, time, json
import pandas as pd
from datetime import datetime, timezone
from pyspark.sql import functions as F
import mlflow

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
ENDPOINT, DATASET = g("endpoint"), g("dataset")
IMAGE_DIR, FILE_GLOB = g("image_dir"), g("file_glob")
N_IMAGES, NUM_PARTITIONS = int(g("n_images")), int(g("num_partitions"))
EXPERIMENT = g("experiment"); PATTERN = "p6_ai_query"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p6/{DATASET}"
ANN_DIR, RESULTS_DIR = f"{OUT}/annotated", f"{OUT}/results"
B64_TABLE = f"{CATALOG}.{SCHEMA}.bench_{DATASET}_b64"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.bench_p6_{DATASET}_predictions"
OUT_TABLE = f"{CATALOG}.{SCHEMA}.bench_p6_{DATASET}_written"
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
SCRIPT_START = time.time()

# COMMAND ----------

# DBTITLE 1,Stage 1 (CPU) — raw images -> base64 table
_t = time.time()
(spark.read.format("binaryFile").option("pathGlobFilter", FILE_GLOB).option("recursiveFileLookup", "true")
     .load(IMAGE_DIR)
     .withColumn("path", F.regexp_replace("path", "^dbfs:", ""))
     .selectExpr("path", "base64(content) AS image_b64")
     .orderBy("path").limit(N_IMAGES)
     .write.mode("overwrite").saveAsTable(B64_TABLE))
n_b64 = spark.table(B64_TABLE).count()
b64_s = time.time() - _t
print(f"base64 table {B64_TABLE}: {n_b64} rows in {b64_s:.1f}s")

# COMMAND ----------

# DBTITLE 1,Stage 2 (CPU) — ai_query -> GPU endpoint -> predictions
_t = time.time()
spark.sql(f"""
  CREATE OR REPLACE TABLE {PRED_TABLE} AS
  SELECT path, ai_query('{ENDPOINT}', named_struct('image', image_b64),
                        returnType => 'STRUCT<label STRING, score DOUBLE>') AS prediction
  FROM {B64_TABLE}
""")
n_pred = spark.table(PRED_TABLE).count()
aiquery_s = time.time() - _t
print(f"ai_query -> {n_pred} preds in {aiquery_s:.1f}s ({n_pred/max(aiquery_s,1e-9):.1f} img/s)")

# COMMAND ----------

# DBTITLE 1,Stage 3 (CPU, distributed) — decode base64, annotate + write
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([StructField("path", StringType()), StructField("label", StringType()),
                         StructField("score", DoubleType()), StructField("output_path", StringType())])

def annotate_write(itr):
    import io, os, base64
    from PIL import Image, ImageDraw
    for pdf in itr:
        rows = []
        for _, r in pdf.iterrows():
            im = Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
            text = f'{r["label"]} ({float(r["score"]):.1%})'
            d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
            d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
            outp = f"{ANN_DIR}/{os.path.basename(r['path'])}"; im.save(outp, "JPEG", quality=90)
            rows.append((r["path"], r["label"], float(r["score"]), outp))
        yield pd.DataFrame(rows, columns=["path", "label", "score", "output_path"])

_t = time.time()
src = (spark.table(PRED_TABLE).selectExpr("path", "prediction.label AS label", "prediction.score AS score")
       .join(spark.table(B64_TABLE).select("path", "image_b64"), "path").repartition(NUM_PARTITIONS))
src.mapInPandas(annotate_write, schema=out_schema).write.mode("overwrite").saveAsTable(OUT_TABLE)
n_written = spark.table(OUT_TABLE).count()
write_s = time.time() - _t
total_wall = time.time() - SCRIPT_START
print(f"annotate+wrote {n_written} in {write_s:.1f}s | total_wall={total_wall:.1f}s")

# COMMAND ----------

summary = {"pattern": PATTERN, "dataset": DATASET, "endpoint": ENDPOINT, "endpoint_workload": "GPU_SMALL",
           "compute_caller": "serverless_spark_cpu", "num_partitions": NUM_PARTITIONS, "num_images": n_written,
           "img_to_b64_s": round(b64_s, 3), "aiquery_s": round(aiquery_s, 3),
           "annotate_write_s": round(write_s, 3), "total_wall_s": round(total_wall, 3),
           "aiquery_images_per_sec": round(n_pred/max(aiquery_s, 1e-9), 2),
           "end_to_end_images_per_sec": round(n_written/max(total_wall, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "endpoint": ENDPOINT, "num_partitions": NUM_PARTITIONS})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
