# Databricks notebook source
# DBTITLE 1,P4 · Auto Loader -> binary Delta table -> distributed CPU inference
# MAGIC %md
# MAGIC Stage 1: Auto Loader (`cloudFiles`, `binaryFile`, `availableNow`) ingests images into a
# MAGIC binary Delta table. Stage 2: a `mapInPandas` UDF decodes the bytes, runs `bench_vit`
# MAGIC (CPU), annotates + writes, lands predictions. Serverless Spark CPU.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "bench_vit")
dbutils.widgets.text("model_alias", "prod")
dbutils.widgets.text("dataset", "coco")
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("file_glob", "*.jpg")                    # pathGlobFilter (COCO=*.jpg, Imagenette=*.JPEG)
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("num_partitions", "128")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, time, json
import pandas as pd
import mlflow
from datetime import datetime, timezone

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{g('model_name')}@{g('model_alias')}"
DATASET, IMAGE_DIR, FILE_GLOB = g("dataset"), g("image_dir"), g("file_glob")
N_IMAGES, NUM_PARTITIONS = int(g("n_images")), int(g("num_partitions"))
EXPERIMENT = g("experiment"); PATTERN = "p4_autoloader"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p4/{DATASET}"
ANN_DIR, RESULTS_DIR, CHK = f"{OUT}/annotated", f"{OUT}/results", f"{OUT}/_autoloader"
MODEL_LOCAL = f"{OUT}/_model"
BINARY_TABLE = f"{CATALOG}.{SCHEMA}.bench_{DATASET}_binary"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.bench_p4_{DATASET}_predictions"
CHUNK = 8
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

mlflow.set_registry_uri("databricks-uc")
comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
comp["model"].save_pretrained(MODEL_LOCAL)
(comp.get("image_processor") or comp.get("feature_extractor")).save_pretrained(MODEL_LOCAL)
print("materialized ->", MODEL_LOCAL)

# COMMAND ----------

# DBTITLE 1,Stage 1 — Auto Loader ingest images as binary
SCRIPT_START = time.time()
_t = time.time()
(spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", f"{CHK}/schema")
    .option("pathGlobFilter", FILE_GLOB).option("recursiveFileLookup", "true")
    .load(IMAGE_DIR)
    .writeStream.option("checkpointLocation", f"{CHK}/chk")
    .trigger(availableNow=True).toTable(BINARY_TABLE)).awaitTermination()
ingest_s = time.time() - _t
n_binary = spark.table(BINARY_TABLE).count()
print(f"ingested {n_binary} -> {BINARY_TABLE} in {ingest_s:.1f}s")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([StructField("path", StringType()), StructField("label", StringType()),
                         StructField("confidence", DoubleType())])

def infer_partition(itr):
    import torch, os, io
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    from transformers import ViTForImageClassification, AutoImageProcessor
    gg = globals()
    if "_M4" not in gg:
        gg["_M4"] = ViTForImageClassification.from_pretrained(MODEL_LOCAL).eval()
        _p = AutoImageProcessor.from_pretrained(MODEL_LOCAL)
        gg["_TF4"] = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                T.Normalize(mean=_p.image_mean, std=_p.image_std)])
        gg["_L4"] = gg["_M4"].config.id2label
    model, tf, id2label = gg["_M4"], gg["_TF4"], gg["_L4"]
    for pdf in itr:
        rows = list(zip(pdf["path"], pdf["content"]))
        for s in range(0, len(rows), CHUNK):
            chunk = rows[s:s+CHUNK]
            imgs = [Image.open(io.BytesIO(c)).convert("RGB") for _, c in chunk]
            x = torch.stack([tf(im) for im in imgs])
            with torch.no_grad():
                logits = model(pixel_values=x).logits
            preds = logits.argmax(-1).tolist(); confs = torch.softmax(logits, -1).max(-1).values.tolist()
            out = []
            for (path, _), im, pr, cf in zip(chunk, imgs, preds, confs):
                lab = id2label[pr]; text = f"{lab} ({cf:.1%})"
                d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
                d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
                im.save(f"{ANN_DIR}/{os.path.basename(path)}", "JPEG", quality=90)
                out.append((path, lab, float(cf)))
            yield pd.DataFrame(out, columns=["path", "label", "confidence"])

_t = time.time()
src = spark.table(BINARY_TABLE).select("path", "content").orderBy("path").limit(N_IMAGES).repartition(NUM_PARTITIONS)
src.mapInPandas(infer_partition, schema=out_schema).write.mode("overwrite").saveAsTable(PRED_TABLE)
infer_s = time.time() - _t
n_pred = spark.table(PRED_TABLE).count()
total_wall = time.time() - SCRIPT_START
print(f"P4: ingest={ingest_s:.1f}s infer={infer_s:.1f}s wall={total_wall:.1f}s ({n_pred} imgs)")

# COMMAND ----------

summary = {"pattern": PATTERN, "dataset": DATASET, "compute": "serverless_spark_cpu",
           "num_partitions": NUM_PARTITIONS, "num_ingested": n_binary, "num_images": n_pred,
           "autoloader_ingest_s": round(ingest_s, 3), "inference_s": round(infer_s, 3),
           "total_wall_s": round(total_wall, 3), "images_per_sec": round(n_pred/max(total_wall, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "model_uri": MODEL_URI, "num_partitions": NUM_PARTITIONS})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
