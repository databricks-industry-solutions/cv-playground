# Databricks notebook source
# DBTITLE 1,P3 · Distributed CPU inference via Spark UDF (mapInPandas)
# MAGIC %md
# MAGIC Materializes `bench_vit@prod` to a Volume once on the driver, then a `mapInPandas` UDF
# MAGIC loads it per partition (plain `from_pretrained`) and does CPU inference + annotate +
# MAGIC distributed write. Plain serverless Spark (CPU, autoscaled). Predictions -> Delta table.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "bench_vit")
dbutils.widgets.text("model_alias", "prod")
dbutils.widgets.text("dataset", "coco")
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("image_glob", "*.jpg")
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("num_partitions", "128")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, time, json, glob
import pandas as pd
import mlflow
from datetime import datetime, timezone

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{g('model_name')}@{g('model_alias')}"
DATASET, IMAGE_DIR, IMAGE_GLOB = g("dataset"), g("image_dir"), g("image_glob")
N_IMAGES, NUM_PARTITIONS = int(g("n_images")), int(g("num_partitions"))
EXPERIMENT = g("experiment"); PATTERN = "p3_spark_udf"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p3/{DATASET}"
ANN_DIR, RESULTS_DIR = f"{OUT}/annotated", f"{OUT}/results"
MODEL_LOCAL = f"{OUT}/_model"
PRED_TABLE = f"{CATALOG}.{SCHEMA}.bench_p3_{DATASET}_predictions"
CHUNK = 8
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"{PATTERN} | dataset={DATASET} | partitions={NUM_PARTITIONS}")

# COMMAND ----------

# DBTITLE 1,Materialize bench_vit@prod to a Volume once (driver)
mlflow.set_registry_uri("databricks-uc")
comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
comp["model"].save_pretrained(MODEL_LOCAL)
(comp.get("image_processor") or comp.get("feature_extractor")).save_pretrained(MODEL_LOCAL)
print("materialized ->", MODEL_LOCAL)

SCRIPT_START = time.time()
paths = sorted(glob.glob(f"{IMAGE_DIR}/{IMAGE_GLOB}"))[:N_IMAGES]
n_img = len(paths)
df = spark.createDataFrame(pd.DataFrame({"path": paths})).repartition(NUM_PARTITIONS)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
out_schema = StructType([StructField("path", StringType()), StructField("label", StringType()),
                         StructField("confidence", DoubleType())])

def infer_partition(itr):
    import torch, os
    from PIL import Image, ImageDraw
    import torchvision.transforms as T
    from transformers import ViTForImageClassification, AutoImageProcessor
    gg = globals()
    if "_M" not in gg:
        gg["_M"] = ViTForImageClassification.from_pretrained(MODEL_LOCAL).eval()
        _p = AutoImageProcessor.from_pretrained(MODEL_LOCAL)
        gg["_TF"] = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                               T.Normalize(mean=_p.image_mean, std=_p.image_std)])
        gg["_L"] = gg["_M"].config.id2label
    model, tf, id2label = gg["_M"], gg["_TF"], gg["_L"]
    for pdf in itr:
        allp = list(pdf["path"])
        for s in range(0, len(allp), CHUNK):
            chunk = allp[s:s+CHUNK]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            x = torch.stack([tf(im) for im in imgs])
            with torch.no_grad():
                logits = model(pixel_values=x).logits
            preds = logits.argmax(-1).tolist(); confs = torch.softmax(logits, -1).max(-1).values.tolist()
            out = []
            for p, im, pr, cf in zip(chunk, imgs, preds, confs):
                lab = id2label[pr]; text = f"{lab} ({cf:.1%})"
                d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
                d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
                im.save(f"{ANN_DIR}/{os.path.basename(p)}", "JPEG", quality=90)
                out.append((p, lab, float(cf)))
            yield pd.DataFrame(out, columns=["path", "label", "confidence"])

df.mapInPandas(infer_partition, schema=out_schema).write.mode("overwrite").saveAsTable(PRED_TABLE)
n_pred = spark.table(PRED_TABLE).count()
total_wall = time.time() - SCRIPT_START
print(f"P3 done: {n_pred} preds | total_wall={total_wall:.1f}s")

# COMMAND ----------

summary = {"pattern": PATTERN, "dataset": DATASET, "compute": "serverless_spark_cpu",
           "num_partitions": NUM_PARTITIONS, "num_images": n_pred,
           "total_wall_s": round(total_wall, 3), "images_per_sec": round(n_pred/max(total_wall, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "model_uri": MODEL_URI,
                       "num_partitions": NUM_PARTITIONS, "num_images": n_pred})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
