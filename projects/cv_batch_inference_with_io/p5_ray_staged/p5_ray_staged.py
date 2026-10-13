# Databricks notebook source
# DBTITLE 1,P5 · Ray Data 3-stage (CPU decode -> GPU infer -> CPU annotate+write)
# MAGIC %md
# MAGIC Ray Data on serverless GPU, following the Databricks industry-solutions reference:
# MAGIC `.map` CPU decode (autoscaled) -> `.map_batches` GPU inference (`num_gpus=1`, model loaded
# MAGIC once) -> `.map` CPU annotate+write. Writes stay OFF the GPU actor. GPU job (`databricks_ai_v5`
# MAGIC + `ray[data]`).

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "bench_vit")
dbutils.widgets.text("model_alias", "prod")
dbutils.widgets.text("dataset", "coco")
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("image_glob", "*.jpg")
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, time, json, glob
import numpy as np, torch
from datetime import datetime, timezone
from PIL import Image, ImageDraw
import mlflow, ray

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{g('model_name')}@{g('model_alias')}"
DATASET, IMAGE_DIR, IMAGE_GLOB = g("dataset"), g("image_dir"), g("image_glob")
N_IMAGES = int(g("n_images")); EXPERIMENT = g("experiment"); PATTERN = "p5_ray_staged"
CPU_CONCURRENCY, GPU_BATCH = (2, 16), 64
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p5/{DATASET}"
ANN_DIR, RESULTS_DIR, PRED_DIR = f"{OUT}/annotated", f"{OUT}/results", f"{OUT}/predictions"
MODEL_LOCAL = f"{OUT}/_model"
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
SCRIPT_START = time.time()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
_t = time.time()
comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
comp["model"].save_pretrained(MODEL_LOCAL)
_proc = comp.get("image_processor") or comp.get("feature_extractor")
_proc.save_pretrained(MODEL_LOCAL)
MEAN, STD = [float(x) for x in _proc.image_mean], [float(x) for x in _proc.image_std]
ID2LABEL = comp["model"].config.id2label
model_read_s = time.time() - _t
print(f"materialized in {model_read_s:.1f}s")

ray.init(ignore_reinit_error=True)
paths = sorted(glob.glob(f"{IMAGE_DIR}/{IMAGE_GLOB}"))[:N_IMAGES]
n_img = len(paths)
ds = ray.data.from_items([{"path": p} for p in paths])

class Preprocess:
    def __init__(self):
        import torchvision.transforms as T
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    def __call__(self, row):
        return {"path": row["path"], "pixel_values": self.tf(Image.open(row["path"]).convert("RGB")).numpy()}

class GPUClassifier:
    def __init__(self):
        from transformers import ViTForImageClassification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(MODEL_LOCAL).to(self.device).eval()
    def __call__(self, batch):
        x = torch.from_numpy(np.asarray(batch["pixel_values"])).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                logits = self.model(pixel_values=x).logits.float()
        preds = logits.argmax(-1).cpu().numpy(); confs = torch.softmax(logits, -1).max(-1).values.cpu().numpy()
        return {"path": np.asarray(batch["path"]),
                "label": np.array([ID2LABEL[int(p)] for p in preds]), "confidence": confs.astype(np.float64)}

class AnnotateWrite:
    def __call__(self, row):
        im = Image.open(row["path"]).convert("RGB")
        text = f'{row["label"]} ({float(row["confidence"]):.1%})'
        d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
        d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
        outp = f"{ANN_DIR}/{os.path.basename(row['path'])}"; im.save(outp, "JPEG", quality=90)
        return {"path": row["path"], "label": row["label"], "confidence": float(row["confidence"]), "output_path": outp}

_t = time.time()
(ds.map(Preprocess, concurrency=CPU_CONCURRENCY, num_cpus=1)
   .map_batches(GPUClassifier, concurrency=1, num_gpus=1, batch_size=GPU_BATCH)
   .map(AnnotateWrite, concurrency=CPU_CONCURRENCY, num_cpus=1)
   .write_parquet(PRED_DIR))
pipeline_s = time.time() - _t
total_wall = time.time() - SCRIPT_START
print(f"pipeline={pipeline_s:.1f}s wall={total_wall:.1f}s ({n_img} imgs)")

# COMMAND ----------

summary = {"pattern": PATTERN, "dataset": DATASET, "compute": "serverless_gpu_ray_1xA10",
           "num_images": n_img, "model_read_s": round(model_read_s, 3),
           "pipeline_s": round(pipeline_s, 3), "total_wall_s": round(total_wall, 3),
           "images_per_sec": round(n_img/max(pipeline_s, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "model_uri": MODEL_URI, "num_images": n_img})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
