# Databricks notebook source
# DBTITLE 1,P1 · Single model, SERIAL writes (Serverless GPU)
# MAGIC %md
# MAGIC Loads `bench_vit@prod` once on the GPU, one batched forward pass, then a **serial**
# MAGIC annotate+write of one JPEG per image. Dataset-agnostic: point the widgets at your own
# MAGIC images/model. Phase timings are logged to a shared **MLflow experiment** + a results JSON.

# COMMAND ----------

# DBTITLE 1,Params (job params / widgets)
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "bench_vit")
dbutils.widgets.text("model_alias", "prod")
dbutils.widgets.text("dataset", "coco")                       # tag for output naming
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("image_glob", "*.jpg")                   # COCO flat="*.jpg"; Imagenette="*/*"
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, time, json, glob
import torch
from datetime import datetime, timezone
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import mlflow

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{g('model_name')}@{g('model_alias')}"
DATASET, IMAGE_DIR, IMAGE_GLOB = g("dataset"), g("image_dir"), g("image_glob")
N_IMAGES = int(g("n_images")); EXPERIMENT = g("experiment")
PATTERN = "p1_gpu_serial"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p1/{DATASET}"
ANN_DIR, RESULTS_DIR = f"{OUT}/annotated", f"{OUT}/results"
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{PATTERN} | device={device} | dataset={DATASET} | model={MODEL_URI}")
SCRIPT_START = time.time()

# COMMAND ----------

# DBTITLE 1,Load bench_vit@prod (MLflow) + build dataset
mlflow.set_registry_uri("databricks-uc")
_t = time.time()
comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
model = comp["model"].to(device).eval()
proc = comp.get("image_processor") or comp.get("feature_extractor")
id2label = model.config.id2label                              # labels auto-derived from the model
model_read_s = time.time() - _t

_t = time.time()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(mean=proc.image_mean, std=proc.image_std)])
paths = sorted(glob.glob(f"{IMAGE_DIR}/{IMAGE_GLOB}"))[:N_IMAGES]

class DS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return tf(Image.open(self.paths[i]).convert("RGB")), i

loader = DataLoader(DS(paths), batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
n_img = len(paths); dataset_prep_s = time.time() - _t
print(f"images={n_img} | model_read={model_read_s:.1f}s prep={dataset_prep_s:.1f}s")

# COMMAND ----------

# DBTITLE 1,One batched GPU forward pass
_t = time.time()
all_logits, all_idx = [], []
with torch.no_grad():
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(pixel_values=x).logits
        all_logits.append(logits.float().cpu()); all_idx.append(idx)
logits_t = torch.cat(all_logits); idx_t = torch.cat(all_idx)
preds = logits_t.argmax(-1); conf = torch.softmax(logits_t, -1).max(-1).values
inference_s = time.time() - _t
print(f"inference={inference_s:.1f}s")

# COMMAND ----------

# DBTITLE 1,SERIAL annotate + write
_t = time.time()
order, pl, cl = idx_t.tolist(), preds.tolist(), conf.tolist()
for k in range(n_img):
    i = order[k]; im = Image.open(paths[i]).convert("RGB")
    text = f"{id2label[pl[k]]} ({cl[k]:.1%})"
    d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
    d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
    im.save(f"{ANN_DIR}/{i:06d}.jpg", "JPEG", quality=90)
write_s = time.time() - _t
total_wall = time.time() - SCRIPT_START
print(f"annotate_write={write_s:.1f}s | total_wall={total_wall:.1f}s")

# COMMAND ----------

# DBTITLE 1,Log timings to MLflow + results JSON
summary = {"pattern": PATTERN, "dataset": DATASET, "num_images": n_img,
           "model_read_s": round(model_read_s, 3), "dataset_prep_s": round(dataset_prep_s, 3),
           "inference_s": round(inference_s, 3), "annotate_write_s": round(write_s, 3),
           "total_wall_s": round(total_wall, 3), "images_per_sec": round(n_img/max(total_wall, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "model_uri": MODEL_URI, "num_images": n_img})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
