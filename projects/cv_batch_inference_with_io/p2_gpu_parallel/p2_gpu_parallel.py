# Databricks notebook source
# DBTITLE 1,P2 · Single model, PARALLEL-thread writes (Serverless GPU)
# MAGIC %md
# MAGIC Same as P1 (load `bench_vit@prod`, one GPU forward pass) but the annotate+write phase
# MAGIC is parallelized with a thread pool (`write_workers`). Serverless GPU.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "bench_vit")
dbutils.widgets.text("model_alias", "prod")
dbutils.widgets.text("dataset", "coco")
dbutils.widgets.text("image_dir", "/Volumes/main/cv/coco2017/val2017")
dbutils.widgets.text("image_glob", "*.jpg")
dbutils.widgets.text("n_images", "3925")
dbutils.widgets.text("write_workers", "32")
dbutils.widgets.text("experiment", "/Users/<your-username>/bench_vit_inference")

import os, time, json, glob
import torch
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import mlflow

g = dbutils.widgets.get
CATALOG, SCHEMA = g("catalog"), g("schema")
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{g('model_name')}@{g('model_alias')}"
DATASET, IMAGE_DIR, IMAGE_GLOB = g("dataset"), g("image_dir"), g("image_glob")
N_IMAGES, WRITE_WORKERS = int(g("n_images")), int(g("write_workers"))
EXPERIMENT = g("experiment"); PATTERN = "p2_gpu_parallel"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/bench_out_p2/{DATASET}"
ANN_DIR, RESULTS_DIR = f"{OUT}/annotated", f"{OUT}/results"
os.makedirs(ANN_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{PATTERN} | device={device} | dataset={DATASET} | write_workers={WRITE_WORKERS}")
SCRIPT_START = time.time()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
_t = time.time()
comp = mlflow.transformers.load_model(MODEL_URI, return_type="components")
model = comp["model"].to(device).eval()
proc = comp.get("image_processor") or comp.get("feature_extractor")
id2label = model.config.id2label
model_read_s = time.time() - _t

_t = time.time()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(mean=proc.image_mean, std=proc.image_std)])
paths = sorted(glob.glob(f"{IMAGE_DIR}/{IMAGE_GLOB}"))[:N_IMAGES]

class DS(Dataset):
    def __init__(self, p): self.p = p
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return tf(Image.open(self.p[i]).convert("RGB")), i

loader = DataLoader(DS(paths), batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
n_img = len(paths); dataset_prep_s = time.time() - _t

# COMMAND ----------

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

# COMMAND ----------

order, pl, cl = idx_t.tolist(), preds.tolist(), conf.tolist()
jobs = [(order[k], pl[k], cl[k]) for k in range(n_img)]

def annotate_one(job):
    i, p, c = job
    im = Image.open(paths[i]).convert("RGB")
    text = f"{id2label[p]} ({c:.1%})"
    d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), text); tw, th = bb[2]-bb[0], bb[3]-bb[1]
    d.rectangle([(2, 2), (tw+10, th+10)], fill="black"); d.text((6, 4), text, fill="white")
    im.save(f"{ANN_DIR}/{i:06d}.jpg", "JPEG", quality=90)

_t = time.time()
with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as ex:
    list(ex.map(annotate_one, jobs))
write_s = time.time() - _t
total_wall = time.time() - SCRIPT_START
print(f"annotate_write={write_s:.1f}s ({WRITE_WORKERS}w) | total_wall={total_wall:.1f}s")

# COMMAND ----------

summary = {"pattern": PATTERN, "dataset": DATASET, "num_images": n_img, "write_workers": WRITE_WORKERS,
           "model_read_s": round(model_read_s, 3), "dataset_prep_s": round(dataset_prep_s, 3),
           "inference_s": round(inference_s, 3), "annotate_write_s": round(write_s, 3),
           "total_wall_s": round(total_wall, 3), "images_per_sec": round(n_img/max(total_wall, 1e-9), 2)}
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=f"{PATTERN}_{DATASET}"):
    mlflow.log_params({"pattern": PATTERN, "dataset": DATASET, "model_uri": MODEL_URI,
                       "num_images": n_img, "write_workers": WRITE_WORKERS})
    mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
now = datetime.now(timezone.utc).isoformat()
with open(f"{RESULTS_DIR}/results_{now[:19].replace(':', '-')}.json", "w") as f:
    json.dump([summary], f, indent=2)
print(json.dumps(summary, indent=2)); print("Done.")
