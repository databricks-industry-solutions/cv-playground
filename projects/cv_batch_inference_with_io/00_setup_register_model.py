# Databricks notebook source
# DBTITLE 1,00 · Setup — register the clean benchmark model(s) + create volumes
# MAGIC %md
# MAGIC One-time setup for the clean P1–P6 benchmark. Registers **two MLflow/UC models from
# MAGIC the same weights** and creates the output volumes. Everything is MLflow-native.
# MAGIC
# MAGIC - **`{catalog}.{schema}.bench_vit`** — transformers flavor, alias **`@prod`** → used by
# MAGIC   the batch patterns P1–P5 via `models:/…bench_vit@prod`.
# MAGIC - **`{catalog}.{schema}.bench_vit_serving`** — pyfunc (base64 image → `{label, score}`),
# MAGIC   for P6's serving endpoint. Same weights, serving contract.
# MAGIC
# MAGIC Point `source_model` at your own registered classifier to benchmark it instead.
# MAGIC Labels are auto-derived from the model's `id2label` (falls back to class index).

# COMMAND ----------

# DBTITLE 1,Params (job params / widgets — no code edits needed)
dbutils.widgets.text("catalog", "main", "catalog")
dbutils.widgets.text("schema", "cv", "schema")
dbutils.widgets.text("source_model", "main.cv.vit_imagenette_soup", "source UC model (weights) — point at your own")
dbutils.widgets.text("base_processor_dir", "/Volumes/main/cv/vit_models/vit_imagenette/base_model", "image processor dir — point at your own")
dbutils.widgets.text("bench_model", "bench_vit", "clean model name")
dbutils.widgets.text("serving_model", "bench_vit_serving", "clean serving model name")
dbutils.widgets.dropdown("deploy_endpoint", "true", ["true", "false"], "deploy GPU serving endpoint (for P6)")
dbutils.widgets.text("endpoint_name", "bench-vit-serving", "serving endpoint name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SOURCE_MODEL = dbutils.widgets.get("source_model")
PROCESSOR_DIR = dbutils.widgets.get("base_processor_dir")
BENCH_MODEL = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('bench_model')}"
SERVING_MODEL = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('serving_model')}"

# Optional: make the clean model self-describing with real Imagenette labels.
CLASS_ORDER = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
               "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMAGENETTE_LABELS = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}

import os, io, base64
import mlflow, torch
import pandas as pd
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

# DBTITLE 1,Load source weights, stamp readable labels, re-log as clean bench_vit (transformers)
from transformers import AutoImageProcessor
LOCAL = "/tmp/bench_vit_weights"
_ver = max(int(v.version) for v in client.search_model_versions(f"name = '{SOURCE_MODEL}'"))
comp = mlflow.transformers.load_model(f"models:/{SOURCE_MODEL}/{_ver}", return_type="components")
model = comp["model"]
# stamp id2label so labels are auto-derivable downstream (self-describing model)
model.config.id2label = {i: IMAGENETTE_LABELS[c] for i, c in enumerate(CLASS_ORDER)}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}
model.save_pretrained(LOCAL)
proc = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
proc.save_pretrained(LOCAL)

from transformers import pipeline as hf_pipeline, ViTForImageClassification
reloaded = ViTForImageClassification.from_pretrained(LOCAL)
pipe = hf_pipeline("image-classification", model=reloaded, image_processor=proc)
with mlflow.start_run(run_name="register_bench_vit") as run:
    info = mlflow.transformers.log_model(transformers_model=pipe, artifact_path="model",
                                         registered_model_name=BENCH_MODEL)
bench_ver = info.registered_model_version if hasattr(info, "registered_model_version") else \
    max(int(v.version) for v in client.search_model_versions(f"name = '{BENCH_MODEL}'"))
client.set_registered_model_alias(BENCH_MODEL, "prod", bench_ver)
print(f"registered {BENCH_MODEL} v{bench_ver} + alias @prod")

# COMMAND ----------

# DBTITLE 1,Inference-only pyfunc wrapper (base64 image -> label/score) for serving
class ViTInferenceOnly(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch, torchvision.transforms as T
        from transformers import ViTForImageClassification, AutoImageProcessor
        md = context.artifacts["model_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(md).to(self.device).eval()
        self.id2label = self.model.config.id2label
        p = AutoImageProcessor.from_pretrained(md)
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(mean=p.image_mean, std=p.image_std)])

    def predict(self, context, model_input, params=None):
        import io, base64, torch
        from PIL import Image
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in model_input["image"]]
        x = torch.stack([self.tf(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=(self.device.type == "cuda")):
                logits = self.model(pixel_values=x).logits.float()
        preds = logits.argmax(-1).tolist()
        confs = torch.softmax(logits, -1).max(-1).values.tolist()
        return pd.DataFrame({"label": [self.id2label[p] for p in preds],
                             "score": [float(c) for c in confs]})

# COMMAND ----------

# DBTITLE 1,Log + register bench_vit_serving (pyfunc), tag lineage to bench_vit
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
sig = ModelSignature(inputs=Schema([ColSpec("string", "image")]),
                     outputs=Schema([ColSpec("string", "label"), ColSpec("double", "score")]))
_smpl = sorted(__import__("glob").glob(f"{PROCESSOR_DIR}/../data/val/*/*"))[:1]
example = pd.DataFrame({"image": [base64.b64encode(open(_smpl[0], "rb").read()).decode()]}) if _smpl else None

with mlflow.start_run(run_name="register_bench_vit_serving") as run:
    mlflow.pyfunc.log_model(artifact_path="model",
        python_model=ViTInferenceOnly(), artifacts={"model_dir": LOCAL},
        pip_requirements=["torch==2.5.1", "torchvision==0.20.1", "transformers==4.49.0", "Pillow"],
        input_example=example, signature=sig)
    srun = run.info.run_id
sreg = mlflow.register_model(model_uri=f"runs:/{srun}/model", name=SERVING_MODEL)
client.set_registered_model_alias(SERVING_MODEL, "prod", sreg.version)
client.set_model_version_tag(SERVING_MODEL, sreg.version, "source_model", f"{BENCH_MODEL}@prod v{bench_ver}")
print(f"registered {SERVING_MODEL} v{sreg.version} + alias @prod (wraps {BENCH_MODEL} v{bench_ver})")

# COMMAND ----------

# DBTITLE 1,Create clean output volumes bench_out_p1..p6
for i in range(1, 7):
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.bench_out_p{i}")
print("created volumes bench_out_p1..p6")

# COMMAND ----------

# DBTITLE 1,(Optional) deploy the GPU serving endpoint for P6 — shared, dataset-agnostic infra
if dbutils.widgets.get("deploy_endpoint") == "true":
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (ServedEntityInput, EndpointCoreConfigInput,
                                                 ServingModelWorkloadType)
    from datetime import timedelta
    ENDPOINT = dbutils.widgets.get("endpoint_name")
    cfg = EndpointCoreConfigInput(name=ENDPOINT, served_entities=[ServedEntityInput(
        entity_name=SERVING_MODEL, entity_version=sreg.version,
        workload_type=ServingModelWorkloadType.GPU_SMALL,   # ENUM, not the string
        workload_size="Small", scale_to_zero_enabled=True)])
    w = WorkspaceClient()
    exists = next((e for e in w.serving_endpoints.list() if e.name == ENDPOINT), None)
    if exists is None:
        print(f"Creating GPU endpoint {ENDPOINT} (blocks until ready, ~10-20 min)...")
        w.serving_endpoints.create_and_wait(name=ENDPOINT, config=cfg, timeout=timedelta(minutes=45))
    else:
        print(f"Updating endpoint {ENDPOINT}...")
        w.serving_endpoints.update_config_and_wait(served_entities=cfg.served_entities, name=ENDPOINT,
                                                    timeout=timedelta(minutes=45))
    print("endpoint ready:", ENDPOINT)
else:
    print("skipped endpoint deploy (deploy_endpoint=false)")

print("SETUP DONE:", BENCH_MODEL, "@prod |", SERVING_MODEL, "@prod")
