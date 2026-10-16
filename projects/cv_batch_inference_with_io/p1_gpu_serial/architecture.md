# P1 — Single model, serial writes (architecture)

**Compute:** Serverless GPU, 1×A10 (`databricks_ai_v5` base environment)
**Model load:** `models:/{catalog}.{schema}.bench_vit@prod` via `mlflow.transformers.load_model`

```
                     ┌──────────────────  single GPU driver  ──────────────────┐
  UC Volume          │                                                          │
  (JPEGs)  ──read──► │  DataLoader (batch=64) ──► ViT forward ──► for each img: │
                     │                                            annotate + save│──► UC Volume
                     │                                           (SERIAL loop)   │    (annotated/)
                     └──────────────────────────────────────────────────────────┘
                              inference ~75s              write ~635s  ◄── bottleneck
```

**Why it's shaped this way:** the simplest possible baseline — one process, one forward
pass, one write loop. Everything is on the driver GPU node.

**Trade-off:** serial writes never exceed the S3 per-prefix request rate, so it is the
**most consistent** pattern run-to-run — but writes are 86% of wall time, making it the
**slowest**. Use it as the correctness reference.

**Timing (3,925 imgs, cold, script wall):** COCO **740s** · Imagenette **629s**
(model_read ~25s · inference ~43–75s · **write 560–635s**).

**Cost (us-east-1, settled billing):** **~$0.39/run** — 0.61 DBU on the serverless GPU
Model Training SKU (longer than P2 because serial writes hold the GPU longer).
