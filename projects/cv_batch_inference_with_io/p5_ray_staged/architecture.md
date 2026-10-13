# P5 — Ray Data 3-stage pipeline (GPU) (architecture)

**Compute:** Serverless GPU, 1×A10 with Ray (`databricks_ai_v5` + `ray[data]`)
**Model load:** loaded **once** in the GPU actor's `__init__` (not per batch).

```
  UC Volume     Stage 1 (CPU)              Stage 2 (GPU)                 Stage 3 (CPU)
  (JPEGs) ──► .map(Preprocess) ──────► .map_batches(GPUClassifier) ──► .map(AnnotateWrite) ──► write_parquet
              decode → PIL              ViT forward, batch=64          annotate + save JPEG    + UC Volume
              concurrency=(2,16)        concurrency=1, num_gpus=1      concurrency=(2,16)
              autoscaled CPU            single GPU actor               autoscaled CPU
```

**Why it's shaped this way:** the three stages run concurrently as a Ray Data pipeline.
CPU decode and CPU annotate+write autoscale independently and feed the single GPU actor,
so the GPU stays busy and **writes never run on the GPU actor**.

> **Anti-pattern this fixes:** doing decode + per-image writes *inside* the GPU actor
> (`concurrency=1`) pins the GPU at ~0% utilization — that earlier version was **~11×
> slower** (3,804s vs 345s).

**Trade-off:** most complex to author (Ray / AI Runtime, staged pipeline), but scales
cleanly to multi-GPU. Because wall time is dominated by the GPU inference stage, it is the
**most dataset-size-independent** pattern.

**Timing (3,925 imgs, cold, script wall):** COCO **350s** · Imagenette **350s**
(identical — GPU-compute-bound).

**Cost (us-east-1, settled billing):** **~$0.20–0.30/run** — ~0.3 DBU on the serverless GPU
Model Training SKU. Fastest tier *and* among the cheapest.
