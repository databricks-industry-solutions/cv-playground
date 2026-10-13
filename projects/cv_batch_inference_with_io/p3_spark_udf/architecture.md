# P3 — Spark UDF `mapInPandas` (distributed CPU) (architecture)

**Compute:** Serverless Spark, CPU, 128 partitions (`torch`/`torchvision`/`transformers`
propagated to executors)
**Model load:** `bench_vit` materialized once to a UC Volume (`_model`), then every
executor does `from_pretrained(volume_path)` — no MLflow/UC auth needed on workers.
**Knob:** `num_partitions` (default 128).

```
  UC Volume        ┌──────────────  Spark executors (autoscaled CPU)  ──────────────┐
  (JPEGs) ─binary─► │  partition 1: from_pretrained ─► infer(chunk=8) ─► annotate+write│─► UC Vol
                    │  partition 2: from_pretrained ─► infer(chunk=8) ─► annotate+write│─► + Delta
       repartition  │  ...                                                            │    table
       (128) ─────► │  partition N: from_pretrained ─► infer(chunk=8) ─► annotate+write│
                    └────────────────────────────────────────────────────────────────┘
                       inference + writes spread across many prefixes/tasks
```

**Why it's shaped this way:** no GPU required. `mapInPandas` fans both inference *and*
writes across autoscaled executors, so writes hit many S3 prefixes at once (less
single-node FUSE contention than P1/P2).

**Trade-off:** costs more (autoscaled CPU DBUs, and CPU inference is slower per image) and
swings ±25% with how many workers spin up. Best when you have no GPU or want pure
horizontal scale.

**Timing (3,925 imgs, cold, script wall):** COCO **387s** · Imagenette **417s**.

**Cost (us-east-1, settled billing):** **~$1.59/run** — 3.53 DBU on Jobs Serverless Compute
($0.45/DBU). ~4× a GPU pattern: CPU inference is slower and fans across several billed
workers.

**Tuning — `num_partitions` helps at scale, `CHUNK` doesn't.** At this scale P3 is
write-bound, not inference-bound, so raising the inference micro-batch `CHUNK` (8 → 16 → 32)
does nothing useful (it was flat-to-slightly-worse). `num_partitions` 128 → 256 gave a small
win (**−3.5%**, COCO val 5,000) — but this lever pays off far more at **118K**, where it drove
the **3.8× throughput scaling**; at 5K there aren't enough images to feed a bigger pool.
