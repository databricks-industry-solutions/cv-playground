# P2 — Single model, 32-thread parallel writes (architecture)

**Compute:** Serverless GPU, 1×A10 (`databricks_ai_v5` base environment)
**Model load:** `models:/{catalog}.{schema}.bench_vit@prod`
**Knob:** `write_workers` (default 32) — thread-pool size over the write phase.

```
                     ┌──────────────────  single GPU driver  ──────────────────┐
  UC Volume          │                          ┌─ thread ─► annotate+save ─┐   │
  (JPEGs)  ──read──► │  DataLoader ─► ViT ──►    ├─ thread ─► annotate+save ─┤   │──► UC Volume
                     │  (batch=64)  forward      │   ... 32 workers ...      │   │    (annotated/)
                     │                          └─ thread ─► annotate+save ─┘   │
                     └──────────────────────────────────────────────────────────┘
                        inference ~50–75s        ThreadPoolExecutor(write_workers)
```

**Why it's shaped this way:** identical to P1 but the write loop is fanned across a thread
pool, so many annotated JPEGs are created concurrently instead of one at a time.

**Trade-off:** on a **warm** prefix it's the **fastest** pattern (~155s COCO). But the
concurrent write burst is exactly what trips S3 per-prefix throttling on a **cold** prefix
(→ 358s), so it's the **least consistent** (2.3× cold/warm swing). Tune `write_workers`;
warm the target prefix if you can.

**Timing (3,925 imgs, cold, script wall):** COCO **358s** (155s warm) · Imagenette **345s**
(inference ~34–50s · **write 283–288s cold** / 66s warm).

**Cost (us-east-1, settled billing):** **~$0.25/run** — 0.39 DBU on the serverless GPU
Model Training SKU. Cheapest pattern (short GPU hold).

**Tuning — `write_workers` is already at the single-node ceiling.** Raising it 32 → 48 → 64
barely moved the write stage (**354s → 362s → 353s**, COCO val 5,000): the write is saturated
on **single-node S3 bandwidth**, not thread count, so more threads buy nothing. (A worse
*total* at w48 was just run-to-run noise in the cold model-load/inference, not the knob.)
Leave it at 32 — to write faster you need *more nodes* (a distributed pattern, P3/P4), not
more threads.
