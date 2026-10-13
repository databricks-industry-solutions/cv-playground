# P4 — Auto Loader binary → UDF (distributed CPU) (architecture)

**Compute:** Serverless Spark, CPU, 128 partitions (`torch`/`torchvision`/`transformers`
on executors)
**Model load:** same as P3 — materialize to Volume, `from_pretrained` per executor.
**Knobs:** `file_glob` (`pathGlobFilter`, e.g. `*.jpg` / `*.JPEG`), `num_partitions`.

```
  UC Volume     Stage 1 — Auto Loader ingest            Stage 2 — distributed inference
  (JPEGs) ──► cloudFiles(binaryFile, availableNow) ──► Delta ──► mapInPandas UDF ──► UC Vol
                recursiveFileLookup=true               binary    (from_pretrained,     (annotated/)
                pathGlobFilter=file_glob               table      infer, annotate+write) + Delta
              └──────── ingest ~68–97s ────────┘      └──────── infer+write ~368–424s ────────┘
```

**Why it's shaped this way:** P3 plus an incremental **binary-ingest** stage. Decoupling
ingestion (`cloudFiles`) from inference enables reprocessing / streaming and a reusable
binary table.

**Trade-off:** pays an extra ingest phase, and that phase is sensitive to file **layout**
— many small nested files (Imagenette) cost more to list than COCO's flat dir (ingest
68s → 97s). Best when ingestion is ongoing or you want a reusable binary table.

**Timing (3,925 imgs, cold, script wall):** COCO **438s** (ingest 68s) · Imagenette
**522s** (ingest 97s).

**Cost (us-east-1, settled billing):** **~$1.76/run** — 3.91 DBU on Jobs Serverless Compute
($0.45/DBU). Highest of the six: P3's CPU inference cost plus the extra binary-ingest stage.

**Tuning — raise `num_partitions` at scale, leave `CHUNK` at 8.** `num_partitions` 128 → 256
was ~flat at 5,000 imgs (its payoff is at **118K**, where it drove **4× scaling**). Raising
the inference micro-batch `CHUNK` 8 → 16 **hurt** — inference jumped **384s → 528s** (COCO val
5,000), likely GC/memory pressure from decoding 16 images at once in the binary-table UDF.
Keep `CHUNK=8`.
