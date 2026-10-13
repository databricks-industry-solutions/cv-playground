# Databricks Batch Inference with I/O — Pattern Comparison (P1–P6)

> **▶ [View the slide deck (rendered)](https://raw.githack.com/databricks-industry-solutions/cv-playground/main/projects/cv_batch_inference_with_io/batch_inference_deck.html)** — a visual walkthrough of the benchmark, the six patterns, and when to use each. (The [`batch_inference_deck.html`](batch_inference_deck.html) file renders as a live deck at that link; GitHub shows the raw source.)

Six batch-inference patterns for running a ViT image classifier over images on
**Databricks serverless compute**, benchmarked head-to-head. The goal is to compare
*compute strategies* (single-node GPU vs. distributed CPU vs. serving endpoint) and *I/O
strategies* (serial vs. parallel vs. distributed writes) on the same workload.

Everything is **MLflow-native and user-iterable**: all six load the *same* clean UC model
and are parameterized via **job params / widgets**, so you can **swap in your own model +
images and re-run** with no code edits. Each pattern lives in its own folder with the
notebook `.py`, a `job.json` you can `databricks jobs create` from, and an
`architecture.md` diagram.

- **Model:** a **ViT-Base/16-224** image classifier (~86M params, 346 MB) — a 10-class
  Imagenette model built as a weight-averaged "soup" of 10 fine-tuned checkpoints (one load,
  one forward pass — not a runtime ensemble). Registered as `…cv.bench_vit@prod`
  (transformers) — batch patterns load `models:/…bench_vit@prod`; `…cv.bench_vit_serving@prod`
  (pyfunc, base64→`{label,score}`) backs the P6 endpoint. Labels auto-derive from `id2label`.
- **Datasets:** benchmarked on **COCO val2017** (avg **≈159.5 KB**/image, range 8.7–680 KB)
  and **Imagenette val** (avg **≈7.8 KB**/image, range 1.8–22.2 KB) — **~20× smaller** — both
  the first **3,925 images**, so the only variables are image size + file layout. (Model is
  Imagenette-trained, so on COCO the labels are meaningless — **these measure
  timing/throughput, not accuracy.**)
- **Setup:** see [Setup](#setup) below — run [`00_setup_register_model.py`](00_setup_register_model.py)
  once against **your own** model + catalog, then create each pattern's job from its `job.json`.

## Setup

### Prerequisites

- **A Unity Catalog catalog + schema you can create volumes in.** Every notebook defaults to
  `catalog=main`, `schema=cv` — change the `catalog`/`schema` widgets (or the `base_parameters`
  in each `job.json`) to a catalog you own.
- **A registered image-classification model to benchmark.** `00_setup_register_model.py` does
  *not* train one — it takes an existing **transformers**-flavor UC model plus its image-processor
  directory and re-registers it as the two clean benchmark models. The widget defaults
  (`main.cv.vit_imagenette_soup` and `/Volumes/main/cv/vit_models/vit_imagenette/base_model`)
  are the artifacts from the original benchmark and **will not exist in your workspace** — point
  `source_model` and `base_processor_dir` at your own classifier. Labels are auto-derived from the
  model's `id2label`.
- **Images in a UC Volume.** Set `image_dir` + `image_glob` (flat dirs use `*.jpg`, nested use `*/*`).
- **Serverless GPU** for P1/P2/P5 — requires the **"Serverless workspace base environment support
  in Jobs"** workspace preview (see [Prerequisites & gotchas](#prerequisites--gotchas)).
- **Your workspace username.** The `job.json` specs ship `<your-username>` placeholders in
  `notebook_path` and `experiment` — replace both before creating the jobs.

### Steps

1. Import this folder into your Databricks workspace (e.g. clone the repo via Git folders).
2. Run [`00_setup_register_model.py`](00_setup_register_model.py) once, with the widgets pointed
   at your model/catalog. It registers `bench_vit@prod` (transformers) + `bench_vit_serving@prod`
   (pyfunc), creates the `bench_out_p1..p6` output volumes, and — if `deploy_endpoint=true` —
   deploys the GPU serving endpoint that P6 calls.
3. Create and run each pattern's job from its `job.json` (see [Running these](#running-these)).

## The six patterns

| Dir | Pattern | Compute | I/O strategy |
|-----|---------|---------|--------------|
| [`p1_gpu_serial/`](p1_gpu_serial/architecture.md) | Single model, one forward pass | Serverless GPU (1×A10) | **Serial** annotate + write |
| [`p2_gpu_parallel/`](p2_gpu_parallel/architecture.md) | Single model, one forward pass | Serverless GPU (1×A10) | **32-thread** parallel writes |
| [`p3_spark_udf/`](p3_spark_udf/architecture.md) | `mapInPandas` Spark UDF | Serverless Spark (CPU, 128 partitions) | **Distributed** writes across executors |
| [`p4_autoloader/`](p4_autoloader/architecture.md) | Auto Loader → binary table → UDF | Serverless Spark (CPU, 128 partitions) | Distributed ingest + writes |
| [`p5_ray_staged/`](p5_ray_staged/architecture.md) | Ray Data 3-stage (CPU→GPU→CPU) | Serverless GPU (Ray, 1×A10) | Autoscaled CPU writes, off the GPU actor |
| [`p6_ai_query/`](p6_ai_query/architecture.md) | `ai_query` → GPU serving endpoint | GPU endpoint + Serverless Spark CPU | CPU base64 + distributed writes |

Each folder also has the runnable notebook (e.g. [`p1_gpu_serial/p1_gpu_serial.py`](p1_gpu_serial/p1_gpu_serial.py))
and its job spec ([`p1_gpu_serial/job.json`](p1_gpu_serial/job.json)).

## Results — COCO val2017 (3,925 imgs, ≈159 KB avg)

Cold single runs, script wall. **P2 is reported at its cold number** so the comparison is
apples-to-apples with the other single-run patterns (its warm best case is ~155s).

"Script wall" = end-to-end in-script (read + inference + write). "Billed job time" adds the
fixed serverless startup + environment install (~70–110s) that's on the invoice but not in
the script timer — compare *patterns* with script wall, plan *cost/wall-clock* with billed.

| Rank | Pattern | Time breakdown | **Script wall** | **Billed job time** |
|:----:|---------|----------------|:---------------:|:-------------------:|
| 1 | **P5** — Ray staged (GPU) | pipeline 308s | **350s** | ~441s |
| 2 | **P2** — GPU + 32-thread writes | inference ~50s · write 288s (cold) | **358s** | ~447s |
| 3 | **P6** — ai_query → GPU endpoint | ai_query 104s · base64 55s · write 214s | **374s** | ~405s |
| 4 | **P3** — Spark UDF (CPU) | *(distributed read+infer+write)* | **387s** | ~457s |
| 5 | **P4** — Auto Loader + UDF (CPU) | ingest 68s · infer/write | **438s** | ~513s |
| 6 | **P1** — GPU + serial writes | inference ~75s · write 635s (86% of wall) | **740s** | ~848s |

## Key findings

1. **The winner is not always the same — benchmark on *your* workload.** The ranking flips on
   **two** axes: image size / file layout (at fixed count) *and* dataset scale. These are
   **reference templates and a benchmark harness**, not a one-size-fits-all recommendation.

   **By image size & layout (both datasets 3,925 imgs, cold, billed job time).** The fast tier
   (P2/P5/P6) lands within ~40s on both, but the exact winner shuffles: P6 is quickest on COCO
   (405s — lightest startup) and P5 on Imagenette (431s). P2 *warm* (~155s wall) would beat
   either, but it's the least consistent.

   | Pattern | Imagenette job time | COCO job time | What changed |
   |---------|:-------------------:|:-------------:|--------------|
   | P1 — GPU serial | ~740s&nbsp;† | 848s | smaller files → less write time, still slowest |
   | P2 — GPU parallel | **441s** | **447s** | ~flat; fastest tier on both (cold) |
   | P3 — Spark UDF (CPU) | 486s | 457s | +6% on tiny nested files |
   | P4 — Auto Loader (CPU) | 593s | 513s | **+16%** — nested-dir listing + many tiny files cost more at ingest |
   | P5 — Ray staged (GPU) | **431s** | **441s** | ~flat — GPU-compute-bound |
   | P6 — ai_query (endpoint) | 471s | 405s | **+16%** — `ai_query` phase grows on smaller files (104s → 195s) |

   † P1's Imagenette run clocked **1,622s** elapsed, but that includes a ~15-min queue wait
   (six GPU jobs launched at once); queue time isn't billed — its compute was ~740s (629s
   script wall + startup).

   **By scale — the ranking *inverts* (COCO, 5,000 → 118,287 imgs, script wall).** At ~5K the
   single-node GPU patterns win or tie; at 118K the **distributed-CPU** patterns (P3/P4) win
   decisively, because they fan writes across an autoscaling worker pool while P1/P2/P5 are
   bounded by one node. P3 finishes 118K in **38 min vs P1's 5h 44m (9×) and P5's 2h 16m
   (3.6×)**. Throughput (img/s) shows why:

   | Pattern | 5,000 imgs | 118,287 imgs | Scaling |
   |---------|:----------:|:------------:|---------|
   | P3 — Spark UDF (CPU) | 13.4 | **51.4** (38 min) | **3.8×** — autoscales workers |
   | P4 — Auto Loader (CPU) | 11.3 | **45.4** (43 min) | **4.0×** — autoscales workers |
   | P6 — ai_query (endpoint) | 9.1 | 26.6 (74 min) | 2.9× — endpoint + distributed writes |
   | P5 — Ray staged (GPU) | 13.3 | 14.7 (2h 16m) | ~flat — one GPU actor |
   | P2 — GPU parallel | 11.3 | 12.3 (2h 40m) | ~flat — one node's write threads |
   | P1 — GPU serial | 6.1 | 5.7 (5h 44m) | ~flat — serial, single node |

   The takeaway: **below ~5K images pick a single-node GPU pattern (P5/P2); at ~100K+ go
   distributed CPU (P3/P4).** The write bottleneck that dominates P1/P2/P5 is bounded by one
   node; Spark's `mapInPandas` scales it horizontally, so the more images you process, the
   better P3/P4 look.

2. **This workload is I/O-bound on writes, not compute.** GPU inference is only ~50–75s;
   P1's serial annotated-JPEG write to the UC Volume is 560–635s = ~86% of its runtime.
3. **Parallelizing writes is the biggest single win** on a warm prefix (P2 warm ~155s), but
   the parallel write burst is exactly what trips S3 per-prefix throttling on a **cold**
   prefix (P2 cold 358s) — so P2 is the **least consistent** (2.3× cold/warm swing) while
   serial P1 is the **most consistent**.
4. **P5 (Ray) is the most dataset-*size*-stable at fixed count — but doesn't scale
   horizontally.** At 3,925 imgs it's ~350s script wall on both COCO and Imagenette
   (≈441s/431s billed), because time is dominated by the single GPU inference stage, not file
   size. But that single GPU actor is also a ceiling: its throughput stays ~flat (13.3 → 14.7
   img/s) from 5K to 118K (see finding #1), so unlike the distributed-CPU patterns it can't
   trade more workers for more speed as the dataset grows.
5. **File *layout and count* matter as much as total bytes.** The patterns that lean on
   directory listing / per-file endpoint calls (P4 ingest, P6 `ai_query`) are the ones that
   slow down on Imagenette's many small nested files — even though the total data is smaller.
6. **The served model equals direct inference.** P6 (endpoint) vs P3 (direct) match 99.95%
   (2/3,925 borderline argmax flips from GPU fp16 vs CPU fp32).

## Cost Reference (AWS us-east-1 / N. Virginia)

Measured from `system.billing.usage × system.billing.list_prices` (Enterprise, settled
billing) for one 3,925-image COCO run per pattern. GPU compute bills under the serverless
**Model Training** SKU ($0.65/DBU), CPU under **Jobs Serverless Compute** ($0.45/DBU), and
the P6 endpoint under **Serverless Real-Time Inference** ($0.07/DBU). Region rates vary.

| Pattern | Billing SKU | DBUs / run | **Cost / run** | Notes |
|---------|-------------|:----------:|:--------------:|-------|
| **P2** GPU parallel | Model Training (GPU) | 0.39 | **$0.25** | cheapest — short GPU hold |
| **P5** Ray staged | Model Training (GPU) | ~0.30 | **~$0.20–0.30** | GPU, dataset-stable |
| **P1** GPU serial | Model Training (GPU) | 0.61 | **$0.39** | slow serial writes → longer GPU hold |
| **P6** ai_query | Jobs Serverless (CPU) + endpoint | 2.54 + endpoint | **~$1.14 + ~$0.25** | two compute resources (scale-to-zero limits idle endpoint cost) |
| **P3** Spark UDF | Jobs Serverless (CPU) | 3.53 | **$1.59** | autoscaled CPU workers |
| **P4** Auto Loader | Jobs Serverless (CPU) | 3.91 | **$1.76** | + binary-ingest stage |

**GPU wins on both speed and cost** for this write-bound workload — the single-node GPU
patterns (P1/P2/P5) run **~4–7× cheaper** than the distributed-CPU patterns (P3/P4/P6),
because CPU inference is slower *and* fans the work across several billed workers. P2 and P5
are both the fastest tier *and* the cheapest (~$0.25/run).

## What determines the best pattern

Image size is only one axis. Other factors that move the ranking on *your* workload:

- **Do you write images at all?** These examples annotate + write a JPEG per image, which
  is the dominant cost. If you only need predictions to a Delta table (no image output),
  the write bottleneck disappears and GPU inference dominates — a completely different
  ranking.
- **Image size & count** — large files favor parallel-GPU writes (P2); many tiny nested
  files raise ingest/round-trip cost (P4, P6) and can trip thread-pool write throttling.
- **Parallelism knobs** — P2's `write_workers` and P3/P4/P6's `num_partitions` are tunable
  and materially change results; the values here are starting points.
- **Ensemble orchestration** — a single weight-averaged *soup* model = one load + one
  forward pass (what these scripts use). A *logit ensemble* over N checkpoints = N loads +
  N passes, which shifts the balance heavily toward `model_read`/inference cost.
- **Where the model lives / how it loads** — each path has different load cost and
  executor-distribution implications:
  - **Registered in UC** (`models:/<catalog>.<schema>.<model>@<alias>`) — clean versioning;
    GPU patterns (P1/P2/P5) load it straight from MLflow on the driver.
  - **Materialized to a UC Volume** — executors load with plain `from_pretrained(path)`, no
    MLflow/UC auth on workers (how P3/P4 distribute).
  - **Behind a serving endpoint** — inference is governed, versioned and SQL-callable, at
    the cost of a base64 round-trip and a second compute resource (P6).

## Running these

Each pattern is a Databricks notebook-source `.py`. Upload the folder to a workspace and
create the job from its `job.json` (fix `notebook_path` to your workspace path first):

```bash
databricks jobs create --json @p1_gpu_serial/job.json
databricks jobs run-now <job_id> --notebook-params '{"dataset":"imagenette","image_glob":"*/*", ...}'
```

Three compute profiles are used (all captured in the per-pattern `job.json`):

- **Serverless GPU (P1, P2, P5)** — task-level `compute.hardware_accelerator = GPU_1xA10` +
  `environments[].spec.base_environment = databricks_ai_v5`. P5 adds `dependencies: ["ray[data]"]`.
- **Serverless Spark CPU (P3, P4)** — `environments[].spec.dependencies =
  ["torch","torchvision","transformers"]`, which propagate to the `mapInPandas` workers; no
  compute block.
- **Serverless Spark CPU caller (P6)** — only needs `["Pillow"]`; inference runs on the GPU
  serving endpoint via `ai_query`.

### Prerequisites & gotchas

- **Serverless GPU jobs (P1/P2/P5) require a Beta preview:** enable **"Serverless workspace
  base environment support in Jobs"** in the workspace preview settings. Without it, the
  `base_environment: databricks_ai_v5` + `compute.hardware_accelerator` config is rejected /
  silently downgraded to CPU and the GPU task fails. GPU serverless supports only
  `notebook_task` / `python_wheel_task` (not `spark_python_task`), and `for_each` is
  unsupported on GPU jobs.
- **Output dirs under `/Volumes/{catalog}/{schema}/` must be registered UC Volumes.**
  `os.makedirs()` on a non-Volume path fails with `OSError [Errno 95] Operation not
  supported`. `00_setup_register_model.py` creates the `bench_out_p1..p6` volumes for you.
- **P6 endpoint shape matters, not GPU vs CPU.** A raw transformers model served directly
  fails at model load; the working endpoint is a `pyfunc` wrapper with a `ModelSignature`
  (base64 `image` string in → `{label, score}` out), registered as `bench_vit_serving` and
  served on GPU_SMALL. The SDK needs the `ServingModelWorkloadType.GPU_SMALL` **enum** (not
  the string). `ai_query` auto-retries 429s — validate on output completeness + correctness,
  not 429 count.

## Methodology & caveats

- **Cold single runs.** All numbers are single cold runs for a consistent methodology.
  Write-bound patterns (P1–P4, P6) are **cold/warm sensitive**: writing thousands of
  annotated JPEGs to a UC Volume is object-store (S3) I/O, so a fresh prefix throttles
  (`503 SlowDown` → retries) while a warmed one flies (P2: 288s cold → 66s warm write).
- **Script wall vs. billed job time.** Script wall is the in-script timer (read + inference +
  write); billed job time adds a fixed ~70–110s of serverless startup + env install, which
  varies run-to-run with how warm the serverless pool is.
- **Queue time is elapsed but *not* billed.** When several jobs contend for capacity, a run
  can wait in a queue before compute is allocated — that shows up in raw run duration but
  isn't charged (see the P1 Imagenette footnote: 1,622s elapsed, ~740s compute).
- Built and benchmarked on Databricks serverless (AWS us-east-1).
