# P6 — `ai_query` → GPU serving endpoint (architecture)

**Caller compute:** Serverless Spark CPU (only needs `Pillow` for annotate)
**Inference compute:** GPU serving endpoint `bench-vit-serving` (GPU_SMALL), backed by
`models:/{catalog}.{schema}.bench_vit_serving@prod` — a pyfunc that takes a base64 `image`
string and returns `{label, score}`. **The endpoint does inference only** (no Volume I/O).
Deploy it once in `00_setup_register_model.py`.

```
                        ┌─── Serverless Spark CPU job ───┐         ┌─ GPU endpoint ─┐
  UC Volume   Stage 1   │ binaryFile ─► base64(content)  │ Stage 2 │  bench-vit-    │
  (JPEGs) ──► ───────►  │   → base64 table               │ ──────► │  serving       │
                        │                                │ ai_query│  (GPU_SMALL,   │
                        │ Stage 3 (distributed):         │ ◄────── │  scale-to-zero)│
  UC Volume ◄────────── │ mapInPandas: decode b64 →      │ {label, └────────────────┘
  (annotated/)          │   annotate + write JPEG        │  score}
                        └────────────────────────────────┘
                        base64 ~40–55s   ai_query 104–195s   write 206–214s
```

**Why it's shaped this way:** inference is offloaded to a **reusable, governed** endpoint
callable from SQL (`ai_query`). The CPU job does base64 encoding, the `ai_query` fan-out,
and distributed annotate+writes — keeping all Volume I/O out of the endpoint. Base64 is
counted as part of end-to-end time (it's the serialization tax of crossing the endpoint
boundary). `ai_query` auto-retries 429s; validate on output completeness + correctness,
not on 429 count (retried 429s are invisible in the output table).

**Trade-off:** most infra to stand up (pyfunc wrapper + signature + endpoint), and costs
two compute resources at once (CPU job + GPU endpoint, though scale-to-zero limits idle
cost). Unique win: the endpoint is shared/versioned and usable from any SQL outside this
job. The `ai_query` phase is sensitive to per-image round-trips — it nearly doubled
(104s → 195s) on Imagenette's smaller files.

**Correctness:** endpoint vs P3 direct inference match **99.95%** (2/3,925 borderline
argmax flips from GPU fp16 vs CPU fp32).

**Timing (3,925 imgs, cold, script wall):** COCO **374s** · Imagenette **441s**.

**Cost (us-east-1, settled billing):** **~$1.4/run** — ~$1.14 CPU caller (2.54 DBU on Jobs
Serverless) + ~$0.25 GPU endpoint (Serverless Real-Time Inference). Two compute resources at
once; scale-to-zero limits idle endpoint cost between runs.

**Tuning — the P6 knob is endpoint concurrency, not the caller.** `ai_query` throughput is
bound by how many endpoint replicas serve concurrently. Bumping the endpoint `workload_size`
**Small → Medium** roughly *doubled* `ai_query` (COCO val, 5,000 imgs: **263s → 125s**, 19 →
40 img/s) and cut total wall **552s → 418s (−24%)**. This confirms `ai_query` was purely
endpoint-concurrency-bound. The bottleneck then **shifts off `ai_query`** (now 125s) onto the
write stage (~240s) + base64 (~53s), so a further bump to Large helps less — tune the stage
that's actually dominant. **Cost stays ≈flat per run:** Medium provisions ~2× GPU but
`ai_query` time roughly halves, so endpoint DBU/run is about the same, and scale-to-zero keeps
idle cost unchanged. (`num_partitions` on the caller tunes the base64 + write fan-out, not the
`ai_query` rate.)
