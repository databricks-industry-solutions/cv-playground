# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v4"
# environment_version = "4"
# ///
# MAGIC %md
# MAGIC # Finetune SD2 Inpainting for Overlay Text Removal (v0.1.5)
# MAGIC
# MAGIC **Model:** `sd2-community/stable-diffusion-2-inpainting` (865M U-Net params)
# MAGIC
# MAGIC **Strategy:** DDP + gradient checkpointing + 8-bit Adam
# MAGIC - Memory: ~10.8 GB per GPU on A10G, ~14 GB on H100
# MAGIC - Effective batch size auto-configured from GPU type preset (see GPU Presets below)
# MAGIC
# MAGIC **Data:** MDS shards from `convert_otr_to_mds.py` (74K non-augmented training samples)
# MAGIC
# MAGIC **Smoke test mode:** (10 steps) for quick environment validation
# MAGIC
# MAGIC **Training approach:**
# MAGIC - Freeze VAE + text encoder, train only U-Net
# MAGIC - Use diffusers' native noise scheduler (same schedule as pretrained)
# MAGIC - Condition on a fixed prompt — the model learns from mask + image
# MAGIC - Log loss + sample images to MLflow every N steps
# MAGIC
# MAGIC **Compute:** Serverless GPU via [AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/index/) using [`@distributed`](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training/).
# MAGIC Classic GPU clusters via `TorchDistributor` are covered in the companion `_classic` notebook.
# MAGIC
# MAGIC **MVP target:** 1K–5K steps to prove pipeline works; quality iteration later.
# MAGIC
# MAGIC ### GPU Presets
# MAGIC
# MAGIC Select GPU type via the `Compute: GPU Type` widget. Batch size, gradient accumulation, and precision are set automatically from the preset. Override `Compute: Num GPUs` to change GPU count.
# MAGIC
# MAGIC | GPU Type | Batch/GPU | Grad Accum | Default GPUs | Precision | Eff. Batch | Notes |
# MAGIC |---|---|---|---|---|---|---|
# MAGIC | **A10** | 1 | 8 | 1 | fp16 | 8 | Single GPU default. Set Num GPUs > 1 for multi-node (requires remote=true). |
# MAGIC | **H100** | 4 | 2 | 8 | fp16 | 64 | 8 GPUs, single-node NVLink. Set 16 for 2-node. |
# MAGIC
# MAGIC To customize batch/accum/precision, edit the `GPU_PRESETS` dict in the Configuration cell.
# MAGIC
# MAGIC > **Note on precision:** All GPU types use fp16. bf16 (commonly recommended for H100 in LLM/transformer training) causes numerical instability in this diffusion pipeline — the SD2 base model was trained in fp16, and the VAE scaling factor / noise scheduler alpha calculations are sensitive to bf16's lower mantissa precision.
# MAGIC
# MAGIC ### Compute: Remote widget
# MAGIC
# MAGIC The `Compute: Remote` widget controls whether GPUs are provisioned remotely via `@distributed(remote=True)`:
# MAGIC
# MAGIC | Remote | Behavior | Companion Run | Cost | When to use |
# MAGIC |---|---|---|---|---|
# MAGIC | **true** | Provisions GPUs on separate remote node(s) | Yes — per-node GPU logs + system metrics | Driver + remote nodes | Cross-accelerator (A10 → H100), multi-A10, or when you need companion run observability |
# MAGIC | **false** | Uses co-located GPUs on attached compute | No | Attached compute only | Same GPU type attached, want NVLink + no double billing |
# MAGIC
# MAGIC Smart default: `true` when GPU type differs from attached (cross-accelerator required), `false` when same type.
# MAGIC
# MAGIC > **`remote=True` is Private Preview.** Multi-node distributed training support is evolving — a CLI-based offering is expected. The `@distributed` API may change; set remote=false or fall back to v0.1.4 if needed. If you find `remote=True` useful, please provide feedback to Databricks — user demand helps prioritize official support.
# MAGIC
# MAGIC > **Note on billing with `remote=True`:** Remote provisioning creates separate compute node(s) even when the same GPU type is attached. The attached compute serves as the driver (orchestration + inference). For A10 multi-GPU, remote=true is the only multi-GPU option — there is no co-located multi-A10 on serverless.
# MAGIC
# MAGIC > **Note on per-rank logging:** `@distributed` surfaces rank 0's stdout in the cell output. This training function uses a `_TeeStream` + `torch.distributed.gather_object()` pattern to capture all ranks' stdout and save them as MLflow artifacts (`run_info/rank_0_log.txt` through `rank_N_log.txt`).
# MAGIC
# MAGIC **Key features:**
# MAGIC - **GPU type drives presets**: select A10 or H100, batch/accum/precision auto-configure
# MAGIC - **Cross-accelerator launching** (remote=true): train on H100 from an A10-attached notebook
# MAGIC - **Companion MLflow run** (remote=true): per-node GPU logs + system metrics linked from training run tags
# MAGIC - **Checkpoints on UC Volumes**: accessible from all nodes
# MAGIC - **Best checkpoint selection**: saves `best/` on val_loss improvement; inference loads automatically
# MAGIC - **Per-rank logs**: saved as MLflow artifacts since notebook UI only shows rank 0
# MAGIC - See **Appendix** for topology details, scaling paths, and cost tradeoffs

# COMMAND ----------

# DBTITLE 1,Install Required Libraries [and Restart Python Session]
# MAGIC %pip install -q transformers diffusers accelerate huggingface_hub~=0.36 safetensors bitsandbytes

# COMMAND ----------

# DBTITLE 1,Install (2/2): Streaming, Pillow, datasets + restart
# MAGIC %pip install -q mosaicml-streaming~=0.13.0 pillow~=12.1 datasets~=4.8 psutil nvidia-ml-py
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Clear All Widgets to Reset Configuration
dbutils.widgets.removeAll()

# COMMAND ----------

# DBTITLE 1,Configuration (widgets + derived variables)
import os, re, torch

# ━━━ GPU detection & presets ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detect GPU type from hardware — determines batch/accum/precision/gpu count defaults
_gpu_name = torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else "NONE"
if "H100" in _gpu_name:
    _detected = "H100"
elif "A10" in _gpu_name:
    _detected = "A10"
else:
    _detected = "A10"  # safe fallback

# Recommended defaults per GPU type
# NOTE: All GPU types use fp16 (not bf16). While bf16 is commonly recommended
# for H100 in LLM/transformer training, diffusion models have different numerical
# patterns — VAE scaling factors, noise scheduler alpha calculations through
# float64 intermediates, and the base SD2 model was originally trained in fp16.
# bf16's lower mantissa precision causes consistent NaN at step 2 in this pipeline.
#
# GPU Type    | Architecture                      | Batch | Accum | Precision | Default GPUs
# ------------|-----------------------------------|-------|-------|-----------|-------------
# A10 (24GB)  | 1 GPU or multi-node (remote)      |   1   |   8   |   fp16    |   1
# H100 (80GB) | 1-8 GPUs NVLink, 16+ multi-node   |   4   |   2   |   fp16    |   8
GPU_PRESETS = {
    "H100": {"batch": "4", "accum": "2", "precision": "fp16", "gpus": "8"},
    "A10":  {"batch": "1", "accum": "8", "precision": "fp16", "gpus": "1"},
}

# ━━━ Widget definitions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("NOTE: After changing any widget, rerun this cell to apply.\n")

# -- Compute (prefixed so they group together in widget bar) --
# gpu_type: target GPU for training. With remote=true, can differ from attached compute.
# num_gpus: defaults from preset, overridable (e.g. A10 x 4 for multi-node, H100 x 16 for 2-node).
# remote: provisions GPUs remotely via @distributed(remote=True) (Private Preview).
#   - true:  enables companion MLflow run (per-node GPU logs), cross-accelerator launching
#   - false: uses co-located GPUs directly (NVLink on H100), no companion run, no double billing
#   Smart default: true when cross-accelerator (e.g. A10 → H100), false when same type.
dbutils.widgets.dropdown("compute_gpu_type", _detected, ["A10", "H100"], "Compute: GPU Type")
dbutils.widgets.dropdown("compute_num_gpus", str(GPU_PRESETS[_detected]["gpus"]), ["1", "2", "4", "8", "16"], "Compute: Num GPUs")

DETECTED_GPU = dbutils.widgets.get("compute_gpu_type")
_preset = GPU_PRESETS[DETECTED_GPU]
_num_gpus_str = dbutils.widgets.get("compute_num_gpus")
NUM_DISTRIBUTED_GPUS = int(_num_gpus_str)

# Smart default for remote: true when cross-accelerator, false when same type
_default_remote = "true" if DETECTED_GPU != _detected else "false"
dbutils.widgets.dropdown("compute_remote", _default_remote, ["true", "false"], "Compute: Remote (Private Preview)")
USE_REMOTE = dbutils.widgets.get("compute_remote") == "true"

if DETECTED_GPU != _detected and not USE_REMOTE:
    print(f"WARNING: gpu_type target '{DETECTED_GPU}' differs from attached '{_detected}' — remote=true is required for cross-accelerator")
    USE_REMOTE = True
elif DETECTED_GPU != _detected:
    print(f"NOTE: gpu_type target '{DETECTED_GPU}' differs from attached '{_detected}' — will provision remotely")
if USE_REMOTE:
    print(f"  remote=true: companion MLflow run enabled (Private Preview)")

# -- Paths --
dbutils.widgets.text("catalog", "my_catalog", "Catalog")
dbutils.widgets.text("schema", "my_schema", "Schema")
dbutils.widgets.text("volume", "my_volume", "Volume")
dbutils.widgets.text("mds_folder", "OverlayTextRemoval_MDS", "MDS Folder")

# -- Run control --
dbutils.widgets.dropdown("smoke_test", "false", ["true", "false"], "Smoke Test (10 steps)")
dbutils.widgets.text("num_training_steps", "500", "Training Steps")
dbutils.widgets.text("resume_from", "", "Resume From (blank=fresh, 'latest', or 'checkpoint-N')")
dbutils.widgets.text("resume_run_id", "", "Resume Run ID (blank=current run, or paste previous run ID)")

# -- Hyperparameters (overridable) --
dbutils.widgets.text("learning_rate", "1e-5", "Learning Rate")
dbutils.widgets.text("warmup_steps", "50", "Warmup Steps")
dbutils.widgets.text("save_every", "50", "Save Checkpoint Every N Steps")
dbutils.widgets.text("log_every", "50", "Log Loss Every N Steps")
dbutils.widgets.text("sample_every", "50", "Generate Samples Every N Steps")
dbutils.widgets.text("val_every", "50", "Validate Every N Steps")

# ━━━ Read widgets ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
SMOKE_TEST = dbutils.widgets.get("smoke_test") == "true"
NUM_TRAINING_STEPS = int(dbutils.widgets.get("num_training_steps"))
_resume_val = dbutils.widgets.get("resume_from").strip()
RESUME_FROM_CHECKPOINT = _resume_val if _resume_val else None
_resume_run_val = dbutils.widgets.get("resume_run_id").strip()
RESUME_RUN_ID = _resume_run_val if _resume_run_val else None

# Hyperparameters
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
WARMUP_STEPS = int(dbutils.widgets.get("warmup_steps"))
SAVE_EVERY = int(dbutils.widgets.get("save_every"))
LOG_EVERY = int(dbutils.widgets.get("log_every"))
SAMPLE_EVERY = int(dbutils.widgets.get("sample_every"))
VAL_EVERY = int(dbutils.widgets.get("val_every"))

# ━━━ GPU preset-driven config (from gpu_type — num_gpus from widget) ━━━
# Batch, accum, precision derived from GPU_PRESETS. Num GPUs from widget.
PER_GPU_BATCH_SIZE = int(_preset["batch"])
GRADIENT_ACCUMULATION_STEPS = int(_preset["accum"])
MIXED_PRECISION = _preset["precision"]
DDP_ERROR_HANDLING = "fallback"  # "fallback" to single GPU on DDP failure, or "fail"

# ━━━ Derived paths ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOLUME_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DIFFUSION_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/diffusion"
MDS_PATH = f"{VOLUME_BASE}/{dbutils.widgets.get('mds_folder')}"
CHECKPOINT_BASE = f"{DIFFUSION_VOL}/checkpoints"
BASE_PIPELINE_PATH = f"{DIFFUSION_VOL}/_base_pipeline"
CHECKPOINT_PATH = CHECKPOINT_BASE
MODEL_NAME = "sd2-community/stable-diffusion-2-inpainting"

# Cross-run resume
if RESUME_RUN_ID:
    from pathlib import Path as _Path
    _candidates = sorted(_Path(CHECKPOINT_BASE).glob(f"*_{RESUME_RUN_ID}")) if _Path(CHECKPOINT_BASE).exists() else []
    RESUME_CHECKPOINT_PATH = str(_candidates[-1]) if _candidates else None
    if not RESUME_CHECKPOINT_PATH:
        print(f"WARNING: No checkpoint directory found matching run ID {RESUME_RUN_ID} under {CHECKPOINT_BASE}")
else:
    RESUME_CHECKPOINT_PATH = None

# ━━━ Fixed config ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE_RESOLUTION = 512
PROMPT = "remove text from image, clean background"

# ━━━ Smoke test overrides ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if SMOKE_TEST:
    NUM_TRAINING_STEPS = 10
    WARMUP_STEPS = 2
    SAVE_EVERY = 10
    LOG_EVERY = 5
    SAMPLE_EVERY = 10
    VAL_EVERY = 10
    if RESUME_FROM_CHECKPOINT:
        print("WARNING: SMOKE TEST: ignoring resume_from / resume_run_id (fresh start)")
        RESUME_FROM_CHECKPOINT = None
        RESUME_RUN_ID = None
        RESUME_CHECKPOINT_PATH = None

# ━━━ HF cache ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HF_CACHE = "/tmp/hf_cache"
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_eff_batch = PER_GPU_BATCH_SIZE * NUM_DISTRIBUTED_GPUS * GRADIENT_ACCUMULATION_STEPS
if SMOKE_TEST:
    print("WARNING: SMOKE TEST MODE: 10 steps, minimal logging")
print(f"\n{'='*60}")
print(f"  GPU Type:     {DETECTED_GPU} (hw: {_gpu_name})")
print(f"  Precision:    {MIXED_PRECISION}  (auto from gpu_type)")
print(f"  GPUs:         {NUM_DISTRIBUTED_GPUS}  (preset: {_preset['gpus']})")
print(f"  Batch/GPU:    {PER_GPU_BATCH_SIZE}  (preset: {_preset['batch']})")
print(f"  Grad accum:   {GRADIENT_ACCUMULATION_STEPS}  (preset: {_preset['accum']})")
print(f"  Eff. batch:   {_eff_batch}  ({PER_GPU_BATCH_SIZE} x {NUM_DISTRIBUTED_GPUS} GPUs x {GRADIENT_ACCUMULATION_STEPS} accum)")
print(f"  Steps:        {NUM_TRAINING_STEPS}  (lr={LEARNING_RATE})")
print(f"{'='*60}")
print(f"MDS:          {MDS_PATH}")
print(f"Base model:   {BASE_PIPELINE_PATH}")
print(f"Checkpoints:  {CHECKPOINT_BASE}/<YYYYMMDD_HHMMSS>_<run_id>/")
print(f"Logging:      every {LOG_EVERY} steps | ckpt every {SAVE_EVERY} | val every {VAL_EVERY} | samples every {SAMPLE_EVERY}")
print(f"Resume:       {RESUME_FROM_CHECKPOINT or 'fresh start'}" + (f" (from run {RESUME_RUN_ID})" if RESUME_RUN_ID else ""))
if RESUME_FROM_CHECKPOINT == "latest" and not RESUME_RUN_ID:
    print(f"  Note: 'latest' searches the current run's checkpoint dir only -- set resume_run_id for cross-run resume")

# COMMAND ----------

# DBTITLE 1,MLflow Setup (before training — avoids dual experiment in @distributed)
import mlflow

# ━━━ Auto-detect current user ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Works on both serverless and classic compute — no hardcoded email.
_current_user = spark.sql("SELECT current_user()").first()[0]

# Widget for experiment name (just the suffix — full path built from user email)
dbutils.widgets.text("mlflow_experiment_name", "diffusion_inpainting_poc", "MLflow Experiment Name")
_experiment_name = dbutils.widgets.get("mlflow_experiment_name").strip()

# Full experiment path: /Users/<email>/<experiment_name>
MLFLOW_EXPERIMENT_PATH = f"/Users/{_current_user}/{_experiment_name}"

# ━━━ MLflow experiment setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# We do NOT set MLFLOW_EXPERIMENT_NAME as an env var because @distributed workers
# inherit env vars, and the runtime auto-creates a rogue MLflow run when it sees
# that var. The launch cell clears all MLFLOW_* env vars as a safety net.
# Training function calls mlflow.set_experiment() explicitly on rank 0 only.
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
experiment_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH).experiment_id

print(f"User:       {_current_user}")
print(f"Experiment: {MLFLOW_EXPERIMENT_PATH}")
print(f"ID:         {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify MDS data is available

# COMMAND ----------

# DBTITLE 1,Verify Dataset Shards and Index Files in MDS Root
from pathlib import Path

mds_root = Path(MDS_PATH)
print(f"MDS root: {mds_root}")
print("=" * 55)

for split in ["train", "OTR_easy", "OTR_hard"]:
    split_dir = mds_root / split
    if split_dir.exists():
        shards = list(split_dir.glob("*.mds"))
        has_index = (split_dir / "index.json").exists()
        print(f"  {split}: {len(shards)} shards, index={has_index}")
    else:
        print(f"  {split}: NOT FOUND -- run convert_otr_to_mds.py first")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define the training function
# MAGIC ### Mixed-Precision Dtype Strategy (`fp16`)
# MAGIC | Component | Loaded dtype | Compute dtype | Notes |
# MAGIC |---|---|---|---|
# MAGIC | **VAE** | `float16` | `float16` | Frozen, inference-only |
# MAGIC | **Text Encoder** | `float16` | `float16` | Frozen, inference-only |
# MAGIC | **U-Net** | checkpoint default | `float16` via Accelerator | Trainable; `gradient_checkpointing` enabled |
# MAGIC | **U-Net inputs** | `float16` (from VAE) | `float16` | Handled by `torch.autocast` |
# MAGIC | **Loss** | — | `float32` | Both `noise_pred` and `noise` cast to `.float()` for stability |
# MAGIC | **Optimizer** | — | `float32` master weights | 8-bit AdamW via Accelerator |
# MAGIC ### LR Scheduler + Gradient Accumulation
# MAGIC The cosine LR scheduler's `num_warmup_steps` and `num_training_steps` are multiplied by `GRADIENT_ACCUMULATION_STEPS`. This is required because `Accelerator` with multi-GPU DDP wraps the scheduler and internally divides steps by the accumulation factor. Without this multiplication, the learning rate schedule completes prematurely on multi-GPU runs (LR drops to 0 early). On single-GPU, the extra multiplication extends warmup — harmless.

# COMMAND ----------

# DBTITLE 1,Helper: find latest checkpoint
def _find_latest_checkpoint(ckpt_root):
    """Return (path, step) for the latest valid checkpoint directory."""
    from pathlib import Path
    import re
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        return None, 0
    valid = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)", d.name)
        if not m:
            continue
        has_unet = (d / "unet" / "config.json").exists()
        has_resume_state = (d / "training_metadata.json").exists()
        if has_unet and has_resume_state:
            valid.append((d, int(m.group(1))))
    if not valid:
        return None, 0
    return max(valid, key=lambda x: x[1])

# COMMAND ----------

# DBTITLE 1,Training function (attaches to driver's MLflow run)
def train_sd2_inpainting():
    """Training function — owns the entire MLflow lifecycle.

    Runs on each GPU worker via @distributed / direct call.
    Rank 0 creates a single MLflow run, logs params + metrics + samples.
    If RESUME_FROM_CHECKPOINT is set, loads U-Net weights and optimizer state
    from the specified checkpoint and continues training.
    """
    import os
    import json
    import time
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from PIL import Image
    import numpy as np
    import sys
    import io as _io

    # ── Stdout capture for node log artifacts ─────────────────
    # Tee stdout to both original stream and a StringIO buffer.
    # Flushes every write for real-time @distributed stdout forwarding
    # (workers use full buffering by default, hiding logs until return).
    class _TeeStream:
        def __init__(self, original):
            self._original = original
            self._buffer = _io.StringIO()
        def write(self, msg):
            self._original.write(msg)
            self._original.flush()
            self._buffer.write(msg)
            return len(msg)
        def flush(self):
            self._original.flush()
        def get_log(self):
            return self._buffer.getvalue()
        def __getattr__(self, name):
            return getattr(self._original, name)

    _original_stdout = sys.stdout
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee

    # ── Mixed precision dtype (fp16 for all GPU types — see GPU_PRESETS comment) ──
    _mp_dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16

    # ── Environment setup ─────────────────────────────────────
    # Ensure remote workers can reach MLflow tracking server.
    # Plain strings injected via _clean_globals by the launch cell —
    # set as env vars before any MLflow import so remote workers
    # (A10 multi-node, H100 multi-node) can always connect.
    if _MLFLOW_TRACKING_URI:
        os.environ["MLFLOW_TRACKING_URI"] = _MLFLOW_TRACKING_URI
    if _MLFLOW_TRACKING_TOKEN:
        os.environ["MLFLOW_TRACKING_TOKEN"] = _MLFLOW_TRACKING_TOKEN

    for key in ["MLFLOW_EXPERIMENT_ID", "MLFLOW_RUN_ID", "HF_HUB_ENABLE_HF_TRANSFER"]:
        os.environ.pop(key, None)
    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(HF_CACHE, exist_ok=True)

    # ── Distributed setup ────────────────────────────────────
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        step_scheduler_with_optimizer=False,  # we step scheduler manually
    )

    set_seed(42)
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Accelerator initialized: {accelerator.num_processes} processes")
        print(f"Device: {device}")
        print(f"Mixed precision: {MIXED_PRECISION} (dtype: {_mp_dtype})")

    # ── Resolve resume checkpoint ────────────────────────────
    _ckpt_search_path = RESUME_CHECKPOINT_PATH or CHECKPOINT_PATH
    resume_ckpt_path = None
    resume_step = 0

    if RESUME_FROM_CHECKPOINT:
        if RESUME_FROM_CHECKPOINT == "latest":
            resume_ckpt_path, resume_step = _find_latest_checkpoint(_ckpt_search_path)
            if resume_ckpt_path and is_main:
                print(f"\nAuto-discovered latest checkpoint: {resume_ckpt_path.name} (step {resume_step})")
        else:
            candidate = Path(_ckpt_search_path) / RESUME_FROM_CHECKPOINT
            if (candidate / "unet").exists() and (candidate / "training_metadata.json").exists():
                resume_ckpt_path = candidate
                m = __import__("re").match(r"checkpoint-(\d+)", candidate.name)
                resume_step = int(m.group(1)) if m else 0
                if is_main:
                    print(f"\nResuming from: {resume_ckpt_path} (step {resume_step})")
            elif is_main:
                print(f"\nWARNING: Checkpoint not found at {candidate}, starting fresh")

        if resume_step >= NUM_TRAINING_STEPS and is_main:
            print(f"Checkpoint step ({resume_step}) >= NUM_TRAINING_STEPS ({NUM_TRAINING_STEPS}). Nothing to do.")
            return

    # ── Local SSD caching (topology-agnostic) ───────────────
    # Copy base pipeline from UC Volumes to local NVMe SSD for faster
    # model loading. Runs once per physical node, then all local GPUs
    # read from the shared /tmp cache instead of FUSE.
    #   Single-node (H100×8):  one copy, all 8 GPUs share /tmp
    #   Multi-node  (H100×16): each node's local rank 0 copies independently
    #   A10 remote  (A10×8):  each node copies for its 1 GPU
    import shutil as _shutil
    _local_model_cache = "/tmp/sd2_base_pipeline"
    if Path(MODEL_NAME).is_dir():  # UC Volumes / local path → worth caching
        if accelerator.local_process_index == 0 and not Path(_local_model_cache).exists():
            _t_copy = time.time()
            if is_main:
                print(f"\nCaching base pipeline to local SSD...")
            _shutil.copytree(MODEL_NAME, _local_model_cache)
            _cache_gb = sum(f.stat().st_size for f in Path(_local_model_cache).rglob("*") if f.is_file()) / 1e9
            print(f"  [local_rank 0] Copied {_cache_gb:.1f} GB to {_local_model_cache} ({time.time() - _t_copy:.1f}s)")
        elif accelerator.local_process_index == 0:
            print(f"  [local_rank 0] SSD cache exists: {_local_model_cache}")
        accelerator.wait_for_everyone()
        globals()['MODEL_NAME'] = _local_model_cache
        if is_main:
            print(f"  All ranks using local SSD: {_local_model_cache}")

    # ── Load model ─────────────────────────────────────────
    from diffusers import (
        StableDiffusionInpaintPipeline,
        DDPMScheduler,
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    def _log(msg):
        print(msg, flush=True)

    _from_local = Path(MODEL_NAME).is_dir()
    _source_label = "UC Volumes" if _from_local else "HF Hub"
    _extra_kwargs = {} if _from_local else {"cache_dir": HF_CACHE, "token": os.environ.get("HF_TOKEN")}

    if is_main:
        _log(f"\nLoading model from {_source_label}: {MODEL_NAME}")

    if is_main:
        _log(f"  [1/5] Tokenizer ({_source_label})...")
    t_load = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", **_extra_kwargs,
    )
    if is_main:
        _log(f"  [1/5] Tokenizer ready ({time.time() - t_load:.1f}s)")

    if is_main:
        _log(f"  [2/5] Text encoder ~500 MB ({_source_label})...")
    t_load = time.time()
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder",
        torch_dtype=_mp_dtype, **_extra_kwargs,
    )
    if is_main:
        _log(f"  [2/5] Text encoder ready ({time.time() - t_load:.1f}s)")

    if is_main:
        _log(f"  [3/5] VAE ~350 MB ({_source_label})...")
    t_load = time.time()
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME, subfolder="vae",
        torch_dtype=_mp_dtype, **_extra_kwargs,
    )
    if is_main:
        _log(f"  [3/5] VAE ready ({time.time() - t_load:.1f}s)")

    if is_main:
        _log(f"  [4/5] Scheduler ({_source_label})...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_NAME, subfolder="scheduler", **_extra_kwargs,
    )
    if is_main:
        _log("  [4/5] Scheduler ready")

    if is_main:
        if resume_ckpt_path:
            _log(f"  [5/5] U-Net from checkpoint: {resume_ckpt_path.name} (~3.5 GB via FUSE)...")
        else:
            _log(f"  [5/5] U-Net ~3.5 GB ({_source_label})...")
    t_load = time.time()
    if resume_ckpt_path:
        unet = UNet2DConditionModel.from_pretrained(str(resume_ckpt_path / "unet"))
    else:
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_NAME, subfolder="unet", **_extra_kwargs,
        )
    if is_main:
        _log(f"  [5/5] U-Net ready ({time.time() - t_load:.1f}s)")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet.train()
    vae.to(device)
    text_encoder.to(device)

    if is_main:
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"Trainable U-Net params: {trainable_params:,} ({trainable_params/1e6:.0f}M)")

    # ── Optimizer ──────────────────────────────────────────
    import bitsandbytes as bnb

    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2,
    )

    # ── Dataset ────────────────────────────────────────────
    from streaming import StreamingDataset
    from torch.utils.data import DataLoader

    class InpaintingMDSDataset(torch.utils.data.Dataset):
        def __init__(self, mds_path, resolution=512):
            self.ds = StreamingDataset(local=mds_path, shuffle=True)
            self.resolution = resolution

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            sample = self.ds[idx]
            input_img = sample["input"].convert("RGB")
            target_img = sample["target"].convert("RGB")
            mask_img = sample["mask"].convert("L")
            input_img = input_img.resize((self.resolution, self.resolution), Image.LANCZOS)
            target_img = target_img.resize((self.resolution, self.resolution), Image.LANCZOS)
            mask_img = mask_img.resize((self.resolution, self.resolution), Image.NEAREST)
            input_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float() / 127.5 - 1.0
            target_tensor = torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float() / 127.5 - 1.0
            mask_tensor = torch.from_numpy(np.array(mask_img)).unsqueeze(0).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float()
            return {"input": input_tensor, "target": target_tensor, "mask": mask_tensor}

    train_dataset = InpaintingMDSDataset(
        str(Path(MDS_PATH) / "train"), resolution=IMAGE_RESOLUTION,
    )

    val_path = str(Path(MDS_PATH) / "OTR_easy")
    val_dataset = None
    if Path(val_path).exists():
        val_dataset = InpaintingMDSDataset(val_path, resolution=IMAGE_RESOLUTION)

    train_dataloader = DataLoader(
        train_dataset, batch_size=PER_GPU_BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=True,
    )

    if is_main:
        print(f"Training dataset: {len(train_dataset):,} samples")
        if val_dataset:
            print(f"Validation dataset: {len(val_dataset):,} samples (OTR_easy)")
        print(f"Steps per epoch: {len(train_dataloader):,}")
        print(f"Target steps: {NUM_TRAINING_STEPS:,}")

    # ── Prepare with accelerator ─────────────────────────────
    # NOTE: lr_scheduler is intentionally not passed to prepare().
    # We construct it AFTER prepare() so it is bound to the prepared
    # optimizer used by DDP/Accelerate in this process.
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # ── Learning rate scheduler (bind to prepared optimizer) ────────────
    # Scheduler created AFTER prepare() so it binds to the wrapped optimizer.
    # Stepped via adaptive helper that works across serverless and classic runtimes.
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_TRAINING_STEPS,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    def _step_scheduler_if_needed(global_step):
        """Advance LR scheduler exactly once per train step across runtimes.
        Serverless @distributed auto-steps; classic TorchDistributor does not.
        This checks whether the runtime already stepped, and only steps manually if not.
        """
        _sched = lr_scheduler.scheduler if hasattr(lr_scheduler, "scheduler") else lr_scheduler
        before = getattr(_sched, "last_epoch", None)
        should_step = (before is None) or (before < global_step)
        if should_step:
            lr_scheduler.step()


    # ── Restore training state (AFTER accelerator.prepare) ────────────
    if resume_ckpt_path:
        meta_file = resume_ckpt_path / "training_metadata.json"
        if meta_file.exists():
            accelerator.load_state(str(resume_ckpt_path))
            with open(str(meta_file)) as f:
                resume_step = json.load(f)["global_step"]
            if is_main:
                print(f"Restored full training state from step {resume_step}")

        if is_main and resume_step > 0:
            print(f"Remaining steps: {NUM_TRAINING_STEPS - resume_step:,}")

    # ── Encode fixed text prompt ─────────────────────────────
    text_inputs = tokenizer(
        PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    text_embeddings = text_embeddings.to(dtype=_mp_dtype)  # explicit cast: scheduler/VAE may upcast

    # ── MLflow setup (rank 0 only — attach to driver's run) ────────────
    # Driver created the run before @distributed launch. We attach to it
    # here so all metrics/artifacts go to one run. _DRIVER_RUN_ID is a
    # plain string injected into _clean_globals — safe for cloudpickle.
    _run_ckpt_path = CHECKPOINT_PATH  # already includes timestamp + run_id

    if is_main:
        import mlflow

        mlflow.end_run()  # clean up any stale active run

        # Attach to the driver's pre-created run
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        mlflow.enable_system_metrics_logging()
        run = mlflow.start_run(run_id=_DRIVER_RUN_ID)

        # Override source: @distributed sets source=air.py; fix to launch notebook
        _nb_path = _LAUNCH_NOTEBOOK_PATH if _LAUNCH_NOTEBOOK_PATH else ""
        _nb_id = _LAUNCH_NOTEBOOK_ID if _LAUNCH_NOTEBOOK_ID else ""
        if _nb_path:
            mlflow.set_tag("mlflow.source.name", _nb_path)
            mlflow.set_tag("mlflow.source.type", "NOTEBOOK")
            mlflow.set_tag("mlflow.databricks.notebookPath", _nb_path)
            mlflow.set_tag("mlflow.databricks.notebookID", _nb_id)

        # Resume mode for lineage
        if not resume_ckpt_path:
            _resume_mode = "fresh_start"
        elif RESUME_RUN_ID:
            _resume_mode = "explicit_run_id"
        else:
            _resume_mode = "latest_in_current_run"

        # Package versions for reproducibility
        import diffusers, transformers, accelerate
        _versions = {
            "pkg/diffusers": diffusers.__version__,
            "pkg/transformers": transformers.__version__,
            "pkg/accelerate": accelerate.__version__,
            "pkg/torch": torch.__version__,
        }
        try:
            import bitsandbytes as _bnb
            _versions["pkg/bitsandbytes"] = _bnb.__version__
        except Exception:
            pass

        mlflow.log_params({
            "model": MODEL_NAME,
            "checkpoint_path": _run_ckpt_path,
            "learning_rate": LEARNING_RATE,
            "per_gpu_batch_size": PER_GPU_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_GPU_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS,
            "num_training_steps": NUM_TRAINING_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "mixed_precision": MIXED_PRECISION,
            "image_resolution": IMAGE_RESOLUTION,
            "prompt": PROMPT,
            "num_gpus": accelerator.num_processes,
            "gradient_checkpointing": True,
            "optimizer": "AdamW8bit",
            "smoke_test": SMOKE_TEST,
            "resume_resolution_mode": _resume_mode,
            **({
                "resume_run_id": RESUME_RUN_ID,
                "resume_checkpoint_path": str(RESUME_CHECKPOINT_PATH),
                "resumed_from_step": resume_step,
            } if resume_ckpt_path else {}),
            **_versions,
        })

        # Log GPU/node info artifact
        _gpu_props = torch.cuda.get_device_properties(0)
        mlflow.log_text(json.dumps({
            "gpu_type": DETECTED_GPU,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(_gpu_props.total_memory / 1e9, 1),
            "num_gpus": accelerator.num_processes,
            "mixed_precision": MIXED_PRECISION,
            "effective_batch_size": PER_GPU_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS,
            "checkpoint_path": _run_ckpt_path,
            "topology": f"{"remote " if USE_REMOTE else ""}{"NVLink " if DETECTED_GPU == "H100" else ""}{"multi-node" if accelerator.num_processes > 1 else "single-node"} ({accelerator.num_processes} GPUs)",
        }, indent=2), artifact_file="run_info/environment.json")

        print(f"\nAttached to MLflow run: {_DRIVER_RUN_ID}")
        print(f"Checkpoints: {_run_ckpt_path}")

    # ── Training integrity check ─────────────────────────────
    # Verify all components are properly configured before starting
    _integrity_passed = True
    _integrity_issues = []

    try:
        # Check model is in train mode
        if not unet.training:
            _integrity_issues.append("U-Net not in training mode")
            _integrity_passed = False

        # Check VAE/text_encoder are frozen
        if any(p.requires_grad for p in vae.parameters()):
            _integrity_issues.append("VAE has trainable parameters (should be frozen)")
            _integrity_passed = False
        if any(p.requires_grad for p in text_encoder.parameters()):
            _integrity_issues.append("Text encoder has trainable parameters (should be frozen)")
            _integrity_passed = False

        # Check optimizer has parameters
        if len(list(optimizer.param_groups)) == 0:
            _integrity_issues.append("Optimizer has no parameter groups")
            _integrity_passed = False

        # Check dataloader is not empty
        if len(train_dataloader) == 0:
            _integrity_issues.append("Training dataloader is empty")
            _integrity_passed = False

    except Exception as _e:
        _integrity_issues.append(f"Integrity check exception: {_e}")
        _integrity_passed = False

    if is_main:
        mlflow.log_param("training_integrity_passed", _integrity_passed)
        if not _integrity_passed:
            mlflow.log_text("\n".join(_integrity_issues), artifact_file="run_info/integrity_issues.txt")
            print(f"\nWARNING: Training integrity issues detected:")
            for issue in _integrity_issues:
                print(f"  - {issue}")
            raise RuntimeError("Training integrity check failed")

    # ── Training loop ────────────────────────────────────────
    global_step = resume_step
    running_loss = 0.0
    best_val_loss = float('inf')
    log_every = LOG_EVERY
    val_every = VAL_EVERY
    save_every = SAVE_EVERY
    sample_every = SAMPLE_EVERY

    if is_main:
        print(f"\nStarting training from step {global_step}...")
        print(f"Log every {log_every} steps | Val every {val_every} steps | Save every {save_every} steps")

    train_start = time.time()
    _log_interval_start = train_start  # separate timer for steps/sec (train_start stays fixed for total_minutes)
    unet.train()

    dataloader_iter = iter(train_dataloader)

    while global_step < NUM_TRAINING_STEPS:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)

        with accelerator.accumulate(unet):
            # Move batch to device (cast to _mp_dtype for bf16/fp16 frozen model compat)
            input_images = batch["input"].to(device, dtype=_mp_dtype)
            target_images = batch["target"].to(device, dtype=_mp_dtype)  # keep pixels for sample grid
            target_latents = vae.encode(target_images).latent_dist.sample()
            target_latents = (target_latents * vae.config.scaling_factor).to(dtype=_mp_dtype)  # scaling_factor (Python float) promotes bf16→f32
            masks = batch["mask"].to(device, dtype=_mp_dtype)

            # Prepare inpainting condition: mask + masked image in latent space
            masked_input = input_images * (1 - masks)
            with torch.no_grad():
                masked_image_latents = vae.encode(masked_input.to(dtype=_mp_dtype)).latent_dist.sample()
                masked_image_latents = (masked_image_latents * vae.config.scaling_factor).to(dtype=_mp_dtype)
            # Downsample mask to latent resolution (512→64)
            mask_latent = F.interpolate(masks, size=target_latents.shape[-2:], mode='nearest')

            # Sample noise and timesteps
            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=device,
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
            noisy_latents = noisy_latents.to(dtype=_mp_dtype)  # add_noise upcasts via float64 alphas

            # Inpainting U-Net: noisy_latents (4) + mask (1) + masked_image_latents (4) = 9 ch
            unet_input = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)

            # Predict noise — explicit dtype cast at model boundary
            with accelerator.autocast():
                noise_pred = unet(
                    unet_input.to(dtype=_mp_dtype),
                    timesteps,
                    text_embeddings.repeat(bsz, 1, 1).to(dtype=_mp_dtype),
                ).sample

            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backward pass
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Step scheduler outside accumulate — the wrapped scheduler's step()
        # is a no-op inside accumulate() on multi-GPU @distributed. Stepping
        # here on sync boundaries ensures the LR schedule advances correctly.
        if accelerator.sync_gradients:
            lr_scheduler.step()

        # Track loss
        running_loss += loss.detach().item()

        if accelerator.sync_gradients:
            global_step += 1
            _step_scheduler_if_needed(global_step)

            # ── Logging ──────────────────────────────────────────
            if global_step % log_every == 0 and is_main:
                avg_loss = running_loss / log_every
                current_lr = optimizer.param_groups[0]["lr"]

                elapsed = time.time() - _log_interval_start
                steps_per_sec = log_every / elapsed if elapsed > 0 else 0

                mlflow.log_metrics({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "train/steps_per_sec": steps_per_sec,
                }, step=global_step)

                print(f"Step {global_step}/{NUM_TRAINING_STEPS} | "
                      f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                      f"Speed: {steps_per_sec:.2f} steps/s")

                running_loss = 0.0
                _log_interval_start = time.time()

            # ── Validation ───────────────────────────────────────
            if global_step % val_every == 0 and val_dataset and is_main:
                unet.eval()
                val_loss = 0.0
                val_steps = min(10, len(val_dataset) // PER_GPU_BATCH_SIZE)

                with torch.no_grad():
                    for val_i in range(val_steps):
                        val_batch_indices = torch.randint(0, len(val_dataset), (PER_GPU_BATCH_SIZE,))
                        val_batch = torch.utils.data.default_collate(
                            [val_dataset[i.item()] for i in val_batch_indices]
                        )

                        val_input = val_batch["input"].to(device, dtype=_mp_dtype)
                        val_target_latents = vae.encode(val_batch["target"].to(device, dtype=_mp_dtype)).latent_dist.sample()
                        val_target_latents = (val_target_latents * vae.config.scaling_factor).to(dtype=_mp_dtype)
                        val_masks = val_batch["mask"].to(device, dtype=_mp_dtype)

                        val_masked_input = val_input * (1 - val_masks)
                        val_masked_image_latents = vae.encode(val_masked_input.to(dtype=_mp_dtype)).latent_dist.sample()
                        val_masked_image_latents = (val_masked_image_latents * vae.config.scaling_factor).to(dtype=_mp_dtype)
                        val_mask_latent = F.interpolate(val_masks, size=val_target_latents.shape[-2:], mode='nearest')

                        val_noise = torch.randn_like(val_target_latents)
                        val_timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (val_target_latents.shape[0],), device=device,
                        ).long()
                        val_noisy_latents = noise_scheduler.add_noise(
                            val_target_latents, val_noise, val_timesteps
                        )
                        val_noisy_latents = val_noisy_latents.to(dtype=_mp_dtype)

                        val_unet_input = torch.cat([val_noisy_latents, val_mask_latent, val_masked_image_latents], dim=1)

                        with accelerator.autocast():
                            
                            val_noise_pred = unet(
                                val_unet_input,
                                val_timesteps,
                                text_embeddings.repeat(val_target_latents.shape[0], 1, 1),
                            ).sample

                        val_loss += F.mse_loss(
                            val_noise_pred.float(), val_noise.float(), reduction="mean"
                        ).item()

                val_loss /= val_steps
                mlflow.log_metric("val/loss", val_loss, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.log_metric("val/best_loss", best_val_loss, step=global_step)
                    # Save best U-Net for inference (overwrites each time)
                    best_dir = Path(_run_ckpt_path) / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_unet.save_pretrained(str(best_dir / "unet"))
                    with open(str(best_dir / "training_metadata.json"), "w") as f:
                        json.dump({"global_step": global_step, "val_loss": val_loss}, f)
                    mlflow.log_metric("checkpoint/best_step", global_step, step=global_step)
                    mlflow.log_text(
                        json.dumps({
                            "global_step": global_step,
                            "val_loss": val_loss,
                            "checkpoint_path": str(best_dir),
                            "unet_path": str(best_dir / "unet"),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "is_best": True,
                        }, indent=2),
                        artifact_file="checkpoints/best.json",
                    )
                    print(f"Validation | Loss: {val_loss:.4f} (new best -- saved)")
                else:
                    print(f"Validation | Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
                unet.train()

            # ── Save checkpoint ──────────────────────────────────
            if global_step % save_every == 0 or global_step == NUM_TRAINING_STEPS:
                if is_main:
                    # Use run-specific checkpoint path
                    ckpt_dir = Path(_run_ckpt_path) / f"checkpoint-{global_step}"
                    print(f"Saving checkpoint to {ckpt_dir}...")

                # Collective save (all ranks participate)
                accelerator.save_state(str(Path(_run_ckpt_path) / f"checkpoint-{global_step}"))

                if is_main:
                    # Save U-Net in diffusers format (for inference compatibility)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_unet.save_pretrained(str(ckpt_dir / "unet"))

                    # Save metadata + log to MLflow
                    meta = {"global_step": global_step, "checkpoint_path": str(ckpt_dir)}
                    with open(str(ckpt_dir / "training_metadata.json"), "w") as f:
                        json.dump(meta, f, indent=2)

                    # Log checkpoint manifest artifact
                    ckpt_manifest = {
                        "global_step": global_step,
                        "checkpoint_path": str(ckpt_dir),
                        "unet_path": str(ckpt_dir / "unet"),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    mlflow.log_text(
                        json.dumps(ckpt_manifest, indent=2),
                        artifact_file=f"checkpoints/checkpoint-{global_step}.json",
                    )
                    mlflow.log_metric("checkpoint/step", global_step, step=global_step)
                    print(f"Checkpoint saved: {ckpt_dir.name}")

            # ── Generate samples ─────────────────────────────────
            if (global_step % sample_every == 0 or global_step == NUM_TRAINING_STEPS) and is_main:
                unet.eval()
                with torch.no_grad():
                    # Use first 4 samples from training batch
                    sample_input = input_images[:4]
                    sample_mask = masks[:4]
                    _n_samples = sample_input.shape[0]
                    sample_masked = sample_input * (1 - sample_mask)

                    # Encode masked input and downsample mask to latent space
                    sample_masked_latents = vae.encode(sample_masked.to(dtype=_mp_dtype)).latent_dist.sample()
                    sample_masked_latents = (sample_masked_latents * vae.config.scaling_factor).to(dtype=_mp_dtype)
                    sample_mask_latent = F.interpolate(sample_mask, size=sample_masked_latents.shape[-2:], mode='nearest')

                    # Start from pure noise
                    sample_latents = torch.randn(
                        (_n_samples, 4, IMAGE_RESOLUTION // 8, IMAGE_RESOLUTION // 8),
                        device=device, dtype=_mp_dtype,
                    )

                    # Simple denoising loop (10 steps for fast sampling)
                    noise_scheduler.set_timesteps(10, device=device)
                    for t in noise_scheduler.timesteps:
                        # Inpainting: concat noisy latents + mask + masked image latents
                        sample_unet_input = torch.cat([sample_latents, sample_mask_latent, sample_masked_latents], dim=1)
                        with accelerator.autocast():
                            noise_pred = unet(
                                sample_unet_input,
                                t,
                                text_embeddings.repeat(_n_samples, 1, 1),
                            ).sample
                        sample_latents = noise_scheduler.step(
                            noise_pred, t, sample_latents
                        ).prev_sample

                    # Decode to pixel space
                    sample_latents = sample_latents / vae.config.scaling_factor
                    sample_images = vae.decode(sample_latents.to(dtype=_mp_dtype)).sample 
                    sample_images = (sample_images / 2 + 0.5).clamp(0, 1)

                    # Save samples as 4-panel grid: input | mask | inpainted | target
                    _res = IMAGE_RESOLUTION
                    sample_target = target_images[:_n_samples]
                    for i in range(_n_samples):
                        # Denormalize [-1,1] -> [0,255] for input/target
                        inp_arr = ((sample_input[i].cpu().float().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        tgt_arr = ((sample_target[i].cpu().float().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        # Mask: single-channel [0,1] -> grayscale RGB
                        msk_arr = (sample_mask[i, 0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                        # Inpainted: already [0,1] from clamp
                        out_arr = (sample_images[i].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                        grid = Image.new("RGB", (_res * 4, _res))
                        grid.paste(Image.fromarray(inp_arr), (0, 0))
                        grid.paste(Image.fromarray(np.stack([msk_arr]*3, axis=-1)), (_res, 0))
                        grid.paste(Image.fromarray(out_arr), (_res * 2, 0))
                        grid.paste(Image.fromarray(tgt_arr), (_res * 3, 0))
                        mlflow.log_image(grid, f"samples/step_{global_step:05d}_sample_{i}.png")

                unet.train()

    # ── Training complete ────────────────────────────────────
    if is_main:
        # Save final checkpoint
        final_dir = Path(_run_ckpt_path) / "final"
        print(f"\nSaving final checkpoint to {final_dir}...")

    accelerator.save_state(str(Path(_run_ckpt_path) / "final"))

    if is_main:
        # Save U-Net in diffusers format (for inference compatibility)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(str(final_dir / "unet"))

        meta = {"global_step": global_step, "checkpoint_path": str(final_dir)}
        with open(str(final_dir / "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        total_minutes = (time.time() - train_start) / 60
        mlflow.log_metric("train/total_minutes", total_minutes)

        # Final checkpoint manifest
        final_manifest = {
            "global_step": global_step,
            "checkpoint_path": str(final_dir),
            "unet_path": str(final_dir / "unet"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_minutes": round(total_minutes, 2),
            "is_final": True,
        }
        mlflow.log_text(
            json.dumps(final_manifest, indent=2),
            artifact_file="checkpoints/final.json",
        )

        print(f"\nTraining complete: {global_step} steps in {total_minutes:.1f} min")

    # ── Gather and log node logs as MLflow artifacts ──────────
    # Restores original stdout, then collects each rank's captured log
    # via collective gather. Rank 0 logs each as run_info/rank_N_log.txt.
    # This provides per-node logs similar to what the job run UI shows natively.
    sys.stdout = _original_stdout  # restore before collective op
    _my_log = _tee.get_log()

    if torch.distributed.is_initialized():
        _all_logs = [None] * accelerator.num_processes
        torch.distributed.gather_object(
            _my_log,
            _all_logs if is_main else None,
            dst=0,
        )
    else:
        _all_logs = [_my_log]

    if is_main:
        for _rank_idx, _rank_log in enumerate(_all_logs):
            if _rank_log:
                mlflow.log_text(
                    _rank_log,
                    artifact_file=f"run_info/rank_{_rank_idx}_log.txt",
                )
        print(f"Logged {len(_all_logs)} rank log(s) as MLflow artifacts")

        mlflow.end_run()

    # ── Return per-rank info (collected by @distributed runtime) ─────
    # Per @distributed docs, @distributed collects return values from all ranks.
    # Each rank reports its GPU stats; rank 0 adds MLflow run info.
    import socket as _socket
    _local_idx = accelerator.local_process_index
    _gpu_props_end = torch.cuda.get_device_properties(_local_idx)
    rank_info = {
        "global_rank": accelerator.process_index,
        "local_rank": _local_idx,
        "hostname": _socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(_local_idx),
        "gpu_mem_gb": round(_gpu_props_end.total_memory / 1e9, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(_local_idx) / 1e9, 2),
    }
    if is_main:
        rank_info.update({
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run.info.run_name,
            "checkpoint_path": _run_ckpt_path,
            "total_steps": global_step,
            "total_minutes": round(total_minutes, 2),
        })

    return rank_info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Launch training
# MAGIC **Compute:** Serverless GPU via [`@distributed`](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training/). Set `Compute: Remote` widget to control remote vs co-located provisioning.
# MAGIC | Compute | Remote | How |
# MAGIC |---|---|---|
# MAGIC | **Serverless multi-GPU** | `true` | `@distributed(remote=True)` — provisions remote GPUs, companion MLflow run |
# MAGIC | **Serverless multi-GPU** | `false` | `@distributed` — uses co-located GPUs (NVLink on H100), no companion run |
# MAGIC | **Single GPU** (fallback) | N/A | Direct call if `@distributed` unavailable |
# MAGIC | **Classic compute** | N/A | `TorchDistributor` (separate `_classic` notebook) |

# COMMAND ----------

# DBTITLE 1,Resuming from a Previous Run
# MAGIC %md
# MAGIC ### Resuming from a Previous Run
# MAGIC
# MAGIC #### Checkpoint isolation on resume
# MAGIC
# MAGIC Every training launch creates a **new MLflow run** with its own run ID — even when resuming.
# MAGIC New checkpoints are written to a **separate run-scoped directory** so the original checkpoint stays intact as a rollback point:
# MAGIC
# MAGIC ```
# MAGIC CHECKPOINT_BASE/
# MAGIC ├── <timestamp>_<original_run_id>/          ← READ: resume loads weights + optimizer from here
# MAGIC │   └── checkpoint-500/
# MAGIC └── <timestamp>_<new_run_id>/                ← WRITE: new checkpoints saved here
# MAGIC     ├── checkpoint-750/
# MAGIC     └── checkpoint-1000/
# MAGIC ```
# MAGIC
# MAGIC The new run logs `resume_run_id` and `resume_checkpoint_path` as MLflow params, linking it back to the original for traceability.
# MAGIC
# MAGIC #### How to resume
# MAGIC
# MAGIC Set these two widgets in the **Configuration** cell before running the launch cell:
# MAGIC
# MAGIC | Widget | What to set | Notes |
# MAGIC |---|---|---|
# MAGIC | `resume_from` | `latest` or `checkpoint-N` | `latest` auto-finds the newest valid checkpoint; explicit name (e.g. `checkpoint-500`) picks a specific one |
# MAGIC | `resume_run_id` | Previous run's ID | The run-scoped directory name under `CHECKPOINT_BASE/`. Find it in the MLflow run's `checkpoint_path` param, or by listing `CHECKPOINT_BASE/` |
# MAGIC
# MAGIC Leave **both blank** for a fresh start. 
# MAGIC
# MAGIC #### Example: resume from step 500 of a interrupted run
# MAGIC
# MAGIC | Widget | Value |
# MAGIC |---|---|
# MAGIC | `resume_from` | `latest` |
# MAGIC | `resume_run_id` | `7ec5649685be405abc950978b49fdd87` |
# MAGIC
# MAGIC This loads weights and optimizer state from `CHECKPOINT_BASE/<timestamp>_7ec5649685be405abc950978b49fdd87/checkpoint-500/`, then creates a **new** MLflow run with a fresh run ID. All subsequent checkpoints are written to `CHECKPOINT_BASE/<timestamp>_<new_run_id>/` — never back into the original directory.
# MAGIC This separation ensures the source checkpoint remains untouched as a rollback point if the resumed run diverges or fails, and aligns with MLflow's model where each run is an independent experiment with its own artifact lineage.

# COMMAND ----------

# DBTITLE 1,Monitoring, System Metrics & Companion Run
# MAGIC %md
# MAGIC ### Monitoring, System Metrics & Companion Run
# MAGIC
# MAGIC #### What MLflow tracks in your training run
# MAGIC
# MAGIC `mlflow.enable_system_metrics_logging()` starts a background monitor (~10s polling) that logs GPU memory, utilization, power, CPU, RAM, and network. Rank 0's process sees all GPUs on its node via pynvml.
# MAGIC
# MAGIC #### Companion MLflow run (remote=true only)
# MAGIC
# MAGIC When `Compute: Remote = true`, the `@distributed` runtime creates a **companion run** with:
# MAGIC - **Per-node GPU logs** (`logs/node_0/gpu_N-0.chunk.txt`) — individual worker stdout
# MAGIC - **System metrics** per node
# MAGIC - **Node command** (`usercommand.bash`)
# MAGIC
# MAGIC The companion run lands in an auto-generated experiment. This notebook links it to your training run via tags (`companion_run_id`, `companion_run_url`) and copies GPU logs to `companion_logs/` artifacts.
# MAGIC
# MAGIC When `Compute: Remote = false`, there is no companion run — system metrics are logged directly to your training run via pynvml.
# MAGIC
# MAGIC #### What's in each run
# MAGIC
# MAGIC | What you need | Where to find it |
# MAGIC |---|---|
# MAGIC | Training metrics, params, sample images | **Training run** |
# MAGIC | Consolidated rank logs (all ranks) | **Training run** → `run_info/rank_N_log.txt` |
# MAGIC | Per-node GPU logs (remote=true) | **Training run** → `companion_logs/` or **Companion run** → `logs/node_0/` |
# MAGIC | System metrics (GPU mem/util/power) | **Training run** (pynvml) + **Companion run** (runtime, remote=true only) |
# MAGIC | Best checkpoint info | **Training run** → `checkpoints/best.json` |
# MAGIC
# MAGIC #### Return value pattern
# MAGIC
# MAGIC The training function returns a `rank_info` dict from every rank. `@distributed` collects these into a list:
# MAGIC
# MAGIC ```python
# MAGIC results = _run_distributed.distributed()
# MAGIC # [{"global_rank": 0, ..., "run_id": "...", "total_steps": 10}, {"global_rank": 1, ...}, ...]
# MAGIC ```
# MAGIC
# MAGIC For single-GPU (no `@distributed`), the direct call returns a single dict, wrapped in a list.

# COMMAND ----------

# DBTITLE 1,[Optional] Identify Notebook Hostname and IP
# Uncomment to check if training runs on the same node (no extra cost)
# or a separate remote node (double billing with remote=true).
# Compare output with hostname in node_map.json artifact after training.
#
# import socket
# print(f"Notebook host: {socket.getfqdn()}")
# print(f"Notebook IP:   {socket.gethostbyname(socket.gethostname())}")

# COMMAND ----------

# DBTITLE 1,Launch training with unified MLflow run
import torch, os, gc, json, types
from pathlib import Path
from datetime import datetime

# Use GPU type from config cell (consistent with widget / preset)
gpu_type = DETECTED_GPU
num_gpus = torch.cuda.device_count()

print(f"GPU type: {gpu_type}")
print(f"Available GPUs: {num_gpus}")
print(f"Distributed GPUs requested: {NUM_DISTRIBUTED_GPUS}")
print(f"Mixed precision: {MIXED_PRECISION}")

# -- Clean up MLflow state before @distributed ----
# Clear MLFLOW_* env vars that cause issues, but KEEP tracking URI/token
# so @distributed workers can reach the MLflow server and attach to the
# driver-created run.
import mlflow
mlflow.end_run()
mlflow.autolog(disable=True)
_KEEP_MLFLOW = {"MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_INSECURE_TLS"}
for _k in list(os.environ):
    if _k.startswith("MLFLOW_") and _k not in _KEEP_MLFLOW:
        del os.environ[_k]

gc.collect()
torch.cuda.empty_cache()

# -- Use pre-downloaded pipeline from UC Volumes (avoids HF download on workers) ---
if Path(BASE_PIPELINE_PATH).is_dir() and (Path(BASE_PIPELINE_PATH) / "model_index.json").exists():
    MODEL_NAME = BASE_PIPELINE_PATH
    print(f"\nUsing cached pipeline from UC Volumes: {MODEL_NAME}")
else:
    print(f"\nWARNING: Cached pipeline not found at {BASE_PIPELINE_PATH}, will download from HF: {MODEL_NAME}")

# -- Capture notebook identity for MLflow source tags (before stripping dbutils) --
# These plain strings survive pickle to @distributed workers. The training
# function sets them as source tags so the MLflow run links to this notebook
# instead of air.py (the @distributed launcher script).
try:
    _nb_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _LAUNCH_NOTEBOOK_PATH = _nb_ctx.notebookPath().get()
    _LAUNCH_NOTEBOOK_ID = str(_nb_ctx.tags().get("notebookId").get())
except Exception:
    _LAUNCH_NOTEBOOK_PATH, _LAUNCH_NOTEBOOK_ID = "", ""

# -- Strip Spark/dbutils from training function globals ---
# The Databricks runtime injects spark, dbutils, sc, sqlContext into the
# notebook module namespace. train_sd2_inpainting.__globals__ points to
# that same namespace. Even though the training function never uses Spark,
# cloudpickle serializes the function object -> captures __globals__ ->
# encounters live SparkSession -> workers can't deserialize it.
#
# Fix: create a clean copy of the function with Spark objects stripped
# from __globals__. All config variables (MIXED_PRECISION, CHECKPOINT_PATH,
# etc.) remain intact -- only the runtime-injected objects are removed.
_STRIP_FROM_GLOBALS = {'spark', 'dbutils', 'sc', 'sqlContext', 'spark_refs', 'has_spark', 'display'}
_clean_globals = {k: v for k, v in train_sd2_inpainting.__globals__.items()
                  if k not in _STRIP_FROM_GLOBALS}
_train_fn_clean = types.FunctionType(
    train_sd2_inpainting.__code__,
    _clean_globals,
    train_sd2_inpainting.__name__,
    train_sd2_inpainting.__defaults__,
    train_sd2_inpainting.__closure__,
)

_stripped = sorted(_STRIP_FROM_GLOBALS & set(train_sd2_inpainting.__globals__))
print(f"\nStripped from fn.__globals__ for pickle safety: {_stripped}")
print(f"Config vars intact: MIXED_PRECISION={_clean_globals.get('MIXED_PRECISION')}, "
      f"NUM_TRAINING_STEPS={_clean_globals.get('NUM_TRAINING_STEPS')}, "
      f"CHECKPOINT_PATH={'...' + str(_clean_globals.get('CHECKPOINT_PATH', ''))[-30:]}")

# Set up base checkpoint path (timestamp only -- training function appends run_id)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH = f"{CHECKPOINT_BASE}/{timestamp}"
_clean_globals['CHECKPOINT_PATH'] = CHECKPOINT_PATH

# -- Create MLflow run on DRIVER before @distributed ----
# The driver creates the run here so:
#   1. The run ID is a plain string (safe for cloudpickle)
#   2. All workers attach to the SAME run (no duplicate runs)
#   3. System metrics on the driver capture all local GPUs
# We immediately end_run() so mlflow state doesn't pollute the closure.
# The training function re-attaches via mlflow.start_run(run_id=...).
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
mlflow.enable_system_metrics_logging()
_run_name = "sd_inpainting_finetune"
if SMOKE_TEST:
    _run_name += "_smoke"
_run_name += f"_{DETECTED_GPU}_{timestamp}"
_driver_run = mlflow.start_run(run_name=_run_name)
_DRIVER_RUN_ID = _driver_run.info.run_id  # plain string
_DRIVER_EXPERIMENT_ID = _driver_run.info.experiment_id
# NOTE: keep the run OPEN — training function on rank 0 will reattach.
# mlflow.end_run() here would make the run invisible to workers.

# Inject into clean globals so training function can see it
_clean_globals['_DRIVER_RUN_ID'] = _DRIVER_RUN_ID
_clean_globals['_LAUNCH_NOTEBOOK_PATH'] = _LAUNCH_NOTEBOOK_PATH
_clean_globals['_LAUNCH_NOTEBOOK_ID'] = _LAUNCH_NOTEBOOK_ID
_clean_globals['_MLFLOW_TRACKING_URI'] = os.environ.get("MLFLOW_TRACKING_URI", "")
_clean_globals['_MLFLOW_TRACKING_TOKEN'] = os.environ.get("MLFLOW_TRACKING_TOKEN", "")

# Checkpoint path includes run_id (known now since driver created the run)
CHECKPOINT_PATH = f"{CHECKPOINT_PATH}_{_DRIVER_RUN_ID}"
_clean_globals['CHECKPOINT_PATH'] = CHECKPOINT_PATH
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"\nMLflow run: {_run_name}")
print(f"Run ID: {_DRIVER_RUN_ID}")
print(f"Checkpoints: {CHECKPOINT_PATH}")

# Packages that @distributed workers need (they don't inherit %pip install)
_WORKER_PACKAGES = [
    "accelerate", "bitsandbytes", "diffusers", "transformers",
    "huggingface_hub~=0.36", "safetensors",
    "mosaicml-streaming~=0.13.0", "pillow~=12.1",
]


def _patch_gradient_checkpoint():
    """Patch torch.utils.checkpoint to use use_reentrant=False.

    PyTorch >=2.4 with reentrant checkpointing (the default) doesn't preserve
    torch.autocast context during recomputation, causing CheckpointError with
    mixed-precision training. Non-reentrant mode fixes this.
    """
    import torch.utils.checkpoint as _ckpt
    _orig = _ckpt.checkpoint
    def _compat(*args, **kwargs):
        kwargs.setdefault('use_reentrant', False)
        return _orig(*args, **kwargs)
    _ckpt.checkpoint = _compat


# -- Post-training summary (shared by all compute paths) ---
def _print_run_summary(results):
    """Print node summary table + MLflow UI URLs from returned rank info."""
    if not results or results[0] is None:
        print("\nWARNING: Training returned no results -- check logs above.")
        return

    results = sorted(results, key=lambda r: r["global_rank"])

    # Node summary table
    print(f"\n{'='*78}")
    print(f"  {'Rank':>4}  {'Host':<20}  {'GPU':<28}  {'Mem':>7}  {'Peak':>7}")
    print(f"{'-'*78}")
    for r in results:
        host_short = r["hostname"][-20:]
        gpu_short = r["gpu_name"][:28]
        print(f"  {r['global_rank']:>4}  {host_short:<20}  {gpu_short:<28}  "
              f"{r['gpu_mem_gb']:>5.1f}GB  {r['peak_mem_gb']:>5.2f}GB")
    print(f"{'='*78}")

    # Topology detection
    hostnames = set(r["hostname"] for r in results)
    if len(hostnames) == 1:
        topo = f"single-node ({len(results)} GPUs)"
    else:
        topo = f"multi-node ({len(hostnames)} hosts, {len(results)} GPUs)"
    print(f"  Topology: {topo}")

    # MLflow UI URLs (from rank 0)
    rank0 = next((r for r in results if r["global_rank"] == 0), None)
    if rank0 and "run_id" in rank0:
        run_id = rank0["run_id"]
        exp_id = rank0["experiment_id"]

        try:
            _ws_url = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
            if not _ws_url:
                _ws_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
        except Exception:
            _ws_url = ""

        if _ws_url:
            exp_url = f"https://{_ws_url}/ml/experiments/{exp_id}"
            run_url = f"https://{_ws_url}/ml/experiments/{exp_id}/runs/{run_id}"
        else:
            exp_url = f"(experiment {exp_id})"
            run_url = f"(run {run_id})"

        print(f"\n  Run:        {rank0.get('run_name', 'N/A')}")
        print(f"  Steps:      {rank0.get('total_steps', 'N/A')}")
        print(f"  Time:       {rank0.get('total_minutes', 'N/A')} min")
        print(f"  Checkpoint: {rank0.get('checkpoint_path', 'N/A')}")
        print(f"\n{'-'*78}")
        print(f"  >> Experiment: {exp_url}")
        print(f"  >> This Run:   {run_url}")
        print(f"{'='*78}\n")

        # Log node_map.json artifact to MLflow
        try:
            client = mlflow.tracking.MlflowClient()
            client.log_text(
                run_id,
                json.dumps(results, indent=2),
                artifact_file="run_info/node_map.json",
            )
        except Exception as _e:
            print(f"  (Could not log node_map artifact: {_e})")


# -- Launch training ----
results = None

try:
    if NUM_DISTRIBUTED_GPUS > 1:
        try:
            from serverless_gpu import distributed

            # Launch with or without remote=True based on widget selection.
            # remote=True (Private Preview):
            #   + Companion MLflow run with per-node GPU logs
            #   + Cross-accelerator launching (e.g. A10 → H100)
            #   - Provisions remote node even when same GPU type attached (double billing)
            #   - /tmp permission issues on multi-node (inference cache)
            # remote=False (documented):
            #   + Uses co-located GPUs directly (NVLink on H100)
            #   + No double billing, clean /tmp
            #   - No companion MLflow run (no per-GPU logs)
            #   - Notebook must be attached to matching GPU type
            # If you find remote=True useful, provide feedback to Databricks —
            # user demand helps prioritize official support.
            if USE_REMOTE:
                @distributed(gpus=NUM_DISTRIBUTED_GPUS, gpu_type=gpu_type, remote=True)
                def _run_distributed():
                    import subprocess, sys
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q"] + _WORKER_PACKAGES
                    )
                    _patch_gradient_checkpoint()
                    return _train_fn_clean()

                print(f"Launching {NUM_DISTRIBUTED_GPUS}x {gpu_type} via @distributed (remote=True)...")
            else:
                @distributed(gpus=NUM_DISTRIBUTED_GPUS, gpu_type=gpu_type)
                def _run_distributed():
                    import subprocess, sys
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q"] + _WORKER_PACKAGES
                    )
                    _patch_gradient_checkpoint()
                    return _train_fn_clean()

                print(f"Launching {NUM_DISTRIBUTED_GPUS}x {gpu_type} via @distributed (co-located)...")

            results = _run_distributed.distributed()

            # Validate all ranks completed (per @distributed docs pattern)
            assert results is not None, "@distributed returned None"
            assert len(results) == NUM_DISTRIBUTED_GPUS, (
                f"Expected {NUM_DISTRIBUTED_GPUS} rank results, "
                f"got {len(results)}"
            )
            ranks_seen = sorted(r["global_rank"] for r in results)
            expected_ranks = list(range(NUM_DISTRIBUTED_GPUS))
            assert ranks_seen == expected_ranks, (
                f"Missing ranks: expected {expected_ranks}, got {ranks_seen}"
            )

        except ImportError:
            if DDP_ERROR_HANDLING == "fail":
                raise RuntimeError(
                    "serverless_gpu not available and ddp_error_handling=fail"
                ) from None
            print("serverless_gpu not available -- falling back to single GPU...")
            _patch_gradient_checkpoint()
            result = _train_fn_clean()
            assert result is not None, "Training returned None"
            results = [result]

        except AssertionError:
            raise  # re-raise rank validation failures

        except Exception as e:
            if DDP_ERROR_HANDLING == "fail":
                raise
            print(f"@distributed failed: {e}")
            print("Falling back to single GPU...")
            _patch_gradient_checkpoint()
            result = _train_fn_clean()
            assert result is not None, "Training returned None"
            results = [result]
    else:
        print("Running on single GPU...")
        _patch_gradient_checkpoint()
        result = _train_fn_clean()
        assert result is not None, "Training returned None"
        results = [result]

    # Print run summary (all compute paths converge here)
    _print_run_summary(results)

    # Adopt run on driver so notebook UI shows "Logged 1 run" link
    # The notebook UI reads MLFLOW_EXPERIMENT_ID to build the "Learn more" URL.
    # We cleared MLFLOW_* env vars earlier (to avoid pickle issues), so the
    # notebook defaults to the auto-experiment at the notebook path. Setting
    # the env var here ensures the link points to the actual experiment+run.
    rank0 = next((r for r in results if r["global_rank"] == 0), None)
    if rank0 and "run_id" in rank0:
        _experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        os.environ["MLFLOW_EXPERIMENT_ID"] = _experiment.experiment_id
        os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_PATH

        # Get notebook identity to fix source attribution
        # (run was created inside @distributed where source = air.py)
        try:
            _nb_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            _notebook_path = _nb_ctx.notebookPath().get()
            _notebook_id = str(_nb_ctx.tags().get("notebookId").get())
        except Exception:
            _notebook_path, _notebook_id = "", ""

        # End the driver's active run before re-opening for adoption.
        # The driver started this run earlier and kept it open so workers
        # could attach. The training function's end_run() ran in a separate
        # process (@distributed worker), so the driver still has it active.
        mlflow.end_run()

        with mlflow.start_run(run_id=rank0["run_id"]):
            # Override source from air.py to this notebook
            if _notebook_path:
                mlflow.set_tag("mlflow.source.name", _notebook_path)
                mlflow.set_tag("mlflow.source.type", "NOTEBOOK")
                mlflow.set_tag("mlflow.databricks.notebookPath", _notebook_path)
                mlflow.set_tag("mlflow.databricks.notebookID", _notebook_id)
            if NUM_DISTRIBUTED_GPUS > 1:
                _remote_label = "remote" if USE_REMOTE else "local"
                mlflow.log_param("launch_mode",
                    f"distributed_{_remote_label}_{NUM_DISTRIBUTED_GPUS}x{gpu_type}")
            else:
                mlflow.log_param("launch_mode", "single_gpu")

            # Log companion run link (only exists with remote=True)
            if USE_REMOTE:
                # The companion run has per-node system metrics + GPU logs.
                # It lands in an auto-generated experiment created by @distributed runtime.
                try:
                    _client = mlflow.tracking.MlflowClient()
                    _training_run_id = rank0["run_id"]
                    _training_exp_id = rank0["experiment_id"]
                    _training_start = _driver_run.info.start_time

                    # Search for the companion experiment created by @distributed.
                    # Each @distributed call creates a NEW experiment — match by
                    # experiment creation_time (not run start_time, which catches
                    # stale companions from previous runs).
                    _comp_run_id = None
                    _comp_exp_id = None
                    _all_experiments = _client.search_experiments(
                        order_by=["creation_time DESC"],
                        max_results=10,
                    )
                    for _exp in _all_experiments:
                        if _exp.experiment_id == _training_exp_id:
                            continue
                        if _exp.creation_time < _training_start - 60_000:
                            continue  # too old — skip
                        _runs = _client.search_runs(
                            experiment_ids=[_exp.experiment_id],
                            order_by=["start_time DESC"],
                            max_results=1,
                        )
                        if _runs:
                            _comp_run_id = _runs[0].info.run_id
                            _comp_exp_id = _exp.experiment_id
                            break

                    if _comp_run_id:
                        mlflow.set_tag("companion_run_id", _comp_run_id)
                        mlflow.set_tag("companion_experiment_id", _comp_exp_id)

                        try:
                            _ws_url = os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
                            if not _ws_url:
                                _ws_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
                        except Exception:
                            _ws_url = ""
                        if _ws_url:
                            _comp_url = f"https://{_ws_url}/ml/experiments/{_comp_exp_id}/runs/{_comp_run_id}"
                            mlflow.set_tag("companion_run_url", _comp_url)
                            print(f"  >> Companion run: {_comp_url}")

                        # Copy per-node GPU logs from companion run to our training run.
                        # The companion run uploads artifacts asynchronously after @distributed
                        # returns, so we wait and retry.
                        import time as _time
                        _copied = 0
                        print("  Waiting for companion run artifacts...")
                        for _attempt in range(6):  # up to 30s
                            _time.sleep(5)
                            try:
                                _log_dirs = _client.list_artifacts(_comp_run_id, "logs")
                                if not _log_dirs:
                                    print(f"    attempt {_attempt+1}/6: no artifacts yet...")
                                    continue
                                for _node_dir in _log_dirs:
                                    if not _node_dir.is_dir:
                                        continue
                                    _node_files = _client.list_artifacts(_comp_run_id, _node_dir.path)
                                    for _art in _node_files:
                                        if _art.is_dir:
                                            continue
                                        _local = _client.download_artifacts(_comp_run_id, _art.path)
                                        _dest_folder = f"companion_logs/{_node_dir.path.split('/')[-1]}"
                                        _client.log_artifact(_training_run_id, _local, _dest_folder)
                                        _copied += 1
                                break  # success
                            except Exception as _copy_e:
                                print(f"    attempt {_attempt+1}/6: {_copy_e}")
                        if _copied > 0:
                            print(f"  Copied {_copied} companion GPU log(s) to artifacts")
                        else:
                            print(f"  (Companion artifacts not available after 30s — check companion run directly)")
                    else:
                        print("  (No companion run found within time window)")
                except Exception as _e:
                    print(f"  (Could not link companion run: {_e})")
            else:
                print("  (No companion run — remote=false, using co-located GPUs)")

except Exception:
    mlflow.end_run(status="FAILED")
    raise

# COMMAND ----------

# DBTITLE 1,How to interpret run results
# MAGIC %md
# MAGIC ### How to interpret run results
# MAGIC
# MAGIC **Success criteria** (all must be true):
# MAGIC - `train/loss` metric exists in MLflow
# MAGIC - At least one `checkpoint-N/` directory with U-Net weights
# MAGIC - `training_integrity_passed = true` in MLflow params
# MAGIC
# MAGIC The training function logs `training_integrity_passed` to MLflow params.
# MAGIC
# MAGIC **What `launch_mode` tells you** (also logged to MLflow):
# MAGIC
# MAGIC | `launch_mode` value | Meaning |
# MAGIC |---|---|
# MAGIC | `distributed_remote_8xH100` | DDP on 8 H100 GPUs via `@distributed(remote=True)` |
# MAGIC | `distributed_local_8xH100` | DDP on 8 H100 GPUs via `@distributed` (co-located) |
# MAGIC | `distributed_remote_4xA10` | DDP on 4 A10 GPUs via `@distributed(remote=True)` |
# MAGIC | `single_gpu` | Single-GPU direct call (no `@distributed`) |
# MAGIC
# MAGIC **Why are there two MLflow runs?** (remote=true only)
# MAGIC
# MAGIC > With `Compute: Remote = true`, the `@distributed` runtime creates a **companion run** for per-node monitoring. This is separate from your training run and lands in an auto-generated experiment. Your training run has `companion_run_url` in its tags — click it to navigate.
# MAGIC >
# MAGIC > | Run | Experiment | Contains |
# MAGIC > |---|---|---|
# MAGIC > | **Your training run** | Your custom experiment | Params, metrics, checkpoints, samples, rank logs, companion logs |
# MAGIC > | **Runtime companion** (e.g. `burly-cub-719`) | Auto-generated | Per-node system metrics + GPU logs (`logs/node_0/gpu_N-0.chunk.txt`) |
# MAGIC >
# MAGIC > With `Remote = false`, there is only one MLflow run — no companion.
# MAGIC
# MAGIC **Distributed teardown warnings (safe to ignore):**
# MAGIC
# MAGIC > When using `@distributed(remote=True)`, notebook output may show errors *after training completes*:
# MAGIC > ```
# MAGIC > [rank1] Failed to check the "should dump" flag on TCPStore ... Broken pipe
# MAGIC > ERROR: Global rank 1 (PID 584) exited with code 1
# MAGIC > ```
# MAGIC > This is a teardown race — the driver shuts down the TCPStore before remote workers finish NCCL cleanup. Training completed normally. The MLflow run will show **FINISHED** with valid metrics.
# MAGIC >
# MAGIC > A run with MLflow status FINISHED but **zero metrics** usually means training never ran — that’s the actual failure case.
# MAGIC > However, this can also occur if, for example, you resume from a checkpoint and the requested `num_training_steps` has already been reached (no steps left to run).
# MAGIC > **Advice:** If you see a FINISHED run with zero metrics, review your checkpoint resume settings, `num_training_steps`, and logs to confirm whether training was intentionally skipped or failed to launch.
# MAGIC
# MAGIC **Inference cache + `remote=True`:**
# MAGIC
# MAGIC > When using `remote=True`, the remote worker processes may write to `/tmp` with different permissions. The inference cell's local cache (`/tmp/sd2_inference_cache`) will fall back to loading from UC Volumes if it encounters permission errors. This is slower (~30s) but works automatically. Detaching and reattaching resets `/tmp`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify checkpoints
# MAGIC Confirms checkpoint files were written. Cross-reference with `training_integrity_passed` in MLflow.

# COMMAND ----------

# DBTITLE 1,Checkpoint directory scan
from pathlib import Path

ckpt_root = Path(CHECKPOINT_PATH)
if ckpt_root.exists():
    checkpoints = sorted(ckpt_root.iterdir())
    print(f"Checkpoints at {ckpt_root}:")
    for ckpt in checkpoints:
        if ckpt.is_dir():
            unet_exists = (ckpt / "unet" / "config.json").exists()
            has_state = (ckpt / "training_metadata.json").exists()
            print(f"  {ckpt.name}: unet={unet_exists}, state={has_state}")
else:
    print(f"No checkpoints found at {ckpt_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Quick inference test with finetuned model

# COMMAND ----------

# DBTITLE 1,Note on inference cell load time
# MAGIC %md
# MAGIC > **Note:** For the first inference, allow 1–3 minutes before any output appears below due to model loading and checkpoint discovery. Subsequent runs are much faster once cached.
# MAGIC >
# MAGIC > The bottleneck is **model loading**, not inference. The cell does several heavy I/O operations before the first `print()` fires:
# MAGIC >
# MAGIC > 1. **Checkpoint discovery** — scanning the Volume directory (each file op = HTTP call via FUSE)
# MAGIC > 2. **`UNet2DConditionModel.from_pretrained`** — loading ~3.5 GB of weights from the Volume
# MAGIC > 3. **`StableDiffusionInpaintPipeline.from_pretrained`** — loads VAE, text encoder, tokenizer, and scheduler from HF cache
# MAGIC > 4. **`.to("cuda")`** — transferring everything to GPU memory
# MAGIC >
# MAGIC > The actual inference (2 samples x 50 diffusion steps) adds ~30s after that. No output until steps 1–3 complete is normal — it's FUSE overhead on the weight files.
# MAGIC >
# MAGIC > **Tip:** Copying large model files or checkpoints to local SSD (e.g., `/tmp`) before loading can significantly speed up model initialization, as FUSE-based Volumes have high per-file latency.

# COMMAND ----------

# DBTITLE 1,Load checkpoint and run Stable Diffusion inpainting on  ...
import torch
import random
import re
import gc
import shutil
import time
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from streaming import StreamingDataset
from PIL import Image
import io
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Cache model files to local SSD (avoids slow FUSE reads) ------
_LOCAL_CACHE = Path("/tmp/sd2_inference_cache")
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

def _cache_to_local(vol_path, label):
    """Copy a Volumes directory to /tmp if not already cached. Returns local path."""
    vol_path = Path(vol_path)
    local_path = _LOCAL_CACHE / vol_path.name
    try:
        if local_path.exists() and any(local_path.iterdir()):
            print(f"  {label}: using cached /tmp copy")
            return local_path
    except PermissionError:
        # Root-owned files from @distributed workers — skip cache, use Volumes directly
        print(f"  {label}: cache permission denied, loading from Volumes")
        return vol_path
    t0 = time.time()
    print(f"  {label}: copying {vol_path} -> {local_path} ...", end="", flush=True)
    try:
        shutil.copytree(str(vol_path), str(local_path), dirs_exist_ok=True)
        print(f" done ({time.time()-t0:.1f}s)")
        return local_path
    except PermissionError:
        print(f" permission denied, loading from Volumes")
        return vol_path

print("Caching model files to local SSD...")
_local_base = _cache_to_local(BASE_PIPELINE_PATH, "Base pipeline")
MODEL_NAME = str(_local_base)

# -- Find best checkpoint ------
ckpt_root = Path(CHECKPOINT_PATH)
ckpt_dir = None

if ckpt_root.exists():
    if (ckpt_root / "best" / "unet" / "config.json").exists():
        ckpt_dir = ckpt_root / "best" / "unet"
        print(f"Using best checkpoint: {ckpt_dir}")
    elif (ckpt_root / "final" / "unet" / "config.json").exists():
        ckpt_dir = ckpt_root / "final" / "unet"
        print(f"Using final checkpoint: {ckpt_dir}")
    else:
        try:
            _run_id = ckpt_root.name.split("_")[-1]
            if len(_run_id) == 32:
                _history = mlflow.tracking.MlflowClient().get_metric_history(_run_id, "val/loss")
                if _history:
                    _best = min(_history, key=lambda m: m.value)
                    _candidate = ckpt_root / f"checkpoint-{_best.step}" / "unet"
                    if (_candidate / "config.json").exists():
                        ckpt_dir = _candidate
                        print(f"Using checkpoint with best val/loss ({_best.value:.4f} at step {_best.step}): {ckpt_dir}")
        except Exception:
            pass

if ckpt_dir is None:
    print(f"No checkpoint found under {ckpt_root}")

# -- Run inference if checkpoint found ------
if ckpt_dir:
    _local_ckpt = _cache_to_local(ckpt_dir, "Checkpoint")

    t0 = time.time()
    print(f"Loading finetuned U-Net from: {_local_ckpt}")
    finetuned_unet = UNet2DConditionModel.from_pretrained(str(_local_ckpt), torch_dtype=torch.float16)
    print(f"  U-Net loaded ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print(f"Loading base pipeline from: {MODEL_NAME}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_NAME,
        unet=finetuned_unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    print(f"  Pipeline ready ({time.time()-t0:.1f}s)")

    def _decode_pil(raw, mode="RGB"):
        return (raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw))).convert(mode).resize((512, 512))

    for split_name in ["OTR_easy", "OTR_hard"]:
        ds = StreamingDataset(local=str(Path(MDS_PATH) / split_name))
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]

        input_img = _decode_pil(sample["input"], "RGB")
        mask_img = _decode_pil(sample["mask"], "L")
        target_img = _decode_pil(sample["target"], "RGB")

        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            result = pipeline(
                prompt=PROMPT,
                image=input_img,
                mask_image=mask_img,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images[0]
        print(f"  {split_name} inference: {time.time()-t0:.1f}s")

        # Compact matplotlib rendering
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=100)
        for ax, img, title in zip(
            axes,
            [input_img, mask_img, result, target_img],
            ["Input", "Mask", "Inpainted", "Target"]
        ):
            ax.imshow(np.array(img), cmap="gray" if title == "Mask" else None)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{split_name} (sample {idx})", fontsize=11)
        plt.tight_layout()
        display(fig)
        plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Note on inference quality: This model was trained for a small number of steps —      results are not representative of final quality.    
# MAGIC
# MAGIC Common artifacts at this stage:     
# MAGIC - Large text regions: model struggles to reconstruct the underlying image            
# MAGIC - Small/no text regions: model over-paints areas that should be left unchanged
# MAGIC - OTR_easy/hard split names refer to overlay text complexity, not inpainting         
# MAGIC difficulty — some "easy" samples may appear harder to inpaint than "hard" ones       
# MAGIC depending on the background                                            
# MAGIC                                                                                      
# MAGIC Quality typically improves with longer training runs and appropriate hyperparameter tuning. 

# COMMAND ----------

# DBTITLE 1,Appendix: GPU Scaling and Cost Notes
# MAGIC %md
# MAGIC    
# MAGIC ---
# MAGIC ## Appendix: GPU Scaling & Cost Notes
# MAGIC
# MAGIC **Docs:** [Multi-GPU distributed training](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training/) · [H100 starter tutorial](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-api-h100-starter/) · [AI Runtime overview](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/index/)
# MAGIC
# MAGIC ### `@distributed` pattern
# MAGIC
# MAGIC ```python
# MAGIC from serverless_gpu import distributed
# MAGIC
# MAGIC @distributed(gpus=N, gpu_type='H100', remote=True)  # or remote=False
# MAGIC def run_train():
# MAGIC     ...  # all data loading + model init MUST be inside
# MAGIC ```
# MAGIC
# MAGIC Use the `Compute: Remote` widget to toggle `remote=True` vs `remote=False`.
# MAGIC
# MAGIC > **`remote=True` is Private Preview.** Multi-node distributed training support is evolving — a CLI-based offering is expected. The `@distributed` API may change. If you find `remote=True` useful, provide feedback to Databricks — user demand helps prioritize official support.
# MAGIC
# MAGIC ### Scaling paths
# MAGIC
# MAGIC | Config | GPUs | Precision | Batch/GPU | Grad Accum | Eff. Batch | Remote | Notes |
# MAGIC |---|---|---|---|---|---|---|---|
# MAGIC | A10 x 1 | 1x A10G | fp16 | 1 | 8 | 8 | N/A | Single GPU, direct call |
# MAGIC | A10 x 4 | 4x A10G | fp16 | 1 | 8 | 32 | Required | Multi-node (4 separate machines) |
# MAGIC | H100 x 8 | 8x H100 | fp16 | 4 | 2 | 64 | Optional | Single-node NVLink. Default H100 preset. |
# MAGIC | H100 x 16 | 16x H100 | fp16 | 4 | 2 | 128 | Required | Multi-node (2 × 8 H100 nodes) |
# MAGIC
# MAGIC ### Communication topology
# MAGIC
# MAGIC | Setup | Interconnect | Gradient sync | Notes |
# MAGIC |---|---|---|---|
# MAGIC | **H100 x 8 (co-located, remote=false)** | NVLink (~900 GB/s) | Fast | Best DDP efficiency |
# MAGIC | **H100 x 8 (remote=true)** | NVLink on remote node | Fast | Companion run, but separate billing |
# MAGIC | **A10 x N (remote=true)** | Network fabric | Slower | Amortize with grad accum (x8) |
# MAGIC | **Classic multi-GPU** (e.g. g5.12xlarge) | PCIe | Medium | TorchDistributor (separate notebook) |
# MAGIC
# MAGIC For the 865M-param U-Net, each gradient sync pushes ~3.4 GB. On H100 NVLink this is negligible.
# MAGIC
# MAGIC ### Cost considerations
# MAGIC
# MAGIC | Scenario | Billing |
# MAGIC |---|---|
# MAGIC | Attached to A10, remote=true targeting H100 | A10 driver + H100 remote node |
# MAGIC | Attached to H100, remote=false | H100 only (co-located) |
# MAGIC | Attached to H100, remote=true | H100 driver + separate H100 remote node (double billing) |
# MAGIC | Attached to A10, remote=true targeting A10 x 4 | A10 driver + 4 remote A10 nodes |
# MAGIC
# MAGIC **Tip:** For same-type training (e.g. H100 → H100), use `remote=false` to avoid double billing. Use `remote=true` when you need cross-accelerator launching or companion run observability.
# MAGIC
# MAGIC ### Per-rank log collection
# MAGIC
# MAGIC | Source | Where | Notes |
# MAGIC |---|---|---|
# MAGIC | Rank 0 stdout | Notebook cell output | Only rank 0 surfaces in notebook UI |
# MAGIC | All ranks (consolidated) | Training run → `run_info/rank_N_log.txt` | Via `_TeeStream` + `gather_object` |
# MAGIC | Per-node GPU logs (remote=true) | Training run → `companion_logs/` | Copied from companion run when available |
# MAGIC | Per-node GPU logs (original) | Companion run → `logs/node_0/` | Direct access via `companion_run_url` tag |
# MAGIC
# MAGIC ### Key reminders
# MAGIC - All data loading / model init must be **inside** the `@distributed` function
# MAGIC - Max 7-day runtime — checkpointing is implemented via `SAVE_EVERY`
# MAGIC - Launch distributed execution with `func.distributed()`, not `func()`
# MAGIC - `Compute: GPU Type` sets the target — with `remote=true`, doesn't need to match attached compute
