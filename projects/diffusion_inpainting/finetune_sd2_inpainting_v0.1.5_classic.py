# Databricks notebook source
# MAGIC %md
# MAGIC # Finetune SD2 Inpainting for Overlay Text Removal — Classic GPU Compute (v0.1.5)
# MAGIC
# MAGIC **Model:** `sd2-community/stable-diffusion-2-inpainting` (865M U-Net params)
# MAGIC
# MAGIC **Strategy:** DDP + gradient checkpointing + 8-bit Adam via [TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html)
# MAGIC - Memory: ~10.8 GB per GPU → comfortable on A10G (24 GB)
# MAGIC - Effective batch size: `per_gpu_batch x num_gpus x grad_accum` = 1 x 4 x 2 = 8 (default on g5.12xlarge)
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
# MAGIC **Compute:** Classic GPU clusters via [`TorchDistributor`](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html) (single-node multi-GPU DDP).
# MAGIC For serverless GPU via `@distributed`, see the separate serverless notebook.
# MAGIC
# MAGIC **MVP target:** 1K–5K steps to prove pipeline works; quality iteration later.
# MAGIC
# MAGIC ### Tested Compute Modes
# MAGIC
# MAGIC This notebook auto-detects GPU count and type. No code changes needed between cluster configs.
# MAGIC
# MAGIC | Mode | Instance | GPUs | Interconnect | Notes |
# MAGIC |---|---|---|---|---|
# MAGIC | **A10 × 4** | g5.12xlarge | 4x A10G 24GB | PCIe Gen4 | Default. Tested with this notebook. |
# MAGIC | **A10 × 8** | g5.48xlarge | 8x A10G 24GB | PCIe Gen4 | Untested — increase `num_gpus` widget |
# MAGIC | **A100 × 8** | p4d.24xlarge | 8x A100 40GB | NVSwitch | Untested — uses fp16 (same as A10/H100) |
# MAGIC
# MAGIC > **Note on per-rank logging:** TorchDistributor surfaces rank 0's stdout in the notebook cell. Non-rank-0 worker output is not forwarded. This training function uses a `_TeeStream` + `torch.distributed.gather_object()` pattern: each rank captures its own stdout, rank 0 collects them all, and they're saved as MLflow artifacts (`run_info/rank_0_log.txt` through `rank_N_log.txt`).
# MAGIC
# MAGIC > **Auth (POC only):** This notebook passes the interactive user's API token to TorchDistributor worker processes for MLflow and UC Volumes access. This is standard for development/POC but **not recommended for production or shared environments**. For production, configure a [Service Principal](https://docs.databricks.com/en/admin/users-groups/service-principals.html) on the cluster — workers inherit the cluster's identity automatically, no token passing needed.
# MAGIC
# MAGIC **Key compatibility details:**
# MAGIC - **Precision**: fp16 for all GPU types — driven by GPU type detection (see code comment on why not bf16)
# MAGIC - **Checkpoints on UC Volumes**: accessible from all GPUs on the node
# MAGIC - **Inference cell auto-detects device/dtype**: works on A10, A100, H100, or CPU-only (slow fallback)
# MAGIC - **Best checkpoint selection**: training saves `best/` on val_loss improvement; inference loads it automatically via MLflow metric query
# MAGIC - **Per-rank logs**: saved as MLflow artifacts (`run_info/rank_N_log.txt`) since notebook UI only shows rank 0
# MAGIC - See **Appendix** for topology details, scaling paths, and cost tradeoffs

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -q transformers diffusers accelerate huggingface_hub~=0.36 safetensors bitsandbytes
# MAGIC %pip install -q mosaicml-streaming~=0.13.0 pillow~=12.1 datasets~=4.8 psutil nvidia-ml-py
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Clear All Widgets to Reset Configuration

dbutils.widgets.removeAll()


# COMMAND ----------

# DBTITLE 1,Configuration (widgets + derived variables)
import os, uuid, re, torch

# ━━━ GPU detection (classic compute — auto-detect from cluster) ━━━━━━━━━
_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
_gpu_name = torch.cuda.get_device_name(0).upper() if _gpu_count > 0 else "NONE"

# Auto-detect precision and batch defaults from GPU type
# NOTE: All GPU types use fp16 — bf16 causes intermittent NaN at step 2
# in diffusion training (VAE scaling / noise scheduler alpha precision).
# The base SD2 model was originally trained in fp16.
if "A100" in _gpu_name or "H100" in _gpu_name:
    _detected_precision = "fp16"
    _detected_batch = "4"
    _detected_accum = "2"
elif "A10" in _gpu_name:
    _detected_precision = "fp16"
    _detected_batch = "1"
    _detected_accum = "2"
else:
    _detected_precision = "fp16"  # safe fallback
    _detected_batch = "1"
    _detected_accum = "2"

print(f"Detected: {_gpu_name} x {_gpu_count}")

# ━━━ Widget definitions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# -- GPU config (auto-detected, overridable) --
for _w in ["per_gpu_batch_size", "gradient_accumulation_steps", "num_gpus"]:
    try:
        dbutils.widgets.remove(_w)
    except Exception:
        pass
dbutils.widgets.dropdown("num_gpus", str(_gpu_count), ["1", "2", "4", "8"], "Number of GPUs")
dbutils.widgets.dropdown("per_gpu_batch_size", _detected_batch, ["1", "2", "4"], "Batch Size / GPU")
dbutils.widgets.dropdown("gradient_accumulation_steps", _detected_accum, ["1", "2", "4", "8", "16"], "Gradient Accum Steps")
# bf16 option disabled — while bf16 is commonly recommended for H100 in
# LLM/transformer training, this diffusion pipeline requires fp16 due to
# numerical sensitivity in VAE scaling and noise scheduler calculations.
# The base SD2 model was also trained in fp16. See GPU_PRESETS comment.
# dbutils.widgets.dropdown("mixed_precision", _detected_precision, ["fp16", "bf16"], "Mixed Precision")
dbutils.widgets.dropdown("mixed_precision", "fp16", ["fp16"], "Mixed Precision")

# -- Other training hyperparameters --
dbutils.widgets.text("learning_rate", "1e-5", "Learning Rate")
dbutils.widgets.text("warmup_steps", "50", "Warmup Steps")
dbutils.widgets.text("save_every", "50", "Save Checkpoint Every N Steps")
dbutils.widgets.text("log_every", "50", "Log Loss Every N Steps")
dbutils.widgets.text("sample_every", "50", "Generate Samples Every N Steps")
dbutils.widgets.text("val_every", "50", "Validate Every N Steps")

# ━━━ Read widgets ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
SMOKE_TEST = dbutils.widgets.get("smoke_test") == "true"
NUM_GPUS = int(dbutils.widgets.get("num_gpus"))
NUM_TRAINING_STEPS = int(dbutils.widgets.get("num_training_steps"))
_resume_val = dbutils.widgets.get("resume_from").strip()
RESUME_FROM_CHECKPOINT = _resume_val if _resume_val else None
_resume_run_val = dbutils.widgets.get("resume_run_id").strip()
RESUME_RUN_ID = _resume_run_val if _resume_run_val else None

# Hyperparameters
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
PER_GPU_BATCH_SIZE = int(dbutils.widgets.get("per_gpu_batch_size"))
GRADIENT_ACCUMULATION_STEPS = int(dbutils.widgets.get("gradient_accumulation_steps"))
MIXED_PRECISION = dbutils.widgets.get("mixed_precision")
WARMUP_STEPS = int(dbutils.widgets.get("warmup_steps"))
SAVE_EVERY = int(dbutils.widgets.get("save_every"))
LOG_EVERY = int(dbutils.widgets.get("log_every"))
SAMPLE_EVERY = int(dbutils.widgets.get("sample_every"))
VAL_EVERY = int(dbutils.widgets.get("val_every"))

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
_eff_batch = PER_GPU_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
if SMOKE_TEST:
    print("WARNING: SMOKE TEST MODE: 10 steps, minimal logging")
print(f"\n{'='*60}")
print(f"  GPU:          {_gpu_name} x {NUM_GPUS}")
print(f"  Precision:    {MIXED_PRECISION}")
print(f"  Batch/GPU:    {PER_GPU_BATCH_SIZE}")
print(f"  Grad accum:   {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Eff. batch:   {_eff_batch}  ({PER_GPU_BATCH_SIZE} x {NUM_GPUS} GPUs x {GRADIENT_ACCUMULATION_STEPS} accum)")
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

# DBTITLE 1,MLflow Setup (before training)

import mlflow

# ━━━ Auto-detect current user ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-detect current user — no hardcoded email.
_current_user = spark.sql("SELECT current_user()").first()[0]

# Widget for experiment name (just the suffix — full path built from user email)
dbutils.widgets.text("mlflow_experiment_name", "diffusion_inpainting_poc", "MLflow Experiment Name")
_experiment_name = dbutils.widgets.get("mlflow_experiment_name").strip()

# Full experiment path: /Users/<email>/<experiment_name>
MLFLOW_EXPERIMENT_PATH = f"/Users/{_current_user}/{_experiment_name}"

# ━━━ MLflow experiment setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
experiment_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH).experiment_id

print(f"User:       {_current_user}")
print(f"Experiment: {MLFLOW_EXPERIMENT_PATH}")
print(f"ID:         {experiment_id}")


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Step 1: Verify MDS data is available
# MAGIC

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

# MAGIC
# MAGIC %md
# MAGIC ## Step 2: Define the training function
# MAGIC
# MAGIC ### Mixed-Precision Dtype Strategy (`fp16`)
# MAGIC
# MAGIC | Component | Loaded dtype | Compute dtype | Notes |
# MAGIC |---|---|---|---|
# MAGIC | **VAE** | `float16` | `float16` | Frozen, inference-only |
# MAGIC | **Text Encoder** | `float16` | `float16` | Frozen, inference-only |
# MAGIC | **U-Net** | checkpoint default | `float16` via Accelerator | Trainable; `gradient_checkpointing` enabled |
# MAGIC | **U-Net inputs** | `float16` (from VAE) | `float16` | Handled by `torch.autocast` |
# MAGIC | **Loss** | — | `float32` | Both `noise_pred` and `noise` cast to `.float()` for stability |
# MAGIC | **Optimizer** | — | `float32` master weights | 8-bit AdamW via Accelerator |
# MAGIC
# MAGIC ### LR Scheduler + Gradient Accumulation
# MAGIC
# MAGIC The cosine LR scheduler's `num_warmup_steps` and `num_training_steps` are multiplied by `GRADIENT_ACCUMULATION_STEPS`. This is required because `Accelerator` with multi-GPU DDP wraps the scheduler and internally divides steps by the accumulation factor. Without this multiplication, the learning rate schedule completes prematurely on multi-GPU runs (LR drops to 0 early). On single-GPU, the extra multiplication extends warmup — harmless.
# MAGIC

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

    Runs on each GPU worker via TorchDistributor / direct call.
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
    # Flushes every write for real-time stdout forwarding
    # (TorchDistributor workers use full buffering by default, hiding logs until return).
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

    # ── Mixed precision dtype (fp16 for all GPU types — see header note on bf16) ──
    _mp_dtype = torch.float16  # fp16 for all GPU types (bf16 causes NaN in diffusion training)

    # ── Environment setup ─────────────────────────────────────
    # Ensure remote workers can reach MLflow tracking server.
    # Set as env vars so TorchDistributor worker processes can connect.
    if _MLFLOW_TRACKING_URI:
        os.environ["MLFLOW_TRACKING_URI"] = _MLFLOW_TRACKING_URI
    if _MLFLOW_TRACKING_TOKEN:
        os.environ["MLFLOW_TRACKING_TOKEN"] = _MLFLOW_TRACKING_TOKEN
    # Classic compute: workers need explicit auth for MLflow + UC Volumes access
    if _DATABRICKS_HOST:
        os.environ["DATABRICKS_HOST"] = _DATABRICKS_HOST
    if _DATABRICKS_TOKEN:
        os.environ["DATABRICKS_TOKEN"] = _DATABRICKS_TOKEN

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
    #   Single-node (A10×4 g5.12xlarge): one copy, all 4 GPUs share /tmp
    #   Single-node (A100×8 p4d): one copy, all 8 GPUs share /tmp
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
    # Driver created the run before TorchDistributor launch. We attach to it
    # here so all metrics/artifacts go to one run. _DRIVER_RUN_ID is a
    # plain string passed via module globals.
    _run_ckpt_path = CHECKPOINT_PATH  # already includes timestamp + run_id

    if is_main:
        import mlflow

        mlflow.end_run()  # clean up any stale active run

        # Attach to the driver's pre-created run
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        mlflow.enable_system_metrics_logging()
        run = mlflow.start_run(run_id=_DRIVER_RUN_ID)

        # Override source: TorchDistributor sets source to script path; fix to launch notebook
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
            "topology": f"classic single-node ({accelerator.num_processes} GPUs, PCIe)",
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

    # ── Return per-rank info ─────
    # Each rank reports its GPU stats; rank 0 adds MLflow run info.
    # TorchDistributor only returns rank 0's result, so we gather all
    # rank info on rank 0 before returning.
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

    # ── Gather rank info from all processes ──────────────
    if torch.distributed.is_initialized():
        _all_rank_info = [None] * accelerator.num_processes
        torch.distributed.gather_object(
            rank_info,
            _all_rank_info if is_main else None,
            dst=0,
        )
    else:
        _all_rank_info = [rank_info]

    if is_main:
        return _all_rank_info  # list of dicts, one per rank
    return rank_info  # non-rank-0 return ignored by TorchDistributor


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Launch training
# MAGIC
# MAGIC **Compute:** Classic GPU cluster via [TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html).
# MAGIC
# MAGIC | Compute | Backend | How |
# MAGIC |---|---|---|
# MAGIC | **Classic multi-GPU** | `TorchDistributor(local_mode=True)` | Single-node DDP on cluster GPUs |
# MAGIC | **Single GPU** (or fallback) | Direct call | Runs `train_sd2_inpainting()` directly |
# MAGIC
# MAGIC > **Note:** For production workloads, configure a Service Principal on the cluster instead of using notebook context tokens.

# COMMAND ----------

# DBTITLE 1,Resuming from a Previous Run
# MAGIC
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
# MAGIC Set these two widgets in the **Configuration** cell (cell 4) before running the launch cell:
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
# MAGIC

# COMMAND ----------

# DBTITLE 1,Monitoring & System Metrics (Classic Compute)
# MAGIC %md
# MAGIC ### Monitoring & System Metrics (Classic Compute)
# MAGIC
# MAGIC #### What MLflow tracks automatically
# MAGIC
# MAGIC `mlflow.enable_system_metrics_logging()` starts a background monitor (~10s polling) that logs GPU memory, utilization, power, CPU, RAM, and network to the active MLflow run. On classic compute, rank 0's process sees **all GPUs on the node** via pynvml — so a single-node 4xA10 cluster logs all 4 GPUs.
# MAGIC
# MAGIC #### Per-rank logs
# MAGIC
# MAGIC TorchDistributor surfaces rank 0 stdout in the notebook cell. Non-rank-0 worker stdout is not forwarded to the notebook. The training function uses `_TeeStream` + `torch.distributed.gather_object()` to collect all ranks' logs and save them as MLflow artifacts (`run_info/rank_N_log.txt`).
# MAGIC
# MAGIC #### No companion run
# MAGIC
# MAGIC Unlike serverless `@distributed`, classic compute does **not** create a companion MLflow run. All metrics and artifacts are in your single training run.

# COMMAND ----------

# DBTITLE 1,Launch training with unified MLflow run (TorchDistributor)
import torch, os, gc, json
from pathlib import Path
from datetime import datetime

num_gpus = NUM_GPUS
_gpu_name_launch = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

print(f"GPU: {_gpu_name_launch} x {num_gpus}")
print(f"Mixed precision: {MIXED_PRECISION}")

# -- Clean up MLflow state before TorchDistributor ----
import mlflow
mlflow.end_run()
mlflow.autolog(disable=True)
_KEEP_MLFLOW = {"MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_INSECURE_TLS"}
for _k in list(os.environ):
    if _k.startswith("MLFLOW_") and _k not in _KEEP_MLFLOW:
        del os.environ[_k]

gc.collect()
torch.cuda.empty_cache()

# -- Use pre-downloaded pipeline from UC Volumes ---
if Path(BASE_PIPELINE_PATH).is_dir() and (Path(BASE_PIPELINE_PATH) / "model_index.json").exists():
    MODEL_NAME = BASE_PIPELINE_PATH
    print(f"\nUsing cached pipeline from UC Volumes: {MODEL_NAME}")
else:
    print(f"\nWARNING: Cached pipeline not found at {BASE_PIPELINE_PATH}, will download from HF: {MODEL_NAME}")

# -- Capture notebook identity for MLflow source tags --
try:
    _nb_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _LAUNCH_NOTEBOOK_PATH = _nb_ctx.notebookPath().get()
    _LAUNCH_NOTEBOOK_ID = str(_nb_ctx.tags().get("notebookId").get())
except Exception:
    _LAUNCH_NOTEBOOK_PATH, _LAUNCH_NOTEBOOK_ID = "", ""

# -- Get auth credentials for TorchDistributor worker subprocesses ---
# Workers don't inherit the notebook's Databricks auth context. We capture
# the workspace URL and API token here so the training function can set them
# as env vars on each worker.
# NOTE: For production workloads, use a Service Principal configured on the
# cluster instead of notebook context tokens.
_nb_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_DATABRICKS_HOST = "https://" + _nb_ctx.browserHostName().get()
_DATABRICKS_TOKEN = _nb_ctx.apiToken().get()

# Set up base checkpoint path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH = f"{CHECKPOINT_BASE}/{timestamp}"

# -- Create MLflow run on DRIVER before TorchDistributor ----
# The driver creates the run here so:
#   1. The run ID is a plain string (safe for subprocesses)
#   2. All workers attach to the SAME run (no duplicate runs)
#   3. System metrics on the driver capture all local GPUs
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
mlflow.enable_system_metrics_logging()
_run_name = "sd_inpainting_finetune"
if SMOKE_TEST:
    _run_name += "_smoke"
_run_name += f"_classic_{timestamp}"
_driver_run = mlflow.start_run(run_name=_run_name)
_DRIVER_RUN_ID = _driver_run.info.run_id
_DRIVER_EXPERIMENT_ID = _driver_run.info.experiment_id
# NOTE: keep the run OPEN — training function on rank 0 will reattach.

# Checkpoint path includes run_id
CHECKPOINT_PATH = f"{CHECKPOINT_PATH}_{_DRIVER_RUN_ID}"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"\nMLflow run: {_run_name}")
print(f"Run ID: {_DRIVER_RUN_ID}")
print(f"Checkpoints: {CHECKPOINT_PATH}")

# -- Inject config into module globals for training function ---
# TorchDistributor runs training_fn in a subprocess that inherits the
# module's global namespace. We set these globals so the training function
# can read them directly — no cloudpickle or config dict needed.
# Classic compute: MLFLOW_TRACKING_URI is "databricks" (symbolic, relies on
# notebook auth that workers don't have). Replace with the actual workspace URL
# so workers can connect using explicit host+token.
_MLFLOW_TRACKING_URI = _DATABRICKS_HOST  # e.g. "https://<your-workspace>.cloud.databricks.com"
_MLFLOW_TRACKING_TOKEN = _DATABRICKS_TOKEN
DETECTED_GPU = _gpu_name_launch


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


# -- Post-training summary ---
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
    if num_gpus > 1:
        from pyspark.ml.torch.distributor import TorchDistributor

        def _train_wrapper():
            _patch_gradient_checkpoint()
            return train_sd2_inpainting()

        print(f"Launching {num_gpus}x GPU via TorchDistributor (local_mode=True)...")
        distributor = TorchDistributor(
            num_processes=num_gpus,
            local_mode=True,
            use_gpu=True,
        )
        result = distributor.run(_train_wrapper)

        # TorchDistributor returns rank 0's result only.
        # Our training function gathers all rank info on rank 0,
        # so result is already a list of dicts (one per rank).
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            results = result
        elif isinstance(result, dict):
            results = [result]
        else:
            results = result

    else:
        print("Running on single GPU...")
        _patch_gradient_checkpoint()
        result = train_sd2_inpainting()
        assert result is not None, "Training returned None"
        # Single GPU: result is a list with one dict (from gather)
        if isinstance(result, list):
            results = result
        else:
            results = [result]

    _print_run_summary(results)

    # Adopt run on driver so notebook UI shows "Logged 1 run" link
    if results:
        rank0 = results[0] if results else None
        if rank0 and "run_id" in rank0:
            _experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
            os.environ["MLFLOW_EXPERIMENT_ID"] = _experiment.experiment_id
            os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_PATH

            # End the driver's active run before re-opening for adoption.
            mlflow.end_run()

            with mlflow.start_run(run_id=rank0["run_id"]):
                if _LAUNCH_NOTEBOOK_PATH:
                    mlflow.set_tag("mlflow.source.name", _LAUNCH_NOTEBOOK_PATH)
                    mlflow.set_tag("mlflow.source.type", "NOTEBOOK")
                    mlflow.set_tag("mlflow.databricks.notebookPath", _LAUNCH_NOTEBOOK_PATH)
                    mlflow.set_tag("mlflow.databricks.notebookID", _LAUNCH_NOTEBOOK_ID)
                mlflow.log_param("launch_mode", f"classic_local_{num_gpus}xGPU")

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
# MAGIC **What `launch_mode` tells you** (logged to MLflow):
# MAGIC
# MAGIC | `launch_mode` value | Meaning |
# MAGIC |---|---|
# MAGIC | `classic_local_4xGPU` | DDP via TorchDistributor on 4 GPUs |
# MAGIC | `classic_local_8xGPU` | DDP via TorchDistributor on 8 GPUs |
# MAGIC | `single_gpu` | Single-GPU mode |
# MAGIC
# MAGIC **Only one MLflow run:** Unlike serverless `@distributed`, classic compute does not create a companion run. All your metrics, artifacts, and system metrics are in a single run.

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Step 4: Verify checkpoints
# MAGIC
# MAGIC Confirms checkpoint files were written. Cross-reference with `training_integrity_passed` in MLflow.
# MAGIC

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

# MAGIC
# MAGIC %md
# MAGIC ## Step 5: Quick inference test with finetuned model
# MAGIC

# COMMAND ----------

# DBTITLE 1,Note on inference cell load time
# MAGIC
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
# MAGIC

# COMMAND ----------

# DBTITLE 1,[Check] Set Checkpoint Path Using RESUME_RUN_ID Variable

#### if needed update to point to the appropriate RESUME_RUN_ID
## RESUME_RUN_ID
### Set Checkpoint Path Based on Resume Run ID
# CHECKPOINT_PATH = f"/Volumes/my_catalog/my_schema/diffusion/checkpoints/{YYYYMMDD_HHMMSS_RESUME_RUN_ID}"


# COMMAND ----------

# DBTITLE 1,Clear Temporary Inference Cache Directory
# NOTE: The inference cell caches model files to /tmp for faster loading.
# If you see blank/corrupted output after a failed or NaN training run,                  
# the cache may hold stale weights. Clear it and rerun:                                  
#                                                                                        
# import shutil
# shutil.rmtree("/tmp/sd2_inference_cache", ignore_errors=True)
# print("Cleared inference cache")

# COMMAND ----------

# DBTITLE 1,Load checkpoint and run Stable Diffusion inpainting on  ...

import torch, random, re, gc, shutil, time
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from streaming import StreamingDataset
from PIL import Image
import io, numpy as np
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
    if local_path.exists() and any(local_path.iterdir()):
        print(f"  {label}: using cached /tmp copy")
        return local_path
    t0 = time.time()
    print(f"  {label}: copying {vol_path} -> {local_path} ...", end="", flush=True)
    shutil.copytree(str(vol_path), str(local_path), dirs_exist_ok=True)
    print(f" done ({time.time()-t0:.1f}s)")
    return local_path

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
        for ax, img, title in zip(axes,
            [input_img, mask_img, result, target_img],
            ["Input", "Mask", "Inpainted", "Target"]):
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

# DBTITLE 1,Appendix: Classic GPU Scaling and Cost Notes
# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC ## Appendix: Classic GPU Scaling & Cost Notes
# MAGIC
# MAGIC **Docs:** [TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html) · [GPU-enabled clusters](https://docs.databricks.com/en/clusters/gpu.html)
# MAGIC
# MAGIC ### TorchDistributor pattern
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.ml.torch.distributor import TorchDistributor
# MAGIC
# MAGIC distributor = TorchDistributor(
# MAGIC     num_processes=4,      # number of GPU workers
# MAGIC     local_mode=True,      # single-node (all GPUs on driver)
# MAGIC     use_gpu=True,
# MAGIC )
# MAGIC result = distributor.run(train_function)
# MAGIC ```
# MAGIC
# MAGIC ### Classic GPU instance types (AWS)
# MAGIC
# MAGIC | Instance | GPUs | GPU Type | VRAM | Interconnect | Notes |
# MAGIC |---|---|---|---|---|---|
# MAGIC | `g5.12xlarge` | 4x A10G | 24 GB each | PCIe Gen4 | Tested with this notebook |
# MAGIC | `g5.48xlarge` | 8x A10G | 24 GB each | PCIe Gen4 | Untested — double the GPUs |
# MAGIC | `p4d.24xlarge` | 8x A100 | 40 GB each | NVSwitch 600 GB/s | Untested — uses fp16 |
# MAGIC | `p4de.24xlarge` | 8x A100 | 80 GB each | NVSwitch 600 GB/s | Untested — larger VRAM |
# MAGIC
# MAGIC ### Scaling paths (classic)
# MAGIC
# MAGIC | Config | GPUs | Precision | Batch/GPU | Grad Accum | Eff. Batch | Notes |
# MAGIC |---|---|---|---|---|---|---|
# MAGIC | g5.12xlarge | 4x A10G | fp16 | 1 | 2 | 8 | Default preset for this notebook |
# MAGIC | g5.48xlarge | 8x A10G | fp16 | 1 | 1 | 8 | Drop accum with more GPUs |
# MAGIC | p4d.24xlarge | 8x A100 | fp16 | 4 | 2 | 64 | NVSwitch DDP |
# MAGIC
# MAGIC ### Auth for worker processes
# MAGIC
# MAGIC TorchDistributor spawns worker processes on the same node. These workers don't inherit the notebook's
# MAGIC Databricks auth context. The training function sets `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as env vars
# MAGIC on each worker so they can access MLflow and UC Volumes.
# MAGIC
# MAGIC **For production:** Configure a Service Principal on the cluster. Workers inherit the cluster's
# MAGIC identity automatically — no token passing needed.
# MAGIC
# MAGIC ### Key differences from serverless
# MAGIC
# MAGIC | Aspect | Serverless (`@distributed`) | Classic (`TorchDistributor`) |
# MAGIC |---|---|---|
# MAGIC | GPU provisioning | On-demand via decorator | Pre-provisioned cluster |
# MAGIC | Serialization | cloudpickle (requires `__globals__` stripping) | Subprocess (no pickle issues) |
# MAGIC | Worker packages | Installed inside wrapper function | Cluster-scoped `%pip install` |
# MAGIC | Auth | Automatic (serverless token) | Explicit (notebook context token or SP) |   
# MAGIC | Companion MLflow run | Yes (runtime monitoring) | No |
# MAGIC | Inter-GPU link | NVLink (H100) or network (A10 remote) | PCIe (A10 g5) or NVSwitch (A100  p4d)   |
# MAGIC | Per-rank logs | Forwarded by runtime + `_TeeStream` | `_TeeStream` + `gather_object` only |
# MAGIC
# MAGIC ### Key reminders
# MAGIC - All data loading / model init must be **inside** the training function
# MAGIC - Checkpointing is implemented via `SAVE_EVERY` — long runs should set this appropriately
# MAGIC - TorchDistributor `local_mode=True` means all GPUs are on the driver node
