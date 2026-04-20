# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Convert OTR to Mosaic Streaming (MDS) with Paired Augmentation
# MAGIC
# MAGIC Writes MDS shards with deterministic paired augmentations for use with
# MAGIC [MosaicML Streaming](https://github.com/mosaicml/streaming) and multi-node DDP training.
# MAGIC
# MAGIC ### Performance
# MAGIC
# MAGIC The conversion is I/O-bound (reading \~80 GB of individual PNGs, encoding JPEG/PNG,
# MAGIC writing compressed MDS shards). Multithreaded reads (8 workers) overlap file loading
# MAGIC with shard writing:
# MAGIC
# MAGIC | Compute | Time (1×, \~74K samples) | Notes |
# MAGIC |---|---|---|
# MAGIC | Serverless, single-threaded | \~10 hrs | Baseline |
# MAGIC | **Serverless, 8 I/O threads (this notebook)** | **\~1.5 hrs** | **\~7× speedup — measured** |
# MAGIC | Classic compute + local SSD | \~30–45 min (est.) | See `convert_otr_to_mds_classic` |
# MAGIC
# MAGIC With **5× augmentation** (original + 4 augmented = 370K records), expect \~7.5 hours
# MAGIC on serverless. Toggle via the `augments_per_sample` widget.
# MAGIC
# MAGIC > For faster conversion, the **classic-compute variant** (`convert_otr_to_mds_classic`)
# MAGIC > reads from Volumes and writes shards to the node's local SSD, then syncs back.
# MAGIC > Recommended config: `i3.2xlarge` single-node, ML LTS runtime, 120 min auto-termination.
# MAGIC
# MAGIC ### When is MDS recommended?   
# MAGIC <br>     
# MAGIC
# MAGIC ```
# MAGIC ① Single GPU                  ② Single node, multi-GPU         ③ Multi-node, multi-GPU
# MAGIC ─────────────────────────   ───────────────────────────────   ─────────────────────────────────
# MAGIC ┌─────────────────────┐      ┌───────────────────────────┐      ┌────────────┐  ┌────────────┐
# MAGIC │  Node               │      │  Node                     │      │  Node 1    │  │  Node 2    │
# MAGIC │  ┌─────┐            │      │  ┌─────┐ ┌─────┐ ┌─────┐  │      │ ┌───┐┌───┐ │  │ ┌───┐┌───┐ │
# MAGIC │  │ GPU │            │      │  │ GPU │ │ GPU │ │ GPU │  │      │ │GPU││GPU│ │  │ │GPU││GPU│ │
# MAGIC │  └─────┘            │      │  └─────┘ └─────┘ └─────┘  │      │ └───┘└───┘ │  │ └───┘└───┘ │
# MAGIC └─────────────────────┘      └───────────────────────────┘      └────────────┘  └────────────┘
# MAGIC                                                                         │  network  │
# MAGIC ✔ Plain PyTorch Dataset       ✔ MDS or PyTorch Dataset           ✔ MDS recommended
# MAGIC   On-the-fly augmentation       MDS adds deterministic             Deterministic sharding
# MAGIC   Simplest approach             sharding + resumption              + resumption essential
# MAGIC ```
# MAGIC
# MAGIC [MDS](https://docs.mosaicml.com/projects/streaming/en/stable/) shines for
# MAGIC [DDP](https://pytorch.org/docs/stable/notes/ddp.html) training where deterministic
# MAGIC sharding, resumption, and efficient shuffling across workers matter. 
# MAGIC
# MAGIC For single-GPU workflows (≤74K samples),
# MAGIC a plain `torch.utils.data.Dataset` reading PNGs directly from UC Volumes with
# MAGIC **on-the-fly augmentation** may be simpler:
# MAGIC
# MAGIC | | MDS (this notebook) | Direct PyTorch Dataset |
# MAGIC |---|---|---|
# MAGIC | Augmentation | Pre-computed, fixed per shard | On-the-fly, different each epoch |
# MAGIC | Flexibility | Re-shard to change strategy | Change transforms instantly |
# MAGIC | Disk overhead | 2–3× (original + shards) | Zero |
# MAGIC | Multi-GPU scaling | Excellent (deterministic sharding) | Fine with `DistributedSampler` |
# MAGIC | Resumption | Exact sample-level | Epoch-level (usually sufficient) |
# MAGIC
# MAGIC **Use MDS when:** DDP across 4+ GPUs (single node or multi-node via `TorchDistributor`),
# MAGIC or when the dataset grows to millions of samples.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Two source modes:**
# MAGIC
# MAGIC | Mode | Best for | How it works |
# MAGIC |------|----------|-------------|
# MAGIC | `"volumes"` | Already ran `download_otr_v0.2x.py` | Reads PNGs + meta from UC Volumes |
# MAGIC | `"huggingface"` | No prior download needed | Streams directly from HF parquets — re-downloads \~80 GB |
# MAGIC
# MAGIC **Output:** `OverlayTextRemoval_MDS/{split}/` (MDS shards)
# MAGIC
# MAGIC **Augmentation strategy** (when `augments_per_sample=5`):
# MAGIC * 5 versions per sample (aug_idx 0–4): original + 4 augmented
# MAGIC * Same random seed for spatial transforms on input + target + mask (preserves pixel alignment)
# MAGIC * Color jitter on input ONLY (target stays clean, mask stays binary)
# MAGIC * 74K train × 5 = 370K MDS records
# MAGIC
# MAGIC **MDS columns:**
# MAGIC
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `id` | str | Original sample ID |
# MAGIC | `aug_idx` | int | Augmentation index (0=original) |
# MAGIC | `input` | jpeg | Input image (with text, possibly augmented) |
# MAGIC | `target` | jpeg | Ground truth (text-free, spatially augmented only) |
# MAGIC | `mask` | png | Binary mask (spatially augmented only) |

# COMMAND ----------

# MAGIC %pip install -q mosaicml-streaming~=0.13.0 pillow~=12.1 datasets~=4.8 huggingface_hub~=0.36
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, uuid

# ━━━ Widget definitions ━━━
dbutils.widgets.text("catalog", "my_catalog", "Catalog")          ## update to your own Catalog
dbutils.widgets.text("schema", "my_schema", "Schema")                  ## update to your own Schema
dbutils.widgets.text("volume", "my_volume", "Volume")                ## update to your own Volume
dbutils.widgets.dropdown("source_mode", "volumes", ["huggingface", "volumes"], "Source Mode")
dbutils.widgets.dropdown("overwrite", "no", ["yes", "no"], "Overwrite")
dbutils.widgets.dropdown("augments_per_sample", "1", ["1", "5"], "Augments Per Sample")  # 1=MVP (~74K), 5=Full (~373K)
dbutils.widgets.text("mds_output_name", "OverlayTextRemoval_MDS", "MDS Output Folder")

# ━━━ Configuration ━━━
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
SOURCE_MODE = dbutils.widgets.get("source_mode")
AUGMENTS_PER_SAMPLE = int(dbutils.widgets.get("augments_per_sample"))
OVERWRITE = dbutils.widgets.get("overwrite") == "yes"

# Derived paths
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
MDS_OUTPUT_NAME = dbutils.widgets.get("mds_output_name")
MDS_OUTPUT_PATH = f"{VOLUME_PATH}/{MDS_OUTPUT_NAME}"
# Note: The finetune notebook reads from this MDS path — ensure they match.

# Serverless sets HF_DATASETS_CACHE to a stale /tmp path with bad permissions.
# Override with UUID-based path before import AND pass cache_dir explicitly to load_dataset.
# If disk space is an issue, switch to: /local_disk0/hf_cache_{uuid}
HF_CACHE = f"/tmp/hf_cache_{uuid.uuid4().hex[:8]}"
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.environ["HF_HOME"] = HF_CACHE

print(f"Catalog:            {CATALOG}")
print(f"Schema:             {SCHEMA}")
print(f"Volume:             {VOLUME}")
print(f"Volume path:        {VOLUME_PATH}")
print(f"MDS output:         {MDS_OUTPUT_PATH}")
print(f"Source mode:        {SOURCE_MODE}")
print(f"Augments/sample:    {AUGMENTS_PER_SAMPLE}")
print(f"Overwrite:          {OVERWRITE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Check source availability

# COMMAND ----------

from pathlib import Path

otr_root = Path(VOLUME_PATH) / "OverlayTextRemoval"
mds_root = Path(MDS_OUTPUT_PATH)

OTR_EXPECTED = {"train": 74_716, "OTR_easy": 5_538, "OTR_hard": 9_055}

if SOURCE_MODE == "volumes":
    print("Source mode: VOLUMES (reading PNGs from UC Volumes)")
    print("=" * 55)
    splits_ready = {}
    for split_name, expected in OTR_EXPECTED.items():
        split_dir = otr_root / split_name
        input_count = len(list((split_dir / "input").glob("*.png"))) if (split_dir / "input").exists() else 0
        ready = input_count >= expected
        splits_ready[split_name] = ready
        print(f"  {split_name}: {input_count:,}/{expected:,} {'READY' if ready else 'MISSING -- run download_otr_v0.2x.py first'}")
    if not any(splits_ready.values()):
        print("\nNo OTR data found in Volumes. Run download_otr_v0.2x.py first, or switch to SOURCE_MODE = 'huggingface'.")
elif SOURCE_MODE == "huggingface":
    print("Source mode: HUGGINGFACE (streaming directly from HF -- no Volumes PNGs needed)")
    print("=" * 55)
    splits_ready = {split: True for split in OTR_EXPECTED}
    print("  All splits available via HuggingFace datasets API")
else:
    raise ValueError(f"Unknown SOURCE_MODE: {SOURCE_MODE}. Use 'volumes' or 'huggingface'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define paired augmentation transforms
# MAGIC
# MAGIC Key constraint: spatial transforms (crop, flip, rotate) must use the SAME random seed
# MAGIC for input, target, and mask to preserve pixel alignment.
# MAGIC Color jitter is applied to input ONLY.

# COMMAND ----------

import random
from PIL import Image, ImageEnhance, ImageDraw
import io

# Pillow 10+ compatibility — deprecated top-level constants moved to enums
try:
    _FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _FLIP_LR = Image.FLIP_LEFT_RIGHT
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST


def apply_spatial_transform(img, seed, aug_idx):
    """Apply deterministic spatial transforms using a fixed seed.
    Same seed = same transform for input, target, and mask."""
    if aug_idx == 0:
        return img  # Original, no transform

    rng = random.Random(seed)
    w, h = img.size

    # Random horizontal flip (50% chance)
    if rng.random() > 0.5:
        img = img.transpose(_FLIP_LR)

    # Random rotation (-15 to +15 degrees)
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=_BILINEAR, expand=False, fillcolor=0)

    # Random crop and resize back to original size (85-100% area)
    scale = rng.uniform(0.85, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    left = rng.randint(0, w - crop_w)
    top = rng.randint(0, h - crop_h)
    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((w, h), _BILINEAR)

    return img


def apply_spatial_transform_mask(mask, seed, aug_idx):
    """Same spatial transform as above but with NEAREST resampling for binary masks."""
    if aug_idx == 0:
        return mask

    rng = random.Random(seed)
    w, h = mask.size

    if rng.random() > 0.5:
        mask = mask.transpose(_FLIP_LR)

    angle = rng.uniform(-15, 15)
    mask = mask.rotate(angle, resample=_NEAREST, expand=False, fillcolor=0)

    scale = rng.uniform(0.85, 1.0)
    crop_w = int(w * scale)
    crop_h = int(h * scale)
    left = rng.randint(0, w - crop_w)
    top = rng.randint(0, h - crop_h)
    mask = mask.crop((left, top, left + crop_w, top + crop_h))
    mask = mask.resize((w, h), _NEAREST)

    return mask


def apply_color_jitter(img, seed, aug_idx):
    """Apply color jitter to input image ONLY. Not applied to target or mask."""
    if aug_idx == 0:
        return img

    rng = random.Random(seed + 999)  # Different seed offset for color

    # Brightness (0.8 - 1.2)
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast (0.8 - 1.2)
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation (0.8 - 1.2)
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Color(img).enhance(factor)

    return img


def generate_mask_from_bboxes(bboxes, width, height):
    """Generate binary mask from word_bboxes list."""
    mask = Image.new("L", (width, height), 0)
    if bboxes and len(bboxes) > 0:
        draw = ImageDraw.Draw(mask)
        # word_bboxes is list of [x1, y1, x2, y2] lists
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Normalize — some bboxes have inverted coords
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            pad = 3
            draw.rectangle([x_min-pad, y_min-pad, x_max+pad, y_max+pad], fill=255)
    return mask


def image_to_jpeg_bytes(img):
    """Convert PIL Image to JPEG bytes for MDS storage."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def image_to_png_bytes(img):
    """Convert PIL Image to PNG bytes for MDS storage (masks)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Write MDS shards with paired augmentation
# MAGIC
# MAGIC Processes each split independently. Skips splits where MDS shards already exist.

# COMMAND ----------

# DBTITLE 1,How shard sizes and numbers are determined
# MAGIC %md
# MAGIC ### How shard sizes and numbers are determined
# MAGIC
# MAGIC Shard creation is controlled by the `size_limit` parameter in
# MAGIC [`MDSWriter`](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html):
# MAGIC
# MAGIC ```python
# MAGIC MDSWriter(
# MAGIC     out=str(mds_split_dir),
# MAGIC     columns=columns,
# MAGIC     compression="zstd",
# MAGIC     size_limit=1 << 27,  # 128 MB per shard
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC `MDSWriter` accumulates serialized records into a buffer. When the buffer reaches
# MAGIC `size_limit` bytes (128 MB here), it flushes to a new compressed shard file
# MAGIC (`shard.XXXXX.mds.zstd`). The number of shards is therefore:
# MAGIC
# MAGIC > **total serialized data ÷ size_limit ≈ number of shards**
# MAGIC
# MAGIC For this dataset:
# MAGIC
# MAGIC | Split | Total Disk | ÷ 128 MB | Shards | Avg Shard |
# MAGIC |---|---|---|---|---|
# MAGIC | train | \~11.7 GB | ≈ 91 | **116** | \~101 MB |
# MAGIC | OTR_easy | \~1.2 GB | ≈ 9 | **12** | \~98 MB |
# MAGIC | OTR_hard | \~1.6 GB | ≈ 12 | **14** | \~111 MB |
# MAGIC
# MAGIC The `size_limit` is a **pre-compression threshold** — actual on-disk shard sizes are
# MAGIC smaller thanks to `compression="zstd"`. The shard count is slightly higher than the
# MAGIC naive calculation because image sizes vary (some samples are larger, causing earlier
# MAGIC shard rollovers).
# MAGIC
# MAGIC **Tuning guidance:**
# MAGIC * **Larger shards** → fewer files, less filesystem overhead, but coarser shuffling granularity
# MAGIC * **Smaller shards** → better shuffle quality for multi-node DDP, but more file overhead
# MAGIC * 64–256 MB is typical; 128 MB is a sensible default for datasets in the 10–100 GB range
# MAGIC
# MAGIC **References:**
# MAGIC * [MosaicML Streaming — MDSWriter API](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html)
# MAGIC * [MosaicML Streaming — Shard configuration](https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_format.html)
# MAGIC * [GitHub — mosaicml/streaming](https://github.com/mosaicml/streaming)

# COMMAND ----------

# DBTITLE 1,Convert OTR to MDS with 1--Nx augmentation
from streaming import MDSWriter, StreamingDataset
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import PngImagePlugin
from tqdm.auto import tqdm
import json, time, shutil

# Raise Pillow's safety limit for PNG text metadata chunks (some OTR PNGs have large embedded metadata)
PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024 ** 2)  # 10 MB

# ── Parallelism config ──
# Image loading via FUSE is high-latency (~50–100 ms/file). Overlapping reads
# across threads hides this latency while MDSWriter.write() stays single-threaded.
NUM_WORKERS = 8  # parallel I/O threads (tune to network bandwidth; 8–16 typical for FUSE)

mds_root = Path(MDS_OUTPUT_PATH)

columns = {
    "id": "str",
    "aug_idx": "int",
    "input": "jpeg",
    "target": "jpeg",
    "mask": "png",
}

# Load HF dataset if in huggingface mode (one-time download/cache)
hf_ds = None
if SOURCE_MODE == "huggingface":
    from datasets import load_dataset
    import datasets.config
    datasets.config.HF_DATASETS_CACHE = HF_CACHE
    print("Loading OTR from HuggingFace (will cache locally after first download)...")
    hf_ds = load_dataset("cyberagent/OTR", cache_dir=HF_CACHE)
    print("Dataset loaded.\n")

def get_dir_size_mb(path: Path) -> float:
    """Total size of all files in a directory, in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)

def count_mds_shards(path: Path) -> int:
    """Count MDS shards from index.json (authoritative), fallback to glob."""
    idx = path / "index.json"
    if idx.exists():
        with open(idx) as f:
            return len(json.load(f).get("shards", []))
    return len(list(path.glob("shard.*.mds*")))

def count_mds_records(path: Path) -> int:
    """Count total records via StreamingDataset (quick — reads index only)."""
    try:
        return len(StreamingDataset(local=str(path)))
    except Exception:
        return 0

split_stats = []  # collect per-split summaries

for split_name, expected in OTR_EXPECTED.items():
    if not splits_ready.get(split_name, False):
        print(f"\n{split_name}: source data missing -- skipping")
        continue

    mds_split_dir = mds_root / split_name

    # Handle existing output directory
    if mds_split_dir.exists():
        has_shards = any(mds_split_dir.glob("shard.*"))
        if has_shards and not OVERWRITE:
            shard_count = count_mds_shards(mds_split_dir)
            records = count_mds_records(mds_split_dir)
            size_mb = get_dir_size_mb(mds_split_dir)
            print(f"\n{split_name}: MDS shards already exist ({records:,} records, {shard_count} shards, {size_mb:.0f} MB) -- skipping (set overwrite=yes to regenerate)")
            split_stats.append({"split": split_name, "status": "skipped", "records": records, "shards": shard_count, "size_mb": size_mb})
            continue
        shutil.rmtree(str(mds_split_dir))
        print(f"  Cleaned up {'existing shards' if has_shards else 'partial write'} in {mds_split_dir}")

    mds_split_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine source and count ---
    if SOURCE_MODE == "volumes":
        otr_root = Path(VOLUME_PATH) / "OverlayTextRemoval"
        input_dir = otr_root / split_name / "input"
        target_dir = otr_root / split_name / "target"
        mask_dir = otr_root / split_name / "masks"
        meta_dir = otr_root / split_name / "meta"
        sample_ids = sorted([p.stem for p in input_dir.glob("*.png")])
        total_samples = len(sample_ids)
    else:  # huggingface
        split_data = hf_ds[split_name]
        total_samples = len(split_data)

    # --- Per-sample loader: runs in thread pool ---
    # Closure captures per-split variables (sample_ids, input_dir, etc.).
    # All threads finish before the next split iteration, so no race conditions.
    def load_and_process(i):
        """Load images from source, apply augmentations, encode to bytes.
        Returns list of record dicts ready for MDSWriter.write()."""
        if SOURCE_MODE == "volumes":
            sid = sample_ids[i]
            inp = Image.open(str(input_dir / f"{sid}.png"))
            tgt = Image.open(str(target_dir / f"{sid}.png"))
            mp = mask_dir / f"{sid}.png"
            if mp.exists():
                msk = Image.open(str(mp))
            else:
                meta_path = meta_dir / f"{sid}.json"
                with open(str(meta_path), "r") as f:
                    meta = json.load(f)
                w, h = (meta["width"], meta["height"]) if "width" in meta else inp.size
                msk = generate_mask_from_bboxes(meta.get("word_bboxes", []), w, h)
        else:  # huggingface
            sample = split_data[i]
            sid = sample["id"]
            inp = sample["image"]
            tgt = sample["gt_image"]
            w, h = inp.size
            msk = generate_mask_from_bboxes(sample["word_bboxes"], w, h)

        records = []
        for aug_idx in range(AUGMENTS_PER_SAMPLE):
            seed = i * AUGMENTS_PER_SAMPLE + aug_idx
            a_inp = apply_spatial_transform(inp.copy(), seed, aug_idx)
            a_tgt = apply_spatial_transform(tgt.copy(), seed, aug_idx)
            a_msk = apply_spatial_transform_mask(msk.copy(), seed, aug_idx)
            a_inp = apply_color_jitter(a_inp, seed, aug_idx)
            records.append({
                "id": sid,
                "aug_idx": aug_idx,
                "input": image_to_jpeg_bytes(a_inp),
                "target": image_to_jpeg_bytes(a_tgt),
                "mask": image_to_png_bytes(a_msk),
            })
        return records

    total_records = total_samples * AUGMENTS_PER_SAMPLE
    print(f"\n{'='*60}")
    print(f"{split_name}: {total_samples:,} samples x {AUGMENTS_PER_SAMPLE} aug = {total_records:,} MDS records")
    print(f"  Source: {SOURCE_MODE} | Output: {mds_split_dir} | Workers: {NUM_WORKERS}")
    print(f"{'='*60}")
    start_time = time.time()

    with MDSWriter(
        out=str(mds_split_dir),
        columns=columns,
        compression="zstd",
        size_limit=1 << 27,  # 128 MB shards
    ) as writer:
        written = 0
        pbar = tqdm(total=total_samples, desc=split_name, unit="sample",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Thread pool loads + processes images in parallel;
        # pool.map yields results IN ORDER for deterministic shards.
        # Main thread writes to MDSWriter sequentially (not thread-safe).
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            for records in pool.map(load_and_process, range(total_samples)):
                for rec in records:
                    writer.write(rec)
                    written += 1
                pbar.update(1)

                # Update progress bar postfix every 500 samples
                if pbar.n % 500 == 0:
                    elapsed = time.time() - start_time
                    shard_count = count_mds_shards(mds_split_dir)
                    size_mb = get_dir_size_mb(mds_split_dir)
                    pbar.set_postfix(records=f"{written:,}", shards=shard_count, disk=f"{size_mb:.0f}MB")

        pbar.close()

    elapsed = time.time() - start_time
    shard_count = count_mds_shards(mds_split_dir)
    size_mb = get_dir_size_mb(mds_split_dir)
    rate = total_samples / elapsed

    print(f"\n  {split_name} complete:")
    print(f"    Records:    {written:,}")
    print(f"    Shards:     {shard_count}")
    print(f"    Disk:       {size_mb:,.0f} MB")
    print(f"    Time:       {elapsed / 60:.1f} min ({rate:.0f} samples/s)")

    split_stats.append({
        "split": split_name, "status": "done", "records": written,
        "shards": shard_count, "size_mb": size_mb,
        "elapsed_min": round(elapsed / 60, 1), "rate": round(rate),
    })

# --- Overall summary ---
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"{'Split':<12} {'Status':<8} {'Records':>10} {'Shards':>7} {'Size MB':>8} {'Time':>8} {'Rate':>10}")
print("-" * 70)
for s in split_stats:
    if s["status"] == "skipped":
        print(f"{s['split']:<12} {'skip':<8} {s['records']:>10,} {s['shards']:>7} {s['size_mb']:>7.0f} {'—':>8} {'—':>10}")
    else:
        print(f"{s['split']:<12} {'done':<8} {s['records']:>10,} {s['shards']:>7} {s['size_mb']:>7.0f} {s['elapsed_min']:>7.1f}m {s['rate']:>8,} s/s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify MDS shards

# COMMAND ----------

import json
from pathlib import Path
from streaming import StreamingDataset

# Verify from the REMOTE (UC Volumes) path -- confirms sync was successful
mds_root = Path(MDS_OUTPUT_PATH)

def count_mds_shards(path: Path) -> int:
    idx = path / "index.json"
    if idx.exists():
        with open(idx) as f:
            return len(json.load(f).get("shards", []))
    return len(list(path.glob("shard.*.mds*")))

def get_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)

print(f"MDS Output (UC Volumes): {mds_root}")
print("=" * 55)

PEEK_RECORDS = AUGMENTS_PER_SAMPLE + 1  # enough to see all aug indices + next sample

verify_rows = []
for split_name in OTR_EXPECTED:
    mds_split_dir = mds_root / split_name
    if not mds_split_dir.exists():
        print(f"\n  {split_name}: not created")
        continue

    shard_count = count_mds_shards(mds_split_dir)
    size_mb = get_dir_size_mb(mds_split_dir)
    index_exists = (mds_split_dir / "index.json").exists()

    try:
        ds = StreamingDataset(local=str(mds_split_dir))
        total = len(ds)
        avg_shard_mb = size_mb / shard_count if shard_count else 0
        print(f"\n  {split_name}:")
        print(f"    Shards:  {shard_count}")
        print(f"    Records: {total:,}")
        print(f"    Index:   {index_exists}")
        print(f"    Columns: {list(ds[0].keys())}")
        print(f"    First {PEEK_RECORDS} records:")
        for i in range(min(PEEK_RECORDS, total)):
            s = ds[i]
            print(f"      [{i}] id={s['id']}, aug_idx={s['aug_idx']}")
        verify_rows.append({
            "split": split_name, "records": total, "shards": shard_count,
            "size_mb": round(size_mb), "avg_shard_mb": round(avg_shard_mb),
            "index": index_exists,
        })
    except Exception as e:
        print(f"\n  {split_name}: ERROR -- {e}")

if verify_rows:
    print(f"\n{'='*70}")
    print("Verification Summary")
    print(f"{'='*70}")
    print(f"{'Split':<12} {'Records':>10} {'Shards':>8} {'Size MB':>10} {'Avg Shard':>12} {'Index':>7}")
    print("-" * 70)
    for r in verify_rows:
        print(f"{r['split']:<12} {r['records']:>10,} {r['shards']:>8} {r['size_mb']:>10,} {r['avg_shard_mb']:>10} MB {'\u2713' if r['index'] else '\u2717':>5}")
    total_rec = sum(r['records'] for r in verify_rows)
    total_shards = sum(r['shards'] for r in verify_rows)
    total_size = sum(r['size_mb'] for r in verify_rows)
    print("-" * 70)
    print(f"{'TOTAL':<12} {total_rec:>10,} {total_shards:>8} {total_size:>10,} {'':>12} {'':>7}")

print(f"\nMDS shards ready for StreamingDataset + TorchDistributor DDP training.")
print(f"\nUsage in training:")
print(f"  from streaming import StreamingDataset")
print(f"  dataset = StreamingDataset(local='{mds_root}/train', shuffle=True)")

# COMMAND ----------

# DBTITLE 1,Step 5: Visual spot-check
# MAGIC %md
# MAGIC ## Step 5: Visual spot-check
# MAGIC
# MAGIC Inspect the compiled MDS shards — shard-level metadata table, then sample
# MAGIC triplets (input → target → mask) to confirm images were encoded correctly.
# MAGIC
# MAGIC ### What the sharded data looks like
# MAGIC
# MAGIC Each split is stored as a set of **MDS shard files** (`shard.XXXXX.mds.zstd`) plus an
# MAGIC `index.json` manifest:
# MAGIC
# MAGIC ```
# MAGIC OverlayTextRemoval_MDS/
# MAGIC ├─ train/
# MAGIC │   ├─ index.json            ← manifest (shard list, column schema, record counts)
# MAGIC │   ├─ shard.00000.mds.zstd  ← ~134 MB raw, ~100–109 MB compressed (zstd)
# MAGIC │   ├─ shard.00001.mds.zstd
# MAGIC │   └─ ...                   ← 116 shards, 74,716 records total
# MAGIC ├─ OTR_easy/                 ← 12 shards, 5,538 records
# MAGIC └─ OTR_hard/                 ← 14 shards, 9,055 records
# MAGIC ```
# MAGIC
# MAGIC Each shard packs ~530–720 samples (varies by image size). `zstd` compression
# MAGIC reduces shard size by ~20–30%.
# MAGIC
# MAGIC **Per-record schema:**
# MAGIC
# MAGIC | Column | MDS Type | Decoded As | Example |
# MAGIC |--------|----------|------------|----------|
# MAGIC | `id` | str | string | `"0"` |
# MAGIC | `aug_idx` | int | integer | `0` (0 = original) |
# MAGIC | `input` | jpeg | PIL Image (RGB) | 512×512 |
# MAGIC | `target` | jpeg | PIL Image (RGB) | 512×512 |
# MAGIC | `mask` | png | PIL Image (L) | 512×512 grayscale |
# MAGIC
# MAGIC `StreamingDataset` auto-decodes `jpeg`/`png` columns to PIL Images on read —
# MAGIC no manual byte parsing needed.

# COMMAND ----------

# DBTITLE 1,Shard metadata and sample records
import json
import pandas as pd
from pathlib import Path
from streaming import StreamingDataset

mds_root = Path(MDS_OUTPUT_PATH)
SAMPLES_PER_SPLIT = 3

for split_name in OTR_EXPECTED:
    mds_split_dir = mds_root / split_name
    idx_path = mds_split_dir / "index.json"
    if not idx_path.exists():
        print(f"{split_name}: no index.json found -- skipping\n")
        continue

    with open(idx_path) as f:
        index = json.load(f)

    # ── Shard metadata table ──
    shard_rows = []
    for s in index["shards"]:
        shard_rows.append({
            "shard": s.get("raw_data", {}).get("basename", "?"),
            "samples": s.get("samples", 0),
            "raw_MB": round(s.get("raw_data", {}).get("bytes", 0) / 1e6, 1),
            "zip_MB": round(s.get("zip_data", {}).get("bytes", 0) / 1e6, 1) if s.get("zip_data") else None,
        })
    df_shards = pd.DataFrame(shard_rows)
    print(f"\n{'='*60}")
    print(f"{split_name} — {len(shard_rows)} shards, {sum(r['samples'] for r in shard_rows):,} total samples")
    print(f"{'='*60}")
    display(df_shards)

    # ── Sample records table ──
    ds = StreamingDataset(local=str(mds_split_dir))
    n = len(ds)
    indices = list(range(min(SAMPLES_PER_SPLIT, n)))
    sample_rows = []
    for i in indices:
        sample = ds[i]
        inp = sample["input"]
        sample_rows.append({
            "index": i,
            "id": sample["id"],
            "aug_idx": sample["aug_idx"],
            "input_size": f"{inp.size[0]}×{inp.size[1]}",
            "input_mode": inp.mode,
        })
    df_samples = pd.DataFrame(sample_rows)
    print(f"\nFirst {len(indices)} records:")
    display(df_samples)

# COMMAND ----------

# DBTITLE 1,Display sample triplets from MDS shards
import matplotlib.pyplot as plt
from PIL import Image
from streaming import StreamingDataset
from pathlib import Path
import random

mds_root = Path(MDS_OUTPUT_PATH)
SAMPLES_PER_SPLIT = 3

for split_name in OTR_EXPECTED:
    mds_split_dir = mds_root / split_name
    if not mds_split_dir.exists() or not (mds_split_dir / "index.json").exists():
        print(f"{split_name}: no MDS shards found -- skipping")
        continue

    ds = StreamingDataset(local=str(mds_split_dir))
    n = len(ds)
    indices = sorted(random.sample(range(n), min(SAMPLES_PER_SPLIT, n)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.5 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    fig.suptitle(f"{split_name}  ({n:,} records)", fontsize=14, fontweight="bold", y=1.01)

    for row, idx in enumerate(indices):
        sample = ds[idx]
        # StreamingDataset auto-decodes jpeg/png columns to PIL Images
        inp = sample["input"]
        tgt = sample["target"]
        msk = sample["mask"]

        axes[row][0].imshow(inp)
        axes[row][0].set_title(f"input  (id={sample['id']}, aug={sample['aug_idx']})", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].imshow(tgt)
        axes[row][1].set_title("target", fontsize=9)
        axes[row][1].axis("off")

        axes[row][2].imshow(msk, cmap="gray")
        axes[row][2].set_title("mask", fontsize=9)
        axes[row][2].axis("off")

    plt.tight_layout()
    plt.show()
    print()
