# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Download [OTR (Overlay Text Removal)](https://huggingface.co/datasets/cyberagent/OTR) to UC Volumes
# MAGIC
# MAGIC Downloads from HuggingFace `cyberagent/OTR` (~80 GB parquet).
# MAGIC Purpose-built for text removal with paired ground truth — ready for inpainting training.
# MAGIC
# MAGIC **Dataset schema:**
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `id` | string | Sample identifier |
# MAGIC | `image` | Image | Input image (with overlay text) |
# MAGIC | `gt_image` | Image | Ground truth (text-free original) |
# MAGIC | `class` | string | Object class |
# MAGIC | `words` | list[string] | Text words rendered on image |
# MAGIC | `word_bboxes` | list[int] (N x 4) | Bounding boxes [x1,y1,x2,y2] per word |
# MAGIC
# MAGIC **Splits:**
# MAGIC | Split | Size |
# MAGIC |-------|------|
# MAGIC | `train` | 74,716 |
# MAGIC | `OTR_easy` | 5,538 |
# MAGIC | `OTR_hard` | 9,055 |
# MAGIC
# MAGIC **Output structure:** `OTR/{split}/{input,target,masks,meta}/`
# MAGIC
# MAGIC **Pipeline:** HuggingFace parquet -> extract paired PNGs + generate masks from bboxes -> UC Volume
# MAGIC
# MAGIC **Compute:** Runs on serverless or classic compute. Tested on serverless (~73 min total).
# MAGIC The MDS conversion step (`convert_otr_to_mds`) has a separate classic-compute variant for faster I/O.

# COMMAND ----------

# MAGIC %pip install -qU huggingface_hub<1.0 datasets pillow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

# ━━━ Widget definitions ━━━
dbutils.widgets.text("catalog", "my_catalog", "Catalog")          ## update to your own Catalog
dbutils.widgets.text("schema", "my_schema", "Schema")                  ## update to your own Schema
dbutils.widgets.text("volume", "my_volume", "Volume")                ## update to your own Volume
dbutils.widgets.text("output_folder", "OverlayTextRemoval", "Output Folder Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
OUTPUT_FOLDER = dbutils.widgets.get("output_folder")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Component check — skip completed splits

# COMMAND ----------

from pathlib import Path

otr_root = Path(VOLUME_PATH) / OUTPUT_FOLDER

OTR_EXPECTED = {"train": 74_716, "OTR_easy": 5_538, "OTR_hard": 9_055}
otr_download_complete = True
otr_masks_complete = True
otr_status = {}

for split_name, expected in OTR_EXPECTED.items():
    split_dir = otr_root / split_name
    input_count = len(list((split_dir / "input").glob("*.png"))) if (split_dir / "input").exists() else 0
    target_count = len(list((split_dir / "target").glob("*.png"))) if (split_dir / "target").exists() else 0
    mask_count = len(list((split_dir / "masks").glob("*.png"))) if (split_dir / "masks").exists() else 0
    dl_done = input_count >= expected and target_count >= expected
    masks_done = mask_count >= expected
    otr_status[split_name] = {
        "input": input_count, "target": target_count,
        "masks": mask_count, "expected": expected,
        "download_complete": dl_done, "masks_complete": masks_done,
        "complete": dl_done and masks_done,
    }
    if not dl_done:
        otr_download_complete = False
    if not masks_done:
        otr_masks_complete = False

otr_complete = otr_download_complete and otr_masks_complete

print("OTR Component Status")
print("=" * 55)
for split_name, s in otr_status.items():
    dl = "DONE" if s["download_complete"] else f"INCOMPLETE ({s['input']}/{s['expected']})"
    masks = "DONE" if s["masks_complete"] else f"INCOMPLETE ({s['masks']}/{s['expected']})"
    print(f"  {split_name}: download={dl}, masks={masks}")

if otr_complete:
    print(f"\nAll OTR splits fully processed at {otr_root}")
elif otr_download_complete:
    print(f"\nDownload complete. Run mask generation next.")

# COMMAND ----------

# MAGIC %md
# MAGIC    
# MAGIC ## Step 2: Download paired images + metadata to Volumes (threaded)
# MAGIC
# MAGIC Saves `input.png`, `target.png`, and `meta.json` per sample. Mask generation is Step 3.
# MAGIC Resume-safe — skips completed splits and existing files (when `FRESH_RUN = False`).
# MAGIC
# MAGIC **Configuration toggles:**
# MAGIC | Toggle | Default | Effect |
# MAGIC |--------|---------|--------|
# MAGIC | `USE_RAW_BYTES` | `True` | Writes HF-stored PNG bytes directly — no PIL decode/encode |
# MAGIC | `FRESH_RUN` | `True` | Skips `Path.exists()` checks, saves ~178K FUSE `stat` calls |
# MAGIC | `NUM_WORKERS` | 48 (raw) / 16 (PIL) | Thread count for FUSE write concurrency |
# MAGIC
# MAGIC Raw-bytes mode also stores `width`/`height` in `meta.json` so Step 3 skips reopening images.
# MAGIC
# MAGIC **Why threading over multiprocessing?**
# MAGIC | Factor | Threading | Multiprocessing |
# MAGIC |--------|-----------|------------------|
# MAGIC | FUSE writes | Releases GIL — true I/O parallelism | Same FUSE throughput, no advantage |
# MAGIC | Arrow dataset | Shared in-memory (zero cost) | Requires ~80 GB copy or per-sample serialization |
# MAGIC | Per-sample overhead | Direct pointer (free) | Must pickle ~500 KB image bytes |
# MAGIC | Task creation | ~microseconds | ~milliseconds |
# MAGIC
# MAGIC The bottleneck is FUSE write throughput (~39 MB/s), not CPU.
# MAGIC FUSE syscalls release the GIL, so 48 threads achieve true parallelism without multiprocessing overhead.
# MAGIC Multiprocessing would only help if CPU-bound work (e.g. PIL encoding) dominated — raw-bytes mode eliminates that.
# MAGIC
# MAGIC **Notes:**
# MAGIC - `split[i]` random access into HF Arrow tables is fast and thread-safe
# MAGIC - The `_split=split` default-arg trick avoids closure rebinding
# MAGIC - Progress logs include elapsed time, samples/s rate, and ETA
# MAGIC - HF progress bars disabled to avoid notebook rendering overhead (175 concurrent widgets)

# COMMAND ----------

# DBTITLE 1,Threaded OTR download  + raw_bytes toggle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, os, struct, uuid, time

# ── Configuration ──────────────────────────────────────────────────
USE_RAW_BYTES = True   # True  = write HF-stored bytes directly (skip PIL encode)
                       # False = PIL decode → re-encode as PNG (original path)
FRESH_RUN = True       # True  = skip Path.exists() checks (saves ~178K FUSE stats)
                       # False = check for existing files (resume-safe)
NUM_WORKERS = 48 if USE_RAW_BYTES else 16  # raw bytes = pure I/O, no GIL → more threads

VOLUME_PATH = "/Volumes/my_catalog/my_schema/my_volume"
otr_root = Path(VOLUME_PATH) / OUTPUT_FOLDER

# ── HF cache directory ─────────────────────────────────────────────
# /tmp is fast but has limited space (~55 GB on serverless). The OTR dataset
# needs ~155 GB for parquet download + Arrow cache. If the download fails with
# a disk space error (OSError / No space left on device), switch to:
#   HF_CACHE = f"/local_disk0/hf_cache_{uuid.uuid4().hex[:8]}"
# /local_disk0 has elastic storage on serverless and won't run out of space.
# Trade-off: /local_disk0 may be slightly slower than /tmp for random reads.
HF_CACHE = f"/tmp/hf_cache_{uuid.uuid4().hex[:8]}"
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = HF_CACHE
os.environ["HF_HOME"] = HF_CACHE

# ── Helpers ────────────────────────────────────────────────────────

def _png_dims(data: bytes):
    """Width, height from PNG IHDR chunk — no PIL needed."""
    if data[:4] == b"\x89PNG":
        return struct.unpack(">II", data[16:24])
    return None


def _img_dims(data: bytes):
    """Image dimensions; fast-path for PNG, PIL lazy-open fallback."""
    dims = _png_dims(data)
    if dims:
        return dims
    from PIL import Image
    from io import BytesIO
    with Image.open(BytesIO(data)) as img:
        return img.size


def save_one_raw(sample, split_dir):
    """Write HF-stored bytes directly — no decode/encode. Thread-safe."""
    sid = sample["id"]
    inp = split_dir / "input" / f"{sid}.png"
    tgt = split_dir / "target" / f"{sid}.png"

    if not FRESH_RUN and inp.exists() and tgt.exists():
        return False

    inp_bytes = sample["image"]["bytes"]
    tgt_bytes = sample["gt_image"]["bytes"]
    with open(str(inp), "wb") as f:
        f.write(inp_bytes)
    with open(str(tgt), "wb") as f:
        f.write(tgt_bytes)

    # Store dimensions in meta so Step 3 never needs to reopen images
    w, h = _img_dims(inp_bytes)
    meta_path = split_dir / "meta" / f"{sid}.json"
    with open(str(meta_path), "w") as f:
        json.dump({
            "id": sid, "class": sample.get("class", ""),
            "words": sample.get("words", []),
            "word_bboxes": sample.get("word_bboxes", []),
            "width": w, "height": h,
        }, f)
    return True


def save_one_pil(sample, split_dir):
    """PIL decode → re-encode as PNG. Thread-safe."""
    from PIL import PngImagePlugin
    PngImagePlugin.MAX_TEXT_CHUNK = 0

    sid = sample["id"]
    inp = split_dir / "input" / f"{sid}.png"
    tgt = split_dir / "target" / f"{sid}.png"

    if not FRESH_RUN and inp.exists() and tgt.exists():
        return False

    sample["image"].save(str(inp))
    sample["gt_image"].save(str(tgt))

    meta_path = split_dir / "meta" / f"{sid}.json"
    with open(str(meta_path), "w") as f:
        json.dump({
            "id": sid, "class": sample.get("class", ""),
            "words": sample.get("words", []),
            "word_bboxes": sample.get("word_bboxes", []),
        }, f)
    return True


# ── Main ───────────────────────────────────────────────────────────
save_fn = save_one_raw if USE_RAW_BYTES else save_one_pil
timing = {}

if otr_download_complete:
    print(f"All OTR images already saved at {otr_root} \u2014 skipping download")
else:
    # Disable HF progress bars — 175 concurrent widgets cause significant
    # notebook rendering overhead and can slow down the download.
    import datasets.utils.logging
    datasets.utils.logging.disable_progress_bar()

    from datasets import load_dataset
    import datasets.config
    datasets.config.HF_DATASETS_CACHE = HF_CACHE

    print(f"HF cache: {HF_CACHE}")
    print(f"Mode: {'raw bytes (no PIL encode)' if USE_RAW_BYTES else 'PIL decode \u2192 PNG encode'}")
    print(f"Fresh run: {FRESH_RUN} (skip existence checks)")
    print(f"Workers: {NUM_WORKERS}")
    print("Loading OTR dataset from HuggingFace (cyberagent/OTR)...")
    print("This downloads ~80 GB of parquet files \u2014 may take a while.\n")

    t0_load = time.time()
    ds = load_dataset("cyberagent/OTR", cache_dir=HF_CACHE)
    t_load = time.time() - t0_load
    timing["hf_load"] = t_load
    print(f"HF load: {t_load/60:.1f} min\n")

    if USE_RAW_BYTES:
        from datasets import Image as HFImage
        for col in ("image", "gt_image"):
            ds = ds.cast_column(col, HFImage(decode=False))

    t0_total = time.time()
    for split_name in ds:
        split = ds[split_name]
        split_dir = otr_root / split_name
        for sub in ("input", "target", "meta"):
            (split_dir / sub).mkdir(parents=True, exist_ok=True)

        s = otr_status.get(split_name, {})
        if s.get("download_complete", False):
            print(f"{split_name}: already complete ({s['input']:,} samples) \u2014 skipping")
            continue

        total = len(split)
        saved = skipped = 0
        print(f"{split_name}: {total:,} samples ({s.get('input', 0):,} already saved)")

        def process_index(i, _split=split, _dir=split_dir):
            return save_fn(_split[i], _dir)

        t0_split = time.time()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i, was_saved in enumerate(executor.map(process_index, range(total))):
                if was_saved:
                    saved += 1
                else:
                    skipped += 1
                if (i + 1) % 5_000 == 0:
                    elapsed = time.time() - t0_split
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate / 60
                    pct = (i + 1) / total * 100
                    print(f"  {i+1:,}/{total:,} ({pct:.0f}%) \u2014 {saved:,} new, {skipped:,} skipped  [{elapsed/60:.1f} min, {rate:.0f} samples/s, ETA {eta:.1f} min]")

        t_split = time.time() - t0_split
        timing[split_name] = t_split
        print(f"  {split_name} complete: {saved:,} saved, {skipped:,} skipped  [{t_split/60:.1f} min]")

    t_total = time.time() - t0_total
    timing["save_total"] = t_total

    print(f"\nOTR download complete at {otr_root}")
    print(f"\n{'='*55}")
    print(f"TIMING SUMMARY")
    print(f"{'='*55}")
    print(f"  HF load (download + Arrow):  {timing.get('hf_load', 0)/60:.1f} min")
    for split_name in ds:
        if split_name in timing:
            print(f"  {split_name:30s}  {timing[split_name]/60:.1f} min")
    print(f"  {'Save total':30s}  {t_total/60:.1f} min")
    print(f"  {'GRAND TOTAL':30s}  {(timing.get('hf_load', 0) + t_total)/60:.1f} min")
    print(f"{'='*55}")

# COMMAND ----------

# DBTITLE 1,timing refs
# HF load: 18.1 min

# OTR_easy: 5,538 samples (0 already saved)
#   5,000/5,538 (90%) — 5,000 new, 0 skipped  [3.2 min, 26 samples/s, ETA 0.3 min]
#   OTR_easy complete: 5,538 saved, 0 skipped  [3.5 min]
# OTR_hard: 9,055 samples (0 already saved)
#   5,000/9,055 (55%) — 5,000 new, 0 skipped  [3.1 min, 27 samples/s, ETA 2.5 min]
#   OTR_hard complete: 9,055 saved, 0 skipped  [5.7 min]
# train: 74,716 samples (0 already saved)
#   5,000/74,716 (7%) — 5,000 new, 0 skipped  [3.1 min, 27 samples/s, ETA 43.5 min]
#   10,000/74,716 (13%) — 10,000 new, 0 skipped  [6.2 min, 27 samples/s, ETA 40.3 min]
#   15,000/74,716 (20%) — 15,000 new, 0 skipped  [9.3 min, 27 samples/s, ETA 37.2 min]
#   20,000/74,716 (27%) — 20,000 new, 0 skipped  [12.5 min, 27 samples/s, ETA 34.1 min]
#   25,000/74,716 (33%) — 25,000 new, 0 skipped  [15.4 min, 27 samples/s, ETA 30.7 min]
#   30,000/74,716 (40%) — 30,000 new, 0 skipped  [18.5 min, 27 samples/s, ETA 27.6 min]
#   35,000/74,716 (47%) — 35,000 new, 0 skipped  [21.5 min, 27 samples/s, ETA 24.4 min]
#   40,000/74,716 (54%) — 40,000 new, 0 skipped  [24.5 min, 27 samples/s, ETA 21.3 min]
#   45,000/74,716 (60%) — 45,000 new, 0 skipped  [27.5 min, 27 samples/s, ETA 18.1 min]
#   50,000/74,716 (67%) — 50,000 new, 0 skipped  [30.6 min, 27 samples/s, ETA 15.1 min]
#   55,000/74,716 (74%) — 55,000 new, 0 skipped  [33.5 min, 27 samples/s, ETA 12.0 min]
#   60,000/74,716 (80%) — 60,000 new, 0 skipped  [36.5 min, 27 samples/s, ETA 9.0 min]
#   65,000/74,716 (87%) — 65,000 new, 0 skipped  [39.5 min, 27 samples/s, ETA 5.9 min]
#   70,000/74,716 (94%) — 70,000 new, 0 skipped  [42.5 min, 27 samples/s, ETA 2.9 min]
#   train complete: 74,716 saved, 0 skipped  [45.4 min]

# OTR download complete at /Volumes/my_catalog/my_schema/my_volume/OverlayTextRemoval

# =======================================================
# TIMING SUMMARY
# =======================================================
#   HF load (download + Arrow):  18.1 min
#   OTR_easy                        3.5 min
#   OTR_hard                        5.7 min
#   train                           45.4 min
#   Save total                      54.5 min
#   GRAND TOTAL                     72.7 min
# =======================================================

# COMMAND ----------

# DBTITLE 1,Step 3 header
# MAGIC %md
# MAGIC    
# MAGIC ## Step 3: Generate binary masks from word_bboxes
# MAGIC
# MAGIC Reads `meta/*.json` for each split and draws filled rectangles for each bbox with 3 px padding.
# MAGIC If `width`/`height` are present in meta (raw-bytes mode), uses them directly — no FUSE read of input images.
# MAGIC Falls back to opening the input image for dimensions otherwise (PIL mode).
# MAGIC Threaded (16 workers) for FUSE I/O overlap. Resume-safe — skips existing masks.

# COMMAND ----------

# DBTITLE 1,Generate binary masks from word_bboxes
from pathlib import Path
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
import json

otr_root = Path(VOLUME_PATH) / OUTPUT_FOLDER
NUM_WORKERS = 16


def generate_mask(meta_path, split_dir):
    """Generate binary mask from word_bboxes in meta JSON. Thread-safe."""
    with open(str(meta_path), "r") as f:
        meta = json.load(f)

    sample_id = meta["id"]
    mask_path = split_dir / "masks" / f"{sample_id}.png"

    if mask_path.exists():
        return False

    # Use stored dimensions if available (raw-bytes mode), else open input image
    if "width" in meta and "height" in meta:
        w, h = meta["width"], meta["height"]
    else:
        input_path = split_dir / "input" / f"{sample_id}.png"
        with Image.open(str(input_path)) as img:
            w, h = img.size

    # Generate binary mask
    mask = Image.new("L", (w, h), 0)
    bboxes = meta.get("word_bboxes", [])
    if bboxes:
        draw = ImageDraw.Draw(mask)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Normalize — some bboxes have inverted coords
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            pad = 3
            draw.rectangle([x_min - pad, y_min - pad, x_max + pad, y_max + pad], fill=255)
    mask.save(str(mask_path))
    return True


if otr_masks_complete:
    print(f"All OTR masks already generated at {otr_root} \u2014 skipping")
else:
    for split_name, s in otr_status.items():
        split_dir = otr_root / split_name
        meta_dir = split_dir / "meta"
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)

        if s.get("masks_complete", False):
            print(f"{split_name}: masks already complete ({s['masks']:,}) \u2014 skipping")
            continue

        meta_files = sorted(meta_dir.glob("*.json"))
        total = len(meta_files)
        if total == 0:
            print(f"{split_name}: no meta files found \u2014 run download first")
            continue

        saved = 0
        skipped = 0
        print(f"{split_name}: generating masks for {total:,} samples ({s.get('masks', 0):,} already exist)")

        def process_meta(mp, _dir=split_dir):
            return generate_mask(mp, _dir)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i, was_saved in enumerate(executor.map(process_meta, meta_files)):
                if was_saved:
                    saved += 1
                else:
                    skipped += 1
                if (i + 1) % 5_000 == 0:
                    pct = (i + 1) / total * 100
                    print(f"  {i+1:,}/{total:,} ({pct:.0f}%) \u2014 {saved:,} new, {skipped:,} skipped")

        print(f"  {split_name} complete: {saved:,} generated, {skipped:,} skipped")

    print(f"\nMask generation complete at {otr_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify OTR

# COMMAND ----------

from pathlib import Path

otr_root = Path(VOLUME_PATH) / OUTPUT_FOLDER

print(f"OTR: {otr_root}")
print(f"Structure: OTR/{{split}}/{{input,target,masks,meta}}/\n")

total_input = 0
for split_dir in sorted(otr_root.iterdir()):
    if not split_dir.is_dir():
        continue
    input_count = len(list((split_dir / "input").glob("*.png"))) if (split_dir / "input").exists() else 0
    target_count = len(list((split_dir / "target").glob("*.png"))) if (split_dir / "target").exists() else 0
    mask_count = len(list((split_dir / "masks").glob("*.png"))) if (split_dir / "masks").exists() else 0
    meta_count = len(list((split_dir / "meta").glob("*.json"))) if (split_dir / "meta").exists() else 0
    paired = input_count == target_count == mask_count
    total_input += input_count
    print(f"  {split_dir.name}:")
    print(f"    input:  {input_count:,}")
    print(f"    target: {target_count:,}")
    print(f"    masks:  {mask_count:,}")
    print(f"    meta:   {meta_count:,}")
    print(f"    paired: {'YES' if paired else 'MISMATCH'}")

print(f"\nTotal paired samples: {total_input:,}")
print(f"\nReady for inpainting training:")
print(f"  input/*.png  -> image with text (model input)")
print(f"  masks/*.png  -> binary mask of text regions (inpainting mask)")
print(f"  target/*.png -> clean image without text (training target)")
print(f"  meta/*.json  -> word annotations + bounding boxes")
