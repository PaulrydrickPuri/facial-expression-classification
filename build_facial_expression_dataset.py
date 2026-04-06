"""
build_facial_expression_dataset.py

Builds a balanced COCO-classification dataset for facial expression recognition.

Classes : happy, angry, neutral, sad
Splits  : train (70%) | val (15%) | test (15%)

Source   : /Users/paulrydrickpuri/Downloads/processed_data/dataset/<class>/
Output   : /Users/paulrydrickpuri/Documents/code/script/Facial expression project/
           ├── train/
           │   ├── <image files>
           │   └── annotations.json
           ├── val/
           │   ├── <image files>
           │   └── annotations.json
           └── test/
               ├── <image files>
               └── annotations.json

COCO classification format mirrors the carplate reference:
  annotations[i] = {"id": i, "image_id": i, "category_id": <cat_id>}
"""

import os
import json
import shutil
import random
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
SOURCE_ROOT = Path("/Users/paulrydrickpuri/Downloads/processed_data/dataset")
OUTPUT_ROOT = Path("/Users/paulrydrickpuri/Documents/code/script/Facial expression project")

CLASSES = ["happy", "angry", "neutral", "sad"]   # order → category_id 0,1,2,3

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

RANDOM_SEED  = 42
NOW_STR      = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
DATE_TAG     = datetime.now().strftime("%d%b%Y-%H_%M_%S")   # for info block

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def gather_images(class_dir: Path) -> list[Path]:
    """Return sorted list of image paths inside a class directory."""
    files = [
        p for p in sorted(class_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def stratified_split(items: list, ratios: dict, seed: int) -> dict[str, list]:
    """Split a list into named splits with the given ratios (must sum to 1)."""
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    # test gets the remainder so totals always add up
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def get_image_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) – fall back to (0,0) if PIL cannot open."""
    try:
        with Image.open(path) as img:
            return img.size          # (width, height)
    except Exception:
        return 0, 0


def build_coco_annotation(
    split_items: list[tuple[Path, int]],   # [(src_path, category_id), ...]
    split_name: str,
    categories: list[dict],
    date_tag: str,
) -> dict:
    """
    Build the full COCO-classification annotation dict for one split.
    Images + annotations share the same running index (id == image_id == annotation_id).
    """
    info = {
        "year": str(datetime.utcnow().year),
        "version": "1",
        "description": (
            f"Facial Expression COCO classification dataset – {split_name} split "
            f"(generated {date_tag})"
        ),
        "contributor": "",
        "url": "",
        "date_created": NOW_STR,
    }
    licenses = [{"id": 1, "url": "", "name": "Unknown"}]

    images      = []
    annotations = []

    for idx, (src_path, cat_id) in enumerate(split_items):
        w, h = get_image_dims(src_path)
        images.append({
            "id":           idx,
            "license":      1,
            "file_name":    src_path.name,
            "height":       h,
            "width":        w,
            "date_captured": NOW_STR,
        })
        annotations.append({
            "id":          idx,
            "image_id":    idx,
            "category_id": cat_id,
        })

    return {
        "info":        info,
        "licenses":    licenses,
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    # 1. Gather images per class
    class_images: dict[str, list[Path]] = {}
    for cls in CLASSES:
        src_dir = SOURCE_ROOT / cls
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {src_dir}")
        imgs = gather_images(src_dir)
        class_images[cls] = imgs
        print(f"  [{cls:>8s}] found {len(imgs):>6,} images")

    # 2. Balance: cap each class at the minimum count
    min_count = min(len(v) for v in class_images.values())
    print(f"\nBalancing to {min_count:,} images per class (min class size).")
    rng = random.Random(RANDOM_SEED)
    for cls in CLASSES:
        imgs = class_images[cls][:]
        rng.shuffle(imgs)
        class_images[cls] = imgs[:min_count]

    # 3. Stratified split per class, then combine ──────────────────────────────
    #    category_id = index of class in CLASSES list
    split_buckets: dict[str, list[tuple[Path, int]]] = {s: [] for s in SPLIT_RATIOS}
    for cat_id, cls in enumerate(CLASSES):
        splits = stratified_split(class_images[cls], SPLIT_RATIOS, RANDOM_SEED + cat_id)
        for split_name, items in splits.items():
            for p in items:
                split_buckets[split_name].append((p, cat_id))

    # Shuffle each combined split so classes are interleaved
    rng2 = random.Random(RANDOM_SEED)
    for split_name in split_buckets:
        rng2.shuffle(split_buckets[split_name])

    # 4. Print split summary ───────────────────────────────────────────────────
    print("\nSplit summary:")
    categories = [
        {"id": i, "name": cls, "supercategory": cls}
        for i, cls in enumerate(CLASSES)
    ]
    for split_name, items in split_buckets.items():
        counts = {cls: 0 for cls in CLASSES}
        for _, cat_id in items:
            counts[CLASSES[cat_id]] += 1
        counts_str = "  ".join(f"{cls}={n}" for cls, n in counts.items())
        print(f"  {split_name:>5s}: {len(items):>5,} images  ({counts_str})")

    # 5. Copy images + write annotations.json ──────────────────────────────────
    for split_name, items in split_buckets.items():
        split_dir = OUTPUT_ROOT / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing '{split_name}' split …")
        for src_path, _ in items:
            dst = split_dir / src_path.name
            if not dst.exists():
                shutil.copy2(src_path, dst)

        # Build & write annotation
        coco = build_coco_annotation(items, split_name, categories, DATE_TAG)
        ann_path = split_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, separators=(",", ":"))   # compact, like the reference
        print(f"  → annotations.json written ({len(coco['images'])} images)")

    print("\n✅  Dataset built successfully!")
    print(f"   Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
