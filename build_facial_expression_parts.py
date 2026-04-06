"""
build_facial_expression_parts.py

Splits the balanced facial expression dataset into parts of ~5,000 images each.
Each part gets its own train / val / test sub-folders, each containing:
  - image files  (copied from the already-built flat split dirs)
  - annotations.json  (COCO classification format)

Output structure:
  Facial expression project/
  ├── (COCO)Facial Expression(03Apr2026-xx_xx_xx)(pt1)/
  │   ├── train/   <images + annotations.json>
  │   ├── val/     <images + annotations.json>
  │   └── test/    <images + annotations.json>
  ├── (COCO)Facial Expression(03Apr2026-xx_xx_xx)(pt2)/
  │   ...
  └── ...

Source: the already-built train / val / test flat directories produced by
        build_facial_expression_dataset.py

Part size  : PART_SIZE images total (balanced across 4 classes)
Per-class  : PART_SIZE // len(CLASSES)  images per part
Split ratio: 70 % train  |  15 % val  |  15 % test  (applied per class inside each part)
"""

import os
import json
import math
import shutil
import random
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    "/Users/paulrydrickpuri/Documents/code/script/Facial expression project"
)

# The flat train/val/test directories produced by the previous script
FLAT_TRAIN = PROJECT_ROOT / "train"
FLAT_VAL   = PROJECT_ROOT / "val"
FLAT_TEST  = PROJECT_ROOT / "test"

CLASSES   = ["happy", "angry", "neutral", "sad"]   # category_id 0,1,2,3
PART_SIZE = 5_000          # target images per part
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

NOW_STR  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
DATE_TAG = datetime.now().strftime("%d%b%Y-%H_%M_%S")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_image_dims(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size        # (width, height)
    except Exception:
        return 0, 0


def build_coco_annotation(
    items: list[tuple[Path, int]],
    split_name: str,
    part_num: int,
    categories: list[dict],
) -> dict:
    info = {
        "year": str(datetime.utcnow().year),
        "version": "1",
        "description": (
            f"Facial Expression COCO classification – pt{part_num} {split_name} split "
            f"(generated {DATE_TAG})"
        ),
        "contributor": "",
        "url": "",
        "date_created": NOW_STR,
    }
    licenses = [{"id": 1, "url": "", "name": "Unknown"}]

    images      = []
    annotations = []
    for idx, (src_path, cat_id) in enumerate(items):
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
    categories = [
        {"id": i, "name": cls, "supercategory": cls}
        for i, cls in enumerate(CLASSES)
    ]

    # 1. Collect all images from the existing flat dirs, keyed by class
    #    We look for filenames that start with the class name.
    print("Collecting images from flat splits …")
    class_images: dict[str, list[Path]] = {cls: [] for cls in CLASSES}

    for flat_dir in [FLAT_TRAIN, FLAT_VAL, FLAT_TEST]:
        for f in sorted(flat_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            for cls in CLASSES:
                if f.name.startswith(cls):
                    class_images[cls].append(f)
                    break

    min_count = min(len(v) for v in class_images.values())
    print(f"  Images per class (before trim): "
          + "  ".join(f"{c}={len(class_images[c])}" for c in CLASSES))

    # Shuffle each class list reproducibly
    for cat_id, cls in enumerate(CLASSES):
        rng = random.Random(RANDOM_SEED + cat_id)
        rng.shuffle(class_images[cls])
        class_images[cls] = class_images[cls][:min_count]

    print(f"  Balanced to {min_count:,} images per class")

    # 2. Determine part layout
    per_class_per_part = PART_SIZE // len(CLASSES)   # images per class per part
    num_parts = math.ceil(min_count / per_class_per_part)
    print(f"\nPart size : {PART_SIZE:,} images  "
          f"({per_class_per_part} per class)\n"
          f"Num parts : {num_parts}")

    # 3. Build each part
    for part_idx in range(num_parts):
        part_num  = part_idx + 1
        lo        = part_idx * per_class_per_part
        hi        = min(lo + per_class_per_part, min_count)
        part_name = f"(COCO)Facial Expression({DATE_TAG})(pt{part_num})"
        part_dir  = PROJECT_ROOT / part_name

        print(f"\n── Part {part_num} / {num_parts} : {part_name} ──")

        # Slice images for this part
        part_class_images: dict[str, list[Path]] = {
            cls: class_images[cls][lo:hi] for cls in CLASSES
        }
        actual_per_class = hi - lo
        print(f"  {actual_per_class} images × {len(CLASSES)} classes "
              f"= {actual_per_class * len(CLASSES):,} total")

        # Stratified split within the part
        split_buckets: dict[str, list[tuple[Path, int]]] = {
            "train": [], "val": [], "test": []
        }
        for cat_id, cls in enumerate(CLASSES):
            imgs = part_class_images[cls]
            n    = len(imgs)
            n_tr = int(n * SPLIT_RATIOS["train"])
            n_va = int(n * SPLIT_RATIOS["val"])
            split_buckets["train"] += [(p, cat_id) for p in imgs[:n_tr]]
            split_buckets["val"]   += [(p, cat_id) for p in imgs[n_tr:n_tr + n_va]]
            split_buckets["test"]  += [(p, cat_id) for p in imgs[n_tr + n_va:]]

        # Shuffle each combined split
        for split_name in split_buckets:
            random.Random(RANDOM_SEED + part_idx).shuffle(split_buckets[split_name])

        # Print per-split summary
        for split_name, items in split_buckets.items():
            counts = {cls: 0 for cls in CLASSES}
            for _, cid in items:
                counts[CLASSES[cid]] += 1
            counts_str = "  ".join(f"{cls}={n}" for cls, n in counts.items())
            print(f"  {split_name:>5s}: {len(items):>5,}  ({counts_str})")

        # Copy images + write annotations.json
        for split_name, items in split_buckets.items():
            split_dir = part_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for src_path, _ in items:
                dst = split_dir / src_path.name
                if not dst.exists():
                    shutil.copy2(src_path, dst)

            coco = build_coco_annotation(items, split_name, part_num, categories)
            ann_path = split_dir / "annotations.json"
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(coco, f, separators=(",", ":"))

            print(f"    → {split_dir.name}/annotations.json  ({len(coco['images'])} images)")

    print("\n✅  All parts built successfully!")
    print(f"   Output: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
