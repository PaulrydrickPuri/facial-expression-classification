"""
build_new_emotion_dataset_balanced.py

Extracts 4 target classes (happy, anger, neutral, sad) from:
  /Users/paulrydrickpuri/Downloads/emotion classification.v2i.folder

Balances the new dataset so that when combined with the existing
pt1–pt5 (5,920 per class), both datasets together are perfectly balanced.

Balance logic:
  - Existing: 5,920 per class × 4 classes = 23,680
  - Min class in new dataset: happy = 889
  - → cap all new classes at 889
  - → combined per class: 5,920 + 889 = 6,809

Output (3,556 total = 889 × 4 classes):
  Facial expression project/
  └── (COCO)New Emotion Dataset Balanced(<date>)/
      ├── train/  70% → 622 per class = 2,488 total
      ├── val/    15% → 133 per class =   532 total
      └── test/   15% → 134 per class =   536 total

Category IDs:
  0 = happy  |  1 = anger  |  2 = neutral  |  3 = sad
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
SOURCE_ROOT = Path(
    "/Users/paulrydrickpuri/Downloads/emotion classification.v2i.folder"
)
PROJECT_ROOT = Path(
    "/Users/paulrydrickpuri/Documents/code/script/Facial expression project"
)

CLASSES      = ["happy", "anger", "neutral", "sad"]   # → category_id 0,1,2,3
SOURCE_SPLITS = ["train", "valid", "test"]             # source folder names

EXISTING_PER_CLASS = 5_920   # already uploaded pt1–pt5
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

NOW_STR  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
DATE_TAG = datetime.now().strftime("%d%b%Y-%H_%M_%S")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_DIR = PROJECT_ROOT / f"(COCO)New Emotion Dataset Balanced({DATE_TAG})"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_image_dims(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size   # (width, height)
    except Exception:
        return 0, 0


def build_coco(items: list[tuple[Path, int]], split_name: str,
               categories: list[dict]) -> dict:
    """Build a COCO-classification dict for one split."""
    info = {
        "year": str(datetime.utcnow().year),
        "version": "1",
        "description": (
            f"Facial Expression COCO classification – New Emotion Dataset "
            f"(balanced) {split_name} split (generated {DATE_TAG})"
        ),
        "contributor": "",
        "url": "",
        "date_created": NOW_STR,
    }
    images, annotations = [], []
    for idx, (src_path, cat_id) in enumerate(items):
        w, h = get_image_dims(src_path)
        images.append({
            "id":            idx,
            "license":       1,
            "file_name":     src_path.name,
            "height":        h,
            "width":         w,
            "date_captured": NOW_STR,
        })
        annotations.append({
            "id":          idx,
            "image_id":    idx,
            "category_id": cat_id,
        })
    return {
        "info":        info,
        "licenses":    [{"id": 1, "url": "", "name": "Unknown"}],
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }


def stratified_split(items: list, seed: int) -> dict[str, list]:
    """70 / 15 / 15 split."""
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n      = len(shuffled)
    n_tr   = int(n * SPLIT_RATIOS["train"])
    n_va   = int(n * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:n_tr],
        "val":   shuffled[n_tr : n_tr + n_va],
        "test":  shuffled[n_tr + n_va:],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    categories = [
        {"id": i, "name": cls, "supercategory": cls}
        for i, cls in enumerate(CLASSES)
    ]

    # 1. Pool ALL images per class from ALL source splits
    print("Collecting images from source …")
    class_images: dict[str, list[Path]] = {cls: [] for cls in CLASSES}
    for src_split in SOURCE_SPLITS:
        for cls in CLASSES:
            cls_dir = SOURCE_ROOT / src_split / cls
            if not cls_dir.exists():
                continue
            imgs = [
                f for f in sorted(cls_dir.iterdir())
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            class_images[cls].extend(imgs)

    print("\nSource image counts:")
    for cls in CLASSES:
        print(f"  {cls:>8s}: {len(class_images[cls]):,}")

    # 2. Determine balance target
    min_in_new = min(len(v) for v in class_images.values())
    target_per_class = min_in_new   # = 889 (happy)
    combined_per_class = EXISTING_PER_CLASS + target_per_class

    print(f"\nBalance target:")
    print(f"  Smallest class in new dataset   : {min_in_new} (happy)")
    print(f"  → Cap all classes at             : {target_per_class}")
    print(f"  Existing pt1–pt5 per class       : {EXISTING_PER_CLASS:,}")
    print(f"  Combined per class after upload  : {combined_per_class:,}")
    print(f"  Total combined                   : {combined_per_class * len(CLASSES):,}")

    # 3. Trim each class to target, shuffle reproducibly
    rng = random.Random(RANDOM_SEED)
    for cls in CLASSES:
        imgs = class_images[cls][:]
        rng.shuffle(imgs)
        class_images[cls] = imgs[:target_per_class]

    # 4. 70/15/15 split per class, then combine
    split_buckets: dict[str, list[tuple[Path, int]]] = {
        "train": [], "val": [], "test": []
    }
    for cat_id, cls in enumerate(CLASSES):
        splits = stratified_split(class_images[cls], RANDOM_SEED + cat_id)
        for split_name, items in splits.items():
            for p in items:
                split_buckets[split_name].append((p, cat_id))

    # Shuffle each combined split
    rng2 = random.Random(RANDOM_SEED)
    for split_name in split_buckets:
        rng2.shuffle(split_buckets[split_name])

    # 5. Print summary
    print(f"\nNew dataset split summary (balanced at {target_per_class}/class):")
    total_new = 0
    for split_name, items in split_buckets.items():
        counts = {cls: 0 for cls in CLASSES}
        for _, cid in items:
            counts[CLASSES[cid]] += 1
        counts_str = "  ".join(f"{cls}={n}" for cls, n in counts.items())
        print(f"  {split_name:>5s}: {len(items):>5,}  ({counts_str})")
        total_new += len(items)
    print(f"  TOTAL: {total_new:,}")

    # 6. Copy images + write annotations.json
    print(f"\nOutput → {OUTPUT_DIR.name}")
    for split_name, items in split_buckets.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for src_path, _ in items:
            dst = split_dir / src_path.name
            if not dst.exists():
                shutil.copy2(src_path, dst)

        coco = build_coco(items, split_name, categories)
        with open(split_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(coco, f, separators=(",", ":"))
        print(f"  → {split_name}/annotations.json  ({len(coco['images'])} images)")

    print(f"\n✅  Done!  →  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
