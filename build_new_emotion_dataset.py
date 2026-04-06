"""
build_new_emotion_dataset.py

Extracts 4 target classes (happy, anger, neutral, sad) from:
  /Users/paulrydrickpuri/Downloads/emotion classification.v2i.folder

Preserves the source's existing train / valid / test splits
(renames 'valid' → 'val') and generates COCO classification
annotations.json for each split.

Output:
  Facial expression project/
  └── (COCO)New Emotion Dataset(<date>)/
      ├── train/  ← images + annotations.json
      ├── val/    ← images + annotations.json
      └── test/   ← images + annotations.json

Category mapping (matches new source naming):
  0 = happy
  1 = anger
  2 = neutral
  3 = sad
"""

import json
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

# Classes to extract (must match folder names in source)
CLASSES = ["happy", "anger", "neutral", "sad"]   # → category_id 0,1,2,3

# source split name → output split name
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

NOW_STR  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
DATE_TAG = datetime.now().strftime("%d%b%Y-%H_%M_%S")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_DIR = PROJECT_ROOT / f"(COCO)New Emotion Dataset({DATE_TAG})"

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
            f"{split_name} split (generated {DATE_TAG})"
        ),
        "contributor": "",
        "url": "",
        "date_created": NOW_STR,
    }
    licenses = [{"id": 1, "url": "", "name": "Unknown"}]
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

    print(f"Output → {OUTPUT_DIR.name}\n")
    grand_total = 0

    for src_split, out_split in SPLIT_MAP.items():
        items: list[tuple[Path, int]] = []

        for cat_id, cls in enumerate(CLASSES):
            src_dir = SOURCE_ROOT / src_split / cls
            if not src_dir.exists():
                print(f"  [WARN] Missing: {src_dir} — skipping")
                continue

            imgs = [
                f for f in sorted(src_dir.iterdir())
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            for p in imgs:
                items.append((p, cat_id))

        if not items:
            print(f"  [{src_split}] No images found — skipping split.")
            continue

        # Per-class counts
        counts = {cls: 0 for cls in CLASSES}
        for _, cid in items:
            counts[CLASSES[cid]] += 1
        counts_str = "  ".join(f"{cls}={n}" for cls, n in counts.items())
        print(f"  {out_split:>5s}: {len(items):>5,} images  ({counts_str})")

        # Copy images
        split_dir = OUTPUT_DIR / out_split
        split_dir.mkdir(parents=True, exist_ok=True)
        for src_path, _ in items:
            dst = split_dir / src_path.name
            if not dst.exists():
                shutil.copy2(src_path, dst)

        # Write annotations.json
        coco = build_coco(items, out_split, categories)
        ann_path = split_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, separators=(",", ":"))
        print(f"         → annotations.json written ({len(coco['images'])} entries)")
        grand_total += len(items)

    print(f"\nTotal images extracted : {grand_total:,}")
    print(f"Categories (id: name)  : " +
          ", ".join(f"{i}: {cls}" for i, cls in enumerate(CLASSES)))
    print(f"\n✅  Done!  →  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
