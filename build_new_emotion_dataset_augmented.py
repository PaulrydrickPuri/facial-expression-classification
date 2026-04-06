"""
build_new_emotion_dataset_augmented.py

Extracts 4 target classes (happy, anger, neutral, sad) from:
  /Users/paulrydrickpuri/Downloads/emotion classification.v2i.folder

Balances the new dataset by AUGMENTING the smaller classes until they
reach the size of the largest class (sad = 1163).
This ensures the new dataset is locally balanced at 1163 per class,
so when combined with pt1–pt5 (5920 per class), the whole dataset
will be perfectly balanced (7083 per class).

Augmentations used (safe for faces, no geometric/orientation changes):
  - Brightness jitter
  - Contrast jitter
  - Sharpness jitter
  - Mild Gaussian blur

Output (4,652 total = 1163 × 4 classes):
  Facial expression project/
  └── (COCO)New Emotion Dataset Augmented(<date>)/
      ├── train/  70%
      ├── val/    15%
      └── test/   15%
"""

import json
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# ── Config ─────────────────────────────────────────────────────────────────────
SOURCE_ROOT = Path(
    "/Users/paulrydrickpuri/Downloads/emotion classification.v2i.folder"
)
PROJECT_ROOT = Path(
    "/Users/paulrydrickpuri/Documents/code/script/Facial expression project"
)

CLASSES = ["happy", "anger", "neutral", "sad"]   # → category_id 0,1,2,3
SOURCE_SPLITS = ["train", "valid", "test"]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

NOW_STR  = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
DATE_TAG = datetime.now().strftime("%d%b%Y-%H_%M_%S")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_DIR = PROJECT_ROOT / f"(COCO)New Emotion Dataset Augmented({DATE_TAG})"
TEMP_AUG_DIR = PROJECT_ROOT / f"temp_aug_{DATE_TAG}"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_image_dims(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0

def safe_augment(img: Image.Image) -> Image.Image:
    """Apply a random, safe augmentation that doesn't change orientation."""
    aug_type = random.choice(['brightness', 'contrast', 'sharpness', 'blur'])
    
    if aug_type == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(0.7, 1.3))
    elif aug_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(0.7, 1.3))
    elif aug_type == 'sharpness':
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(random.uniform(0.5, 2.0))
    elif aug_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    return img

def build_coco(items: list[tuple[Path, int]], split_name: str,
               categories: list[dict]) -> dict:
    """Build a COCO-classification dict for one split."""
    info = {
        "year": str(datetime.utcnow().year),
        "version": "1",
        "description": (
            f"Facial Expression COCO classification – New Emotion Dataset "
            f"(augmented) {split_name} split (generated {DATE_TAG})"
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
    random.seed(RANDOM_SEED)
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
    max_in_new = max(len(v) for v in class_images.values())
    target_per_class = max_in_new   # = 1163 (sad)

    print(f"\nBalance Target via Augmentation:")
    print(f"  Largest class in new dataset : {max_in_new} (sad)")
    print(f"  → Augment all classes to       : {target_per_class}")

    # 3. Create augmented temp directory
    TEMP_AUG_DIR.mkdir(parents=True, exist_ok=True)
    augmented_class_images: dict[str, list[Path]] = {cls: [] for cls in CLASSES}

    print("\nGenerating augmentations ...")
    for cls in CLASSES:
        cls_temp_dir = TEMP_AUG_DIR / cls
        cls_temp_dir.mkdir(exist_ok=True)
        
        src_imgs = class_images[cls]
        needed = target_per_class - len(src_imgs)
        
        # Copy originals to temp
        for img_path in src_imgs:
            dst = cls_temp_dir / f"{cls}_{img_path.name}"
            # To avoid clashes if same name exists in different splits
            if dst.exists():
                dst = cls_temp_dir / f"{cls}_{uuid.uuid4().hex[:8]}_{img_path.name}"
            shutil.copy2(img_path, dst)
            augmented_class_images[cls].append(dst)
        
        # Generate augmentations
        if needed > 0:
            print(f"  [{cls:>8s}]: Adding {needed:>} augmented images")
            for i in range(needed):
                src_img = random.choice(src_imgs)
                dst = cls_temp_dir / f"{cls}_aug{i:05d}_{src_img.name}"
                try:
                    with Image.open(src_img).convert("RGB") as img:
                        aug_img = safe_augment(img)
                        aug_img.save(dst)
                    augmented_class_images[cls].append(dst)
                except Exception as e:
                    print(f"  [WARN] Failed to augment {src_img}: {e}")
                    # If it fails, just copy the original as a fallback
                    shutil.copy2(src_img, dst)
                    augmented_class_images[cls].append(dst)
        else:
            print(f"  [{cls:>8s}]: No augmentation needed")

    # 4. 70/15/15 split per class, then combine
    split_buckets: dict[str, list[tuple[Path, int]]] = {
        "train": [], "val": [], "test": []
    }
    for cat_id, cls in enumerate(CLASSES):
        splits = stratified_split(augmented_class_images[cls], RANDOM_SEED + cat_id)
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

    # Cleanup temp
    print("\nCleaning up temporary augmentation directory...")
    shutil.rmtree(TEMP_AUG_DIR)

    print(f"\n✅  Done!  →  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
