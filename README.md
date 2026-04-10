# Facial Expression Dataset & Processing

This repository contains the dataset documentation, augmentation pipelines, and processing scripts for building a balanced Facial Expression Classification Dataset.

## Overview
The goal of this project is to build and augment a robust dataset for classifying four target emotions: **Happy, Anger, Neutral, Sad.**

### 1. Augmentation & Balancing Pipeline (`build_new_emotion_dataset_augmented.py`)
This script extracts raw data and uniformly balances under-represented emotion classes via safe facial augmentations (such as mild blurring and gentle brightness/sharpness jitters) to prevent dataset imbalance and over-fitting over dominant classes. 
All final outputs are generated in the strict COCO-JSON annotation schema, ready to be ingested for training split sets (`train`, `val`, `test`).

### 2. Dataset Updates
View the [`DATASET_UPDATES.md`](DATASET_UPDATES.md) file to see the exact lineage, volumes, and logic regarding how the core 23k baseline and the augmented 4,652 add-on patches were constructed.

## Getting Started
Scripts in this utility expect inputs structured in either nested categorical folders or existing COCO-JSON. 

*Ensure your local image subdirectories are registered inside your local `.gitignore` to prevent pushing mass binary image blobs to GitHub.*

---

## Progress

| Date | Session | What Was Done |
|---|---|---|
| 2026-04-10 | [cctv-face-expression-classification](sessions/2026-04-10_cctv-face-expression-classification.md) | Classified 74 CCTV face crops → anger/happy/neutral/sad using ViT model; 0 failures; results in CSV |

---

## Dataset

| Item | Detail |
|---|---|
| Type | Image Classification — Facial Expression |
| Classes | 4: anger, happy, neutral, sad |
| Base dataset | ~23k images (see DATASET_UPDATES.md) |
| Augmented add-on | 4,652 images (balanced) |
| CCTV inference set | 74 face crops (10 camera sessions) |
| Format | COCO-JSON (train/val/test splits) |
| Split | train / val / test |

---

## Model

| Item | Detail |
|---|---|
| Architecture | ViT-base-patch16-224 (`trpakov/vit-face-expression`) |
| Task | Image Classification — Facial Expression |
| Status | Inference on CCTV crops complete; training pipeline in progress |
| Platform | HuggingFace (inference) |

---

## Class Map

| ID | Class | Notes |
|---|---|---|
| 0 | anger | Includes model labels: angry, disgust |
| 1 | happy | Direct match |
| 2 | neutral | Includes model labels: neutral, fear, surprise |
| 3 | sad | Direct match |

---

## Key Files

| File | Purpose |
|---|---|
| `classify_expressions.py` | CCTV face crop expression classifier (ViT pipeline) |
| `expression_results.csv` | Per-image results: expression, confidence, blur, status, reason |
| `build_new_emotion_dataset_augmented.py` | Augmentation + balancing pipeline → COCO-JSON |
| `build_facial_expression_dataset.py` | Core dataset builder |
| `DATASET_UPDATES.md` | Dataset lineage and volume log |
| `sessions/` | Per-session progress logs |
| `PROGRESS.md` | Session index |

---

## Tech Stack
- Python 3.12
- `transformers`, `torch`, `torchvision` — model inference
- `Pillow`, `opencv-python` — image loading, blur detection, upscaling
- `numpy` — array ops

---

## Next Steps
- [ ] Manually review 3 low-confidence classified images (indices 1, 29, 67)
- [ ] Spot-check `fear`→`neutral` remapped images (9 total)
- [ ] Run per-camera expression breakdown analysis
- [ ] Integrate CCTV results into main dataset pipeline
- [ ] Evaluate with second model (`dima806/facial_emotions_image_detection`) for cross-validation

---

*Session logs maintained by [session-github-journal](https://github.com/PaulrydrickPuri) — Claude Code skill*
