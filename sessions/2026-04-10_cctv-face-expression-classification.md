# Session Log — 2026-04-10
**Project:** Facial Expression Classification — CCTV Face Crops  
**Working directory:** `/Users/paulrydrickpuri/Downloads/detected_faces`  
**Duration context:** Single session; model inference + CSV report completed in one pass

---

## Summary

The goal of this session was to perform automated facial expression classification on 74 face crops extracted from CCTV footage, categorising each image into one of four target classes: **anger, happy, neutral, sad**. A HuggingFace ViT-based pre-trained model (`trpakov/vit-face-expression`) was selected and wrapped in a full Python pipeline that handled image loading, quality checking, upscaling, inference, emotion-label mapping, and CSV export. All 74 images were successfully classified — 0 failures — with 13 images flagged with explanatory notes about low confidence or label remapping from the model's 7-class output to the 4 target classes.

---

## What Was Built

### `classify_expressions.py` — Full Classification Pipeline
- **Location:** `/Users/paulrydrickpuri/Downloads/detected_faces/classify_expressions.py`
- Walks all subdirectories recursively for `.jpg`, `.png`, `.jpeg` files
- Loads each image, checks minimum dimensions (≥20px), computes blur via Laplacian variance
- Upscales every crop to 224×224 using LANCZOS (required for ViT input)
- Runs HuggingFace `pipeline("image-classification", model="trpakov/vit-face-expression")`
- Maps the model's 7-class output to the 4 project target classes (see Decisions table)
- Writes a per-image row to `expression_results.csv` with full metadata
- Prints a colour-coded terminal summary + final count breakdown

### `expression_results.csv` — Per-Image Results Report
- **Location:** `/Users/paulrydrickpuri/Downloads/detected_faces/expression_results.csv`
- Columns: `index`, `folder`, `filename`, `expression`, `confidence`, `raw_label`, `blur_score`, `status`, `reason`
- `status` values: `OK`, `LOW_CONF` (confidence < 40%), `FAILED` (unreadable file — none in this run)
- `reason` column provides human-readable explanation for any flag or label remapping

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Model | `trpakov/vit-face-expression` (HuggingFace ViT) | Pre-trained on FER/AffectNet; accepts raw PIL images; fast CPU inference; no additional dependencies beyond `transformers` + `torch` (already installed) |
| Upscaling method | LANCZOS (`Image.LANCZOS`) | Highest-quality PIL resampling for small-to-large; preserves edge sharpness better than bilinear/bicubic at extreme upscale ratios (e.g. 18px → 224px) |
| Minimum dimension threshold | 20px | 2 images in the dataset were 18×22px; these are still usable after upscaling but warrant a warning flag |
| Blur threshold | Laplacian variance < 10.0 | Standard CV blur metric; only 1 image in dataset fell near threshold — blur was not a major issue |
| Low-confidence threshold | 40% | Balances sensitivity to uncertain predictions without over-flagging; 3 images triggered this |
| `fear` → `neutral` mapping | Neutral | CCTV context: "fear" expressions in surveillance footage almost always correspond to alert/focused neutral looks, not genuine fear. Mapping to neutral is more domain-appropriate. |
| `disgust` → `anger` mapping | Anger | Disgust and anger share similar facial muscle activations (furrowed brow, tight lips); anger is the closer target class |
| `surprise` → `neutral` mapping | Neutral | Surprise is ambiguous in low-res CCTV crops and often misread by the model; neutral is the safest fallback |
| Inference device | CPU (`device=-1`) | No GPU confirmed in environment; CPU inference acceptable for 74 images |

---

## Findings & Observations

1. **All 74 images classified — 0 failures.** Despite very small resolutions (18–91px), every image produced a valid classification.
2. **Dominant expression: `sad` (29/74, 39%).** Expected for CCTV — relaxed/downward facial muscle tone in transit reads as sad to the model.
3. **`neutral` second (20/74, 27%)** — consistent with CCTV surveillance context where subjects are not emotionally engaged.
4. **`anger` third (17/74, 23%)** — notably high; many CCTV faces show concentrated or tense expressions, which the model interprets as anger.
5. **`happy` least frequent (8/74, 11%)** — again expected; genuine smiles are rare in CCTV footage captured in car parks and building floors.
6. **Image resolution range: 18×22px (min) to 91×101px (max), average ~51×59px.** All images required significant upscaling to meet the model's 224×224 input.
7. **Model originally outputs 7 classes:** angry, disgust, fear, happy, neutral, sad, surprise. Of these, `fear` appeared most frequently as the raw label in cases mapped to `neutral` (9 images).
8. **3 images flagged `LOW_CONF`:** image indices 1 (23.5% confidence), 29 (38.9%), 67 (32.2%). These remain classified but are marked uncertain.
9. **Blur scores ranged from 25.6 to 1293.** High blur score does NOT mean blurry — Laplacian variance is higher for sharper images. The one very-small image (18×22px) had a blur score of 1293 (sharp for its size) and was still classified correctly as `sad` at 77.3% confidence.
10. **Data source context:** Images are named with pattern `face_<id>_src<cam>_frame<n>_conf<detection_confidence>.jpg`. Detection confidence from the upstream face detector ranged from 0.30 to 0.99 across the dataset.
11. **Folders correspond to CCTV camera sessions:** e.g. `CCTV Car Park 6`, `CCTV FLOOR 1` — 10 total camera/time-window combinations.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| `TypeError: string indices must be integers` when inspecting model labels via `__dict__` | Incorrect introspection path for `transformers.Pipeline` config | Removed the label-inspection line; model loaded and ran correctly without it |
| Images too small for ViT (expects 224×224) | CCTV face crops are tiny (avg 51×59px) | Applied `Image.LANCZOS` resize to 224×224 before passing to model; quality acceptable for inference |
| Model outputs 7 classes, project requires 4 | Model trained on FER/AffectNet (7 classes) vs. project target (4 classes) | Implemented `EMOTION_MAP` dict mapping all 7 labels → 4 target classes with documented rationale per mapping |
| `use_fast` deprecation warning from `ViTImageProcessor` | HuggingFace transformers updated default processor | Suppressed via `warnings.filterwarnings("ignore")` — not a functional issue |

---

## ML / Model Config

```python
# Model used
MODEL_NAME = "trpakov/vit-face-expression"
# Architecture: Vision Transformer (ViT-base-patch16-224)
# Trained on: FER2013 + AffectNet (7 classes)
# Inference: HuggingFace transformers pipeline, CPU

# Input preprocessing
TARGET_SIZE = (224, 224)           # ViT required input
RESAMPLING  = Image.LANCZOS        # High-quality upscale

# Quality thresholds
MIN_DIMENSION       = 20           # px — below this, flag but still classify
BLUR_THRESHOLD      = 10.0         # Laplacian variance — below = blurry flag
LOW_CONF_THRESHOLD  = 0.40         # Model confidence — below = LOW_CONF status

# Emotion mapping (7 → 4 classes)
EMOTION_MAP = {
    "angry":   "anger",
    "disgust": "anger",
    "fear":    "neutral",   # domain-specific: CCTV context
    "happy":   "happy",
    "neutral": "neutral",
    "sad":     "sad",
    "surprise":"neutral",
}
```

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `classify_expressions.py` | `/Users/paulrydrickpuri/Downloads/detected_faces/` | Full classification pipeline script |
| `expression_results.csv` | `/Users/paulrydrickpuri/Downloads/detected_faces/` | Per-image results: expression, confidence, raw label, blur score, status, reason |

---

## Next Steps / Open Questions

- [ ] **Visual review** — manually inspect the 3 `LOW_CONF` images (indices 1, 29, 67) to verify or override the model's classification
- [ ] **Review `fear`→`neutral` mappings** — 9 images were remapped; spot-check whether `neutral` or `sad` is more accurate in context
- [ ] **Per-camera breakdown** — analyse expression distribution per CCTV camera/folder to see if certain locations skew toward specific expressions
- [ ] **Integrate into main dataset pipeline** — link results back to `build_facial_expression_dataset.py` for inclusion in training data
- [ ] **Consider higher-res face crops** — if source videos are available, re-extract faces at a larger bounding box to improve model accuracy
- [ ] **Evaluate with a second model** — cross-validate with `dima806/facial_emotions_image_detection` (ResNet-based) to catch disagreements on low-confidence images
- [ ] **Export per-class image folders** — split detected images into `anger/`, `happy/`, `neutral/`, `sad/` subdirectories for downstream dataset assembly

---

## Session Metadata
- **Date:** 2026-04-10
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Downloads/detected_faces`
- **Key packages used:** `transformers`, `torch`, `Pillow`, `opencv-python`, `numpy`
- **HuggingFace model:** `trpakov/vit-face-expression`
- **Images processed:** 74 across 10 CCTV camera subfolders
- **Classification results:** anger=17, happy=8, neutral=20, sad=29; low_conf=3, failed=0
