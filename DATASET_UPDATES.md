# Facial Expression Dataset Updates

## Update 1: The Core 23,680 Image Dataset
- **Status:** Already uploaded to the training platform.
- **Composition:** Consists of 23,680 perfectly balanced images across the 4 target classes (Happy, Anger, Neutral, Sad), yielding 5,920 images per class.
- **Deduplication:** These images have gone through a comprehensive deduplication process utilizing FAISS and CLIP embeddings across previous split parts (pt1 to pt5) to ensure diversity and quality.

## Update 2: The New 4,192 Image Expansion
- **Source:** Extracted exactly 4 classes (Happy, Anger, Neutral, Sad) from a newly acquired raw dataset (`emotion classification.v2i.folder`).
- **Initial Count:** Yielded exactly 4,192 images (with class imbalances, where "Sad" was the maximum at 1,163 images, and "Happy" at 889 images).
- **Augmentation & Balancing Strategy:** 
  - To match the final aggregated volume balance, we augmented all smaller classes up to the highest target count within the batch: **1,163 images** per class.
  - Safe augmentations applied included minor brightness, contrast, sharpness, and simple blurs. No flip/rotation operations were used in order to preserve the original face orientation.
  - Doing this brought the local slice total to exactly **4,652 images** (4 classes × 1,163).
- **Split & COCO Format:** This specific subset was kept physically separated from the main 23,680 dataset so it can be uploaded separately, split strictly into train (70%), val (15%), and test (15%), and saved down exactly in COCO JSON format. 

### Final Merged Result (When Uploaded)
When uploaded onto the platform alongside dataset 1 (23,680), the new final balanced counts will be exactly **7,083 images** per class (5,920 + 1,163 = 7,083), leading to the grand total of **28,332 images** inside your master dataset!
