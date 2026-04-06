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
