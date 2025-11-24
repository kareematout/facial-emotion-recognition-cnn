# RAF-DB Data Summary Report

**Author:** Kareem Atout  
**Dataset:** RAF-DB (Real-world Affective Faces Database – Basic 7-class emotion recognition)  
**Date:** November 2025  

---

## 1. Dataset Overview
- **Location:** `data/rafdb/`  
- **Emotions:** Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral  
- **Structure:** Train/Test folders with 7 label subdirectories (1–7) and label CSVs  
- **Images Loaded:** 12,271 (train) + 3,068 (test) = 15,339 total  
- **Integrity Check:** All sampled images opened successfully; no missing or unreadable files.

---

## 2. Class Distribution
| Emotion    | Train % | Test % |
|-------------|----------|--------|
| Surprise (1) | 10.5 | 10.7 |
| Fear (2)     | 2.3 | 2.4 |
| Disgust (3)  | 5.8 | 5.2 |
| Happiness (4)| 38.9 | 38.6 |
| Sadness (5)  | 16.1 | 15.6 |
| Anger (6)    | 5.8 | 5.3 |
| Neutral (7)  | 20.6 | 22.2 |

- **Observation:** Happiness dominates the dataset; Fear and Disgust are under-represented.  
- **Imbalance ratio (max/min):** ≈ 17 → significant class imbalance.

---

## 3. Validation Split
- **Strategy:** Stratified 80 / 20 split from training data to create validation set.  
- **Sizes:** 9,816 (train) / 2,455 (val) / 3,068 (test).  
- **Class distribution:** Preserved via stratification.

---

## 4. Image Characteristics
- **Dimensions:** All images 100 × 100 pixels (square).  
- **Color:** RGB.  
- **Preprocessing Target:** Resize to 64 × 64 for CNN input.  
- **Alignment:** Faces are pre-aligned and centered.

---

## 5. Metadata Files Produced
Saved under `data/metadata/`:
- `targets_map_rafdb.json` — mapping of RAF-DB labels (1–7) to standard 0–6 indices.  
- `split_counts_rafdb.csv` — per-class image counts for train/val/test.  
- `dataset_stats_rafdb.json` — dataset-wide statistics and preprocessing settings.  
- `label_distribution.png` — class distribution bar charts.  
- `sample_images.png` — example images by emotion.  
- `preprocessing_example.png` — resize preview (64 × 64).

---

## 6. Key Points
- Dataset structure and labeling verified.  
- Images uniform and valid; preprocessing straightforward.  
- Noticeable class imbalance (Happiness ≫ Fear/Disgust).  
- Validation split ready for model training.  
- All supporting metadata saved for reproducibility.

---

**Status:** Preprocessing complete - dataset ready for CNN model construction.