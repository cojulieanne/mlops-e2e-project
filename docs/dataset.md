# Dataset Documentation

## Overview
This dataset contains survey-style records of student demographics, habits, and academic outcomes.  

- **Target variable:** `Pass/Fail` (binary: `1=Pass`, `0=Fail`)  

- **Size:** 1,001 rows total (including header; 1,000 student records).  

- **Granularity:** Each row represents one student record.  


---

## Source

- **Bronze layer:** Initial raw CSV data pulled from Google Drive, used in our ML2 subject (`bronze/ml2_student_performance.csv`).  

- **Silver layer:** Preprocessed dataset where pass/fail labels were flipped to align with monitoring needs.  

- **Gold layer:** Train/validation/test splits created during feature engineering, stored in `gold/`.  


---

## Preprocessing Workflow

1. **Ingestion:** Raw CSV placed in the `bronze/` folder.  

2. **Transformation:** Pass/fail labels flipped (`0 → 1`, `1 → 0`) and stored in `silver/`.  

3. **Feature Engineering:** Train/validation/test split performed; resulting datasets saved in `gold/`.  


---

## Features & Target

- **Features:**  

  - `Age` — student age in years  
  
  - `Course (STEM=1, Non-STEM=0)` — program type indicator  
  
  - `Hours of Sleep` — average nightly sleep hours  
  
  - `In a Relationship (1=Yes, 0=No, 0.5=It's complicated)` — relationship status (fractional encoding allowed)  
  
  - `Hours Reviewing` — average daily study/review hours  


- **Target:**  
  
  - `Pass/Fail (1=Pass, 0=Fail)` — binary classification outcome (flipped during preprocessing for monitoring emphasis).  


---

## Context
The purpose of the dataset in its original use (ML2 subject) was to demonstrate the effect of imbalanced classification techniques, focusing on how different resampling strategies and models perform when predicting minority outcomes (failed students).

For this project, the same dataset is repurposed to train baseline ML models, demonstrate MLOps practices with MLflow for experiment tracking, Airflow for orchestration, and Evidently for drift detection. It is also used here to provide a realistic but controlled case study for simulating and monitoring data drift (see [`docs/drift_plan.md`](./drift_plan.md)).

The dataset has 6 columns and 1,000 rows.

For detailed schema and feature-level notes, see [`docs/data_dictionary.md`](./data_dictionary.md).