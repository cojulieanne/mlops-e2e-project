# Data Dictionary

This file describes the schema of the **Student Performance dataset**, including feature types, ranges/values, and notes on preprocessing.

| Column Name | Type | Dataset Values | Description | Notes |
|-------------|------|--------------------------|-------------|-------|
| **Age** | Integer | 24 - 54 | Student’s age in years | Used as a numeric feature. Outliers should be checked. |
| **Course (STEM=1, Non-STEM=0)** | Integer (binary) | 1 = STEM, 0 = Non-STEM | Program type indicator | Binary categorical feature. |
| **Hours of Sleep** | Float | 3.0 – 8.0 | Average nightly sleep hours | Continuous feature; skew possible. |
| **In a Relationship (1=Yes, 0=No, 0.5=It's complicated)** | Float (categorical encoding) | 1.0, 0.0, 0.5 | Relationship status | Encoded numerically, but represents categories. |
| **Hours Reviewing** | Float | 1.0 – 6.0 | Average daily study/review hours | Continuous feature. Potential driver of performance. |
| **Pass/Fail (1=Pass, 0=Fail)** | Integer (binary, **Target**) | 1 = Pass, 0 = Fail | Academic outcome | **Target variable.** Labels flipped during preprocessing (silver layer). |

---

## Notes on Schema

- **Target:** `Pass/Fail` is the prediction target. For MLOps monitoring, the project emphasizes recall for the `Fail` class.  

- **Feature types:** Although all features are numeric, some (like `Course` and `Relationship`) are encoded categories and may need different preprocessing in future models.  

- **Preprocessing:** In the pipeline:

  - Labels are flipped in the **silver** layer.
  
  - Train/validation/test splits are created in the **gold** layer.  


---

## Usage

This schema supports:

- **Feature engineering**, such as encoding categorical features, scaling continuous features.

- **Model training**, thru predicting binary `Pass/Fail`.

- **Monitoring**, by detecting drift in both numerical and categorical distributions.

