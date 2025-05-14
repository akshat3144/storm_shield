# StormShield: Predictive Power Outage Forecasting

## Overview

Storms account for 62% of all power outages in the United States, posing significant risks to vulnerable populations who rely on uninterrupted power for medical devices, while also causing major financial losses to businesses and impairing emergency response operations. Accurate prediction of storm-induced power outages is therefore essential. However, existing models often underperform when it comes to forecasting outages caused by moderate and low-severity storms—events that are frequently overlooked in unified modeling approaches that fail to capture their distinct characteristics.

To address this challenge, a **storm severity-aware prediction framework** was developed. This framework routes storm events to **dedicated machine learning models** based on their severity. It utilizes data from **EAGLE-I**, **NOAA Storm Events**, and **PRISM Climate records** spanning 2014 to 2023, aligning storm occurrences with corresponding power outages and environmental variables.

Preprocessing included:

* **Missing data imputation** using Random Forest regression
* **Time zone alignment and normalization**
* **Feature selection** to remove sparse or irrelevant data
* Integration of **NLP-based features** extracted from storm event narratives, combined with structured environmental features

The prediction pipeline consists of three stages:

1. **Storm occurrence prediction** one hour in advance using **XGBoost** (F1-score: **0.9308**)
2. **Storm severity classification** using **LightGBM** (F1-score: **0.9329**)
3. **Outage prediction** via severity-specific models:

   * **LightGBM** for low-severity storms (F1-score: **0.9878**)
   * **XGBoost** for medium-severity storms (F1-score: **0.9739**)
   * **Random Forest** for high-severity storms (F1-score: **0.9755**)

This framework demonstrates competitive performance compared to state-of-the-art models. Additionally, a **custom metric** was proposed to aggregate F1 scores across all pipeline stages, offering a more realistic evaluation of end-to-end performance. This metric achieved a combined score of **0.8503**.

![Pipeline Flowchart](pipeline_flowchart.png)

## Features

- **Data Preprocessing**: Cleans and merges raw data into structured formats.
- **Feature Engineering**: Creates lagged variables for improving temporal prediction.
- **Modeling**: Implements multi-model prediction tasks:

  - Storm occurrence
  - Severity classification
  - Outage forecasting

- **Hyperparameter Tuning**: Uses Optuna for automated tuning.
- **Automation**: Scripts and notebooks streamline processing and evaluation.
- **Modular Structure**: Organized directories for data, models, preprocessing, and scripts.

## File Structure

```
Dynamic_Rhythms/
├── data/
│   ├── 01_raw_data.txt
│   ├── 02_interim_processed_data.txt
│   ├── 03_processed_data_final.txt
│   └── 04_split_data.txt
├── models/
│   ├── 01_storm_xgb_model.ipynb
│   ├── 02_severity_lgm_model.ipynb
│   ├── 03_outage_lgm_xgb_rf_model.ipynb
│   └── 04_models.txt
├── notebooks/
│   └── storm_outage_inference_pipeline.ipynb
├── preprocessing/
│   ├── 01_load_clean_merge_data.ipynb
│   └── 02_lag_feature_engineering.ipynb
├── LICENSE
├── pipeline_flowchart.png
├── README.md
└── requirements.txt
```

## Dependencies

Listed in `requirements.txt`:

- `pandas>=2.0.0`
- `numpy>=1.23.0`
- `scikit-learn>=1.2.0`
- `xgboost>=1.7.0`
- `lightgbm>=3.3.0`
- `tensorflow>=2.15.0`
- `nltk>=3.8.0`
- `optuna>=3.3.0`
- `matplotlib>=3.7.0`y
