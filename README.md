# Dynamic Rhythms

## Overview

**Dynamic Rhythms** is a machine learning-based project designed to predict storm severity and related power outages using meteorological and infrastructure data. It incorporates a complete ML pipeline from data preprocessing to model deployment using Random Forest, LightGBM, and Feedforward Neural Networks. The objective is to enhance disaster readiness and outage management.

![Pipeline Flowchart](pipeline_flowchart.png)

## Features

* **Data Preprocessing**: Cleans and merges raw data into structured formats.
* **Feature Engineering**: Creates lagged variables for improving temporal prediction.
* **Modeling**: Implements multi-model prediction tasks:

  * Storm occurrence
  * Severity classification
  * Outage forecasting
* **Hyperparameter Tuning**: Uses Optuna for automated tuning.
* **Automation**: Scripts and notebooks streamline processing and evaluation.
* **Modular Structure**: Organized directories for data, models, preprocessing, and scripts.

## File Structure

```
Dynamic_Rhythms/
├── data/
│   ├── 01_raw_data.txt
│   ├── 02_interim_processed_data.txt
│   ├── 03_processed_data_final.txt
│   └── 04_split_data.txt
├── models/
│   ├── 01_storm_prediction_model.ipynb
│   ├── 02_severity_prediction_model.ipynb
│   └── 03_outage_prediction_model.ipynb
├── notebooks/
│   ├── 01_initial_storm_severity_outage_prediction.ipynb
│   └── 02_prediction_pipeline_main.ipynb
├── preprocessing/
│   ├── 01_load_clean_merge_data.ipynb
│   └── 02_lag_feature_engineering.ipynb
├── scripts/
│   ├── 01_data_split.py
│   └── 02_tuning_optuna.py
├── LICENSE
├── pipeline_flowchart.png
├── README.md
└── requirements.txt
```

## Dependencies

Listed in `requirements.txt`:

* `pandas>=2.0.0`
* `numpy>=1.23.0`
* `scikit-learn>=1.2.0`
* `xgboost>=1.7.0`
* `lightgbm>=3.3.0`
* `tensorflow>=2.15.0`
* `nltk>=3.8.0`
* `optuna>=3.3.0`
* `matplotlib>=3.7.0`