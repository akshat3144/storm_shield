# StormShield: Predictive Power Outage Forecasting

## Overview

Storms are responsible for 62% of all power outages in the U.S., posing significant risks to vulnerable populations who rely on continuous power for medical devices, as well as causing substantial financial losses to businesses and hampering emergency response operations. Accurate prediction of storm-induced outages is essential, yet existing models struggle particularly with forecasting outages from moderate and low-severity storms. These events often get overlooked in unified models, which fail to capture their distinct characteristics.

To address this, we developed a storm severity-aware prediction framework that routes storm events to dedicated machine learning models based on severity. We used data from EAGLE-I, NOAA Storm Events, and PRISM Climate records from 2014 to 2023 to align storm occurrences with power outages and environmental data. Our preprocessing included Random Forest regression for missing data imputation, normalization with time zone alignment, and elimination of sparse or irrelevant features. Additionally, we applied NLP techniques on event narratives, combining them with structured environmental features.

Our pipeline begins by predicting storm occurrence one hour in advance using XGBoost (F1-score: 0.9308). A subsequent LightGBM model classifies storm severity (F1-score: 0.9329), followed by severity-specific outage prediction models: LightGBM for low severity (0.9878), XGBoost for medium (0.9739), and Random Forest for high severity storms (0.9755). Our models are competitive with state-of-the-art models. We also developed a custom metric aggregating F1 scores across pipeline stages to better reflect real-world performance, achieving a combined score of 0.8503.

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
