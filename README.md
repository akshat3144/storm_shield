# StormShield: Predictive Power Outage Forecasting

## Overview

Storms account for 62% of all power outages in the United States, posing significant risks to vulnerable populations who rely on uninterrupted power for medical devices, while also causing major financial losses to businesses and impairing emergency response operations. Accurate prediction of storm-induced power outages is therefore essential. However, existing models often underperform when it comes to forecasting outages caused by moderate and low-severity storms—events that are frequently overlooked in unified modeling approaches that fail to capture their distinct characteristics.

To address this challenge, **StormShield** introduces a comprehensive, severity-aware prediction framework. This framework routes storm events to **dedicated machine learning models** based on their severity. It leverages data from **EAGLE-I** (real-time power outage monitoring), **NOAA Storm Events** (detailed storm records), and **PRISM Climate records** (high-resolution climate data) spanning 2014 to 2023, aligning storm occurrences with corresponding power outages and environmental variables.

---

## Forecasting Approaches: One-to-One vs Many-to-One & Lag Window Design

StormShield supports two core forecasting paradigms, each with distinct modeling and data preparation strategies:

### 1. One-to-One Forecasting

- **Description:** Each input data point (e.g., a storm event at a specific time and location) is used to predict a single target outcome (such as a power outage) at a fixed future time.
- **Implementation:** This approach is implemented in the notebook `notebooks/01_storm_outage_inference_pipeline_[one-to-one-forecasting].ipynb`.
- **Use Case:** Suitable for scenarios where the goal is to make a direct, point-in-time prediction for each event, using only the features available at that moment (including lagged features).

### 2. Many-to-One (Sequence) Forecasting

- **Description:** A sequence of past events (e.g., a time window of storm and climate data) is used to predict a single future outcome. This approach leverages temporal patterns and dependencies across multiple time steps.
- **Implementation:** Realized in `notebooks/02_storm_outage_inference_pipeline_[many-to-one-forecasting].ipynb`, which uses models like LSTM to process sequences.
- **Use Case:** Ideal for capturing complex temporal relationships, such as how a series of storms or environmental changes over time influence the likelihood of an outage.

### Lag Window Engineering

- **What is a Lag Window?**
  - Lag windows are engineered features that capture the state of key variables (e.g., storm occurrence, severity, climate) at previous time steps relative to the prediction target.
  - For example, a 1-hour lag window means the model uses information from 1 hour before the target event to make its prediction.
- **Scripts:**
  - `scripts/01_lag_window_[45-75min].py` and `scripts/02_lag_window_[1-2hr].py` generate lagged features for different time windows, enabling the models to learn from recent history.
- **Importance:**
  - Lag features are critical for both one-to-one and many-to-one forecasting, as they provide temporal context and improve predictive accuracy.

**These two forecasting versions and the lag window engineering are central to StormShield's design, enabling flexible, robust, and temporally-aware outage prediction.**

---

## Lag Window Scripts Explained

StormShield includes two specialized scripts for generating lagged features, which are essential for temporal modeling:

### `scripts/01_lag_window_[45-75min].py`

- **Purpose:**
  - Creates a lagged feature (`IS_STORM_1HR_AHEAD`) that indicates whether a storm will occur in the same state within a window of 45 to 75 minutes (±15 minutes around 1 hour) after each power outage event.
- **How it works:**
  - For each row (event), it checks if any storm event in the same state starts or ends within 45–75 minutes after the current time.
  - If a storm is found in this window, the feature is set to 1; otherwise, it is set to 0 (or None if no record exists).
  - This approach provides a buffer around the 1-hour mark, making the lag feature robust to small timing uncertainties.
- **Use case:**
  - Useful for models that need to predict the likelihood of a storm shortly after a given event, with some tolerance for timing.

### `scripts/02_lag_window_[1-2hr].py`

- **Purpose:**
  - Generates a lagged feature (`is_storm_lagged`) that captures whether a storm occurs exactly 1 to 2 hours after each event, grouped by state.
- **How it works:**
  - For each event, it looks for the next event in the same state that is at least 1 hour ahead (up to 2 hours ahead) and assigns its `is_storm` value to the current row's `is_storm_lagged`.
  - Rows without a valid lagged match are dropped.
- **Use case:**
  - Enables the model to learn from the state of storms in the near future, supporting sequence-based and lagged temporal prediction.

**Both scripts are critical for engineering features that allow the models to leverage recent and near-future storm activity, improving the accuracy of both one-to-one and many-to-one forecasting approaches.**

---

## Key Features

- **Data Preprocessing**: Cleans, merges, and aligns raw data from multiple sources into structured, analysis-ready formats.
- **Feature Engineering**:
  - Creation of lagged variables for temporal prediction.
  - Extraction of NLP-based features from storm event narratives (e.g., keyword flags, word/character counts, lemmatized text).
  - Integration of environmental and storm-specific features.
- **Missing Data Imputation**: Uses Random Forest regression and classification to impute missing values (e.g., storm magnitude, magnitude type) based on related features.
- **Time Zone Normalization**: All event times are converted to Eastern Time for consistency.
- **Modeling Pipeline**:
  1. **Storm Occurrence Prediction** (1 hour in advance) using **XGBoost** (F1-score: 0.9308)
  2. **Storm Severity Classification** using **LightGBM** (F1-score: 0.9329)
  3. **Outage Prediction** via severity-specific models:
     - **LightGBM** for low-severity storms (F1-score: 0.9878)
     - **XGBoost** for medium-severity storms (F1-score: 0.9739)
     - **Random Forest** for high-severity storms (F1-score: 0.9755)
- **Custom Evaluation Metric**: Aggregates F1 scores across all pipeline stages for a realistic end-to-end performance measure (combined score: 0.8503).
- **Hyperparameter Tuning**: Automated with Optuna for optimal model performance.
- **Automation**: Scripts and notebooks streamline the entire workflow from preprocessing to evaluation.
- **Modular Structure**: Organized directories for data, models, preprocessing, and scripts for easy navigation and reproducibility.

---

## Data Sources

- **NOAA Storm Events**: Detailed records of storm events, including type, location, magnitude, and narrative descriptions.
- **EAGLE-I**: Real-time power outage data, aggregated by state and time.
- **PRISM Climate Data**: Daily climate variables (e.g., precipitation, temperature) at high spatial resolution.

---

## Preprocessing Pipeline

1. **Data Loading & Cleaning**
   - Load raw storm, outage, and climate data.
   - Handle missing values, drop irrelevant or sparse columns.
2. **Datetime & Time Zone Alignment**
   - Convert all event times to a common time zone (Eastern Time).
   - Calculate event durations and align storm and outage events temporally.
3. **Merging Datasets**
   - Merge storm and outage data by state and time using as-of joins.
   - Integrate climate data by matching on state and date.
4. **Feature Engineering**
   - Create lagged features for temporal prediction.
   - Extract and clean text features from event narratives (NLP pipeline).
   - Encode categorical variables using target-based encoding.
5. **Imputation**
   - Impute missing storm magnitude and type using Random Forest models.
6. **Splitting & Encoding**
   - Split data into train, test, and holdout sets.
   - Consistently encode categorical features across all splits.

---

## Modeling Pipeline

- **Stage 1: Storm Occurrence Prediction**
  - Predicts whether a storm will occur one hour in advance using XGBoost.
- **Stage 2: Severity Classification**
  - Classifies storm severity (low, moderate, high) using LightGBM, based on imputed magnitude and engineered features.
- **Stage 3: Outage Prediction**
  - Predicts power outage impact using severity-specific models:
    - LightGBM for low-severity
    - XGBoost for medium-severity
    - Random Forest for high-severity

---

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run preprocessing notebooks** (in order):
   - `preprocessing/01_load_clean_merge_data.ipynb`
   - `preprocessing/02_lag_feature_engineering.ipynb`
   - `preprocessing/03_split_and_encode_data.ipynb`
3. **Train and evaluate models** using the notebooks in `models/`.
4. **Run inference** using the provided pipelines in `notebooks/`.

---

## File Structure

```
project_root/
├── data/
│   ├── 01_raw_data.txt
│   ├── 02_interim_processed_data.txt
│   ├── 03_processed_data.txt
│   └── 04_split_and_encoded_data.txt
├── models/
│   ├── 01a_storm_xgb_model.ipynb
│   ├── 01b_storm_lstm_model.ipynb
│   ├── 02_severity_lgm_model.ipynb
│   ├── 03_outage_lgm_xgb_rf_models.ipynb
│   └── 04_models.txt
├── notebooks/
│   ├── 01_storm_outage_inference_pipeline_[one-to-one-forecasting].ipynb
│   └── 02_storm_outage_inference_pipeline_[many-to-one-forecasting].ipynb
├── plots/
│   ├── 01_storm_forecasting/
│   ├── 02_severity_prediction/
│   └── 03_outage_prediction/
├── preprocessing/
│   ├── 01_load_clean_merge_data.ipynb
│   ├── 02_lag_feature_engineering.ipynb
│   └── 03_split_and_encode_data.ipynb
├── scripts/
│   ├── 01_lag_window_[45-75min].py
│   └── 02_lag_window_[1-2hr].py
├── LICENSE
├── pipeline_flowchart.png
├── README.md
└── requirements.txt
```

---

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, lightgbm, dask, matplotlib, seaborn, nltk, optuna, etc.
- See `requirements.txt` for the full list.

---

## Visualization

- All major results (feature importance, ROC/PR curves) are saved in the `plots/` directory for each modeling stage.
- The pipeline flowchart (`pipeline_flowchart.png`) provides a high-level overview of the end-to-end process.

---

## Reproducibility & Extensibility

- All scripts and notebooks are modular and well-documented for easy adaptation to new data or additional features.
- The pipeline can be extended to other regions or event types by updating the data sources and retraining the models.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
