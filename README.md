# Solar Panel Efficiency Regression Pipeline

A robust, modular machine learning pipeline for predicting solar panel efficiency, featuring domain-specific preprocessing, advanced feature engineering, model training, evaluation, selection, and easy batch prediction from CSV files.

---

## 🧠 Approach

This pipeline follows a modular, end-to-end machine learning workflow tailored for solar panel efficiency prediction. It emphasizes:
- Careful data cleaning and imputation
- Domain-specific feature engineering
- Training and comparing multiple regression models
- Rigorous evaluation and model selection
- Saving all artifacts for reproducible batch and API predictions

Each step is implemented as a separate, reusable module for clarity and maintainability.

## 🛠️ Feature Engineering Details

Feature engineering is designed with photovoltaic domain knowledge, including:
- **Derived Features**: Calculating power (voltage × current), temperature differentials, and soiling-adjusted irradiance
- **Categorical Encoding**: One-hot encoding for string_id, error_code, and installation_type
- **Temporal Features**: Extracting time-based features if timestamps are present (e.g., hour, day, season)
- **Interaction Terms**: Combining features such as humidity × temperature to capture non-linear effects
- **Scaling**: RobustScaler is used to handle outliers in sensor data

All feature engineering logic is implemented in `src/data_preprocessing/data_preprocessing.py`.

## 🧰 Tools Used

- **Data Processing**: numpy, pandas
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn, xgboost, lightgbm
- **Persistence**: joblib
- **Utilities**: Custom logging (`src/utils/terminal_logger.py`)

## 📂 Source File References

- **Data Preprocessing & Feature Engineering**: `src/data_preprocessing/data_preprocessing.py`
- **EDA**: `src/eda/data_exploration.ipynb`
- **Model Training**: `src/modeling/model_training.py`
- **Model Evaluation**: `src/modeling/model_evaluation.py`
- **Model Selection & Tuning**: `src/modeling/model_selection.py`
- **Prediction API**: `src/prediction/predictor.py`
- **Batch Prediction Script**: `predict.py`
- **Pipeline Orchestration**: `main.py`

---

## 📦 Project Structure

```
Solar_Eff_Prediction_Model/
├── main.py                  # Main pipeline orchestration script
├── predict.py               # Command-line batch prediction script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── dataset/                 # Raw and processed datasets
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── processed_data/
│       ├── X_Train.csv
│       ├── y_train.csv
│
├── src/
│   ├── data_preprocessing/
│   │   └── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── eda/
│   │   ├── data_exploration.ipynb  # EDA notebook
│   ├── modeling/
│   │   ├── model_training.py       # Model training
│   │   ├── model_evaluation.py     # Model evaluation
│   │   └── model_selection.py      # Model selection & tuning
│   ├── prediction/
│   │   └── predictor.py            # Prediction class for new data
│   └── utils/
│       └── terminal_logger.py      # Terminal logging utility
│
├── models/                 # Trained model artifacts and preprocessing
│   ├── best_model/
│   │   ├── best_model.joblib
│   │   └── best_model_info.json
│   ├── all_models/
│   │   ├── lightgbm.joblib
│   │   └── xgboost.joblib
│   ├── preprocessing_params.json   # Preprocessing parameters
│   └── robust_scaler.joblib        # Saved RobustScaler for prediction
│
├── logs/                   # Training and pipeline logs
│   └── training_log_*.txt
│
├── results/                # Model comparison, feature importance, etc.
│   ├── model_comparison.csv
│   ├── submission.csv
│   └── *.png
│
├── plots/                  # Visualization outputs
│   └── ...
```

---

## 🚀 Quick Start

### 1. Install dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the full pipeline
```powershell
python main.py --all
```

### 3. Run specific steps
```powershell
python main.py --preprocess
python main.py --train
python main.py --evaluate
python main.py --select
```

---

## 🔮 Predicting on New Data (CSV)

### Command Line
```powershell
python predict.py --input dataset/test.csv --output results/predictions.csv --verbose
```
- `--input`: Path to your input CSV file (columns must match training features)
- `--output`: Path to save predictions (default: predictions.csv)
- `--verbose`: Show detailed progress (optional)

### Python API
```python
from src.prediction.predictor import SolarEfficiencyPredictor
predictor = SolarEfficiencyPredictor()
preds_df = predictor.predict_from_csv(
    csv_path="dataset/test.csv",
    output_path="results/predictions.csv"
)
print(preds_df.head())
```

**Input CSV columns must match those used in training:**
```
temperature,irradiance,humidity,panel_age,maintenance_count,soiling_ratio,voltage,current,module_temperature,cloud_coverage,wind_speed,pressure,string_id,error_code,installation_type
```

---

## 🧩 Pipeline Components

- **Data Preprocessing**: Cleans, imputes, and engineers features from raw data. Saves preprocessing parameters and scaler for consistent prediction.
- **Model Training**: Trains multiple regression models (XGBoost, LightGBM)
- **Model Evaluation**: Evaluates models using cross-validation RMSE and other metrics. Only model files (not scalers) are loaded for evaluation.
- **Model Selection**: Hyperparameter tuning and best model selection (by CV RMSE)
- **Prediction**: Predicts efficiency for new data using the trained model and saved preprocessing/scaler artifacts
- **Logging**: All terminal output and key events are saved in the `logs/` directory

---

## 📊 Results & Visualization
- Model comparison and feature importance plots are saved in `results/` and `plots/`
- All trained models and preprocessing artifacts are saved in `models/`
- Logs for each run are in `logs/`

---

## 📋 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
joblib
tqdm
optuna
```

---
## 📌 Important Notes
**Note:** This pipeline is designed specifically for solar panel efficiency prediction and incorporates domain knowledge about photovoltaic systems. The feature engineering and data preprocessing steps are tailored for solar energy applications.
