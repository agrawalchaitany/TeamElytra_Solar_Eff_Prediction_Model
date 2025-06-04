# Solar Panel Efficiency Regression Pipeline

A robust, modular machine learning pipeline for predicting solar panel efficiency, featuring domain-specific preprocessing, advanced feature engineering, model training, evaluation, selection, and easy batch prediction from CSV files.

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
│   ├── Clean_X_Train.csv
│   ├── Clean_Test_Data.csv
│   ├── Efficiency_y_train.csv
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
│   │   └── data_analysis.ipynb
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
│   │   ├── gradient_boosting.joblib
│   │   ├── knn.joblib
│   │   ├── lightgbm.joblib
│   │   ├── linear.joblib
│   │   ├── random_forest.joblib
│   │   ├── svr.joblib
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
python predict.py --input dataset/Clean_Test_Data.csv --output predictions.csv --verbose
```
- `--input`: Path to your input CSV file (columns must match training features)
- `--output`: Path to save predictions (default: predictions.csv)
- `--verbose`: Show detailed progress (optional)

### Python API
```python
from src.prediction.predictor import SolarEfficiencyPredictor
predictor = SolarEfficiencyPredictor()
preds_df = predictor.predict_from_csv(
    csv_path="dataset/Clean_Test_Data.csv",
    output_path="predictions.csv"
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
- **Model Training**: Trains multiple regression models (Linear, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, LightGBM, SVR, KNN)
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
```

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** This pipeline is designed specifically for solar panel efficiency prediction and incorporates domain knowledge about photovoltaic systems. The feature engineering and data preprocessing steps are tailored for solar energy applications.
