import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.prediction.predictor import SolarEfficiencyPredictor
from src.modeling.model_evaluation import ModelEvaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict solar panel efficiency')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to save predictions CSV')
    parser.add_argument('--model', help='Path to model file', default='models/best_model/best_model.joblib')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--ensemble', action='store_true', help='Use LightGBM+NN ensemble for prediction')
    return parser.parse_args()

def load_preprocessing_params(params_path):
    """Load preprocessing parameters from JSON file"""
    with open(params_path, 'r') as f:
        return json.load(f)

def initialize_preprocessor(params_path):
    """Initialize preprocessor with parameters from JSON file"""
    params = load_preprocessing_params(params_path)
    
    # Create preprocessor with column definitions
    preprocessor = DataPreprocessor(
        numerical_cols=params['numerical_cols'],
        categorical_cols=params['categorical_cols'],
        target_col=params['target_col'],
        prediction_only=True
    )
    
    # Set saved parameters directly on the preprocessor
    preprocessor.medians = params['medians']
    preprocessor.modes = params['modes']
    preprocessor.bounds = {k: tuple(v) for k, v in params['bounds'].items()}
    preprocessor.valid_medians = params['valid_medians']
    preprocessor.train_columns = params['train_columns']
    preprocessor.fitted = True

    # Try to load the scaler
    scaler_path = Path(project_root) / "models" / "robust_scaler.joblib"
    if scaler_path.exists():
        preprocessor.load_scaler(scaler_path)
    
    return preprocessor

def main():
    """Main prediction function"""
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize preprocessor with parameters from JSON
    params_path = Path(project_root) / "models" / "preprocessing_params.json"
    if not params_path.exists():
        print(f"Error: Preprocessing parameters file not found: {params_path}")
        return False
        
    preprocessor = initialize_preprocessor(params_path)
    
    # Load input data
    try:
        input_data = pd.read_csv(args.input)
        if args.verbose:
            print(f"Loaded input data with shape: {input_data.shape}")
            print(f"Input columns: {input_data.columns.tolist()}")
    except Exception as e:
        print(f"Error loading input data: {e}")
        return False
    
    # ENSEMBLE PREDICTION
    if getattr(args, 'ensemble', False):
        if args.verbose:
            print("Using LightGBM+XGBoost ensemble for prediction...")
        # Load both models using ModelEvaluator
        evaluator = ModelEvaluator(
            data_dir="dataset/processed_data",
            models_dir="models/all_models"
        )
        evaluator.load_data()
        evaluator.load_models(model_names=["lightgbm", "xgboost"])
        # Preprocess input
        processed_data = preprocessor.preprocess(
            df=input_data,
            is_train=False,
            handle_outliers_method='percentile',
            handle_invalid=True
        )
        predictions = evaluator.predict_ensemble(processed_data, model_names=["lightgbm", "xgboost"])
    else:
        # Initialize predictor with our preprocessor
        try:
            predictor = SolarEfficiencyPredictor(
                model_path=args.model,
                preprocessor=preprocessor
            )
            if args.verbose:
                print(f"Model loaded from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        # Make predictions
        try:
            if args.verbose:
                print("Preprocessing input data and making predictions...")
            # Save processed test data for inspection
            processed_data = preprocessor.preprocess(input_data, is_train=False)
            processed_data.to_csv("dataset/processed_data/processed_test.csv", index=False)
            predictions = predictor.predict(input_data)
            
            if args.verbose:
                print(f"Generated {len(predictions)} predictions")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return False
    
    # Create output DataFrame
    try:
        # Ensure predictions is 1D
        predictions_1d = np.array(predictions).flatten()
        output_df = pd.DataFrame({
            'id': range(len(predictions_1d)),       
            'efficiency': predictions_1d    
        })      
        # Save predictions
        output_df.to_csv(args.output, index=False)
        
        if args.verbose:
            print(f"Predictions saved to {args.output}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)