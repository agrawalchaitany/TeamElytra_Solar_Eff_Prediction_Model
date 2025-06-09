import os
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Import our project modules
from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.modeling.model_training import ModelTrainer
from src.modeling.model_evaluation import ModelEvaluator
from src.modeling.model_selection import ModelSelector
from src.utils.terminal_logger import TerminalLogger

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "dataset/processed_data",
        "models",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def preprocess_data(args):
    """Preprocess the data"""
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    # Define columns
    numerical_cols = ['temperature', 'irradiance', 'humidity', 'panel_age', 
                     'maintenance_count', 'soiling_ratio', 'voltage', 'current',
                     'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']
    categorical_cols = ['string_id', 'error_code', 'installation_type']
    target_col = 'efficiency'
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target_col=target_col
    )
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv("dataset/train.csv")
    print(f"Train shape: {train_df.shape}")
    
    # Process data
    print("Preprocessing data...")
    train_processed = preprocessor.preprocess(
        df=train_df,
        is_train=True,
        handle_outliers_method='percentile',
        handle_invalid=True
    )
    
    # Save processed data
    print("Saving preprocessed data...")
    preprocessor.save_preprocessed_data(
        df=train_processed,
        df_path="dataset/processed_data/X_Train.csv",
        target_path="dataset/processed_data/y_train.csv"
    )
    print("Save scaler to .joblib file")
    preprocessor.save_scaler()
    print("Save preprocessing parameters to a JSON file")
    preprocessor.save_parameters(
        path="models/preprocessing_params.json"
    )

    print(f"Preprocessed training data shape: {train_processed.shape}")
    print("Preprocessing complete!")
    
    return True

def train_models(args):
    """Train, evaluate, and save all models in models/all_models"""
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)
    
    trainer = ModelTrainer(
        data_dir="dataset/processed_data",
        output_dir="models/all_models"
    )
    
    # Load data
    trainer.load_data()
    
    # Initialize and train models
    print("Initializing models...")
    trainer.initialize_models(include_all=not args.fast)
    
    print("Training models...")
    start_time = time.time()
    trainer.train_all_models()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save all trained models
    print("Saving trained models...")
    saved_paths = trainer.save_all_models()
    print(f"Saved {len(saved_paths)} models.")
    return True

def evaluate_models(args):
    """Evaluate trained models and ensemble"""
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)
    evaluator = ModelEvaluator(
        data_dir="dataset/processed_data",
        models_dir="models/all_models",
    )
    # Load data
    evaluator.load_data()
    # Load models
    evaluator.load_models()
    # Evaluate all models
    print("Evaluating models...")
    evaluator.evaluate_all_models(cross_validate=True, cv=5)
    # Compare models and visualize results
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison)
    # Save comparison results
    comparison_path = os.path.join("results", "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"Comparison results saved to {comparison_path}")
    # For the best model, show more detailed evaluations
    best_model = comparison.index[0]
    print(f"\nBest model based on CV_RMSE: {best_model}")
    if not args.no_plots:
        print("Generating plots for best model...")
        evaluator.plot_predictions(best_model)
        evaluator.plot_residuals(best_model)
        evaluator.feature_importance(best_model)
        plt.savefig(f"results/{best_model}_feature_importance.png", dpi=300, bbox_inches='tight')
    # Add ensemble results to comparison table and save
    comparison = evaluator.compare_models()
    print("\nUpdated Model Comparison (with Ensemble):")
    print(comparison)
    comparison.to_csv(comparison_path)
    return True

def select_best_model(args):
    """Select and hypertune only the best model, saving it in models/best_model"""
    print("\n" + "="*50)
    print("STEP 4: MODEL SELECTION & HYPERPARAMETER TUNING")
    print("="*50)
    
    selector = ModelSelector(
        data_dir="dataset/processed_data",
        output_dir="models/best_model",
    )
    
    # Load data
    selector.load_data()
    
    # Load all trained models from models/all_models for evaluation
    print("Loading all trained models for evaluation...")
    from src.modeling.model_evaluation import ModelEvaluator
    evaluator = ModelEvaluator(
        data_dir="dataset/processed_data",
        models_dir="models/all_models",
    )
    evaluator.load_data()
    evaluator.load_models()
    # Try to load previous evaluation results if available
    comparison_path = os.path.join("results", "model_comparison.csv")
    if os.path.exists(comparison_path):
        print("Loading previous model evaluation results from model_comparison.csv...")
        comparison = pd.read_csv(comparison_path, index_col=0)
    else:
        print("Evaluating all models (no previous results found)...")
        evaluator.evaluate_all_models(cross_validate=True, cv=5)
        comparison = evaluator.compare_models()
        comparison.to_csv(comparison_path)
    print("\nModel Comparison:")
    print(comparison)
    best_model = comparison.index[0]
    print(f"\nBest model based on CV_RMSE: {best_model}")
    
    # Hypertune only the best model
    print(f"\nFine-tuning the best model: {best_model}")
    if best_model == 'xgboost':
        selector.tune_xgboost_optuna(n_trials=50, cv=5, n_jobs=-1, verbose=1)
    elif best_model == 'lightgbm':
        selector.tune_lightgbm_optuna(n_trials=100, cv=5, n_jobs=-1, verbose=1)
    else:
        print(f"No fine-tuning implemented for model: {best_model}")
    
    # Save the best hypertuned model in models/best_model
    best_model_name, best_model_obj, best_score = selector.select_best_model()
    print(f"\nFinal model selection complete. Best model: {best_model_name} with RMSE: {best_score:.4f}")
    print(f"Best model saved in models/best_model.")
    return True

def run_pipeline(args):
    """Run the complete pipeline"""
    start_time = time.time()
    
    # Create necessary directories
    create_directories()
    
    # Run each step based on arguments
    steps_completed = 0
    
    if args.preprocess:
        if preprocess_data(args):
            steps_completed += 1
    
    if args.train:
        if train_models(args):
            steps_completed += 1
    
    if args.evaluate:
        if evaluate_models(args):
            steps_completed += 1
    
    if args.select:
        if select_best_model(args):
            steps_completed += 1
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETED: {steps_completed} steps executed")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("="*50)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Solar Panel Efficiency Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline control arguments
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing step')
    parser.add_argument('--train', action='store_true', help='Run model training step')
    parser.add_argument('--evaluate', action='store_true', help='Run model evaluation step')
    parser.add_argument('--select', action='store_true', help='Run model selection and tuning step')
    parser.add_argument('--all', action='store_true', help='Run all pipeline steps')
    
    # Additional options
    parser.add_argument('--fast', action='store_true', help='Run in fast mode with fewer models and iterations')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # If no specific steps are selected, run all steps
    if not (args.preprocess or args.train or args.evaluate or args.select):
        args.all = True
    
    # If --all is specified, enable all steps
    if args.all:
        args.preprocess = True
        args.train = True
        args.evaluate = True
        args.select = True
    
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    with TerminalLogger() as logger:
        # Print banner
        print("\n" + "="*50)
        print(f" SOLAR PANEL EFFICIENCY PREDICTION PIPELINE")
        print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # Execute pipeline
        run_pipeline(args)