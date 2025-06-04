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
    test_df = pd.read_csv("dataset/test.csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Process data
    print("Preprocessing data...")
    train_processed, test_processed = preprocessor.preprocess(
        train_df=train_df,
        test_df=test_df,
        handle_outliers_method='iqr',
        handle_invalid=True
    )
    
    # Save processed data
    print("Saving preprocessed data...")
    preprocessor.save_preprocessed_data(
        train_df=train_processed,
        test_df=test_processed,
        train_path="dataset/processed_data/X_Train.csv",
        test_path="dataset/processed_data/X_Test.csv",
        target_path="dataset/processed_data/y_train.csv"
    )
    
    print(f"Preprocessed training data shape: {train_processed.shape}")
    print(f"Preprocessed test data shape: {test_processed.shape}")
    print("Preprocessing complete!")
    
    return True

def train_models(args):
    """Train multiple regression models"""
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)
    
    trainer = ModelTrainer(
        data_dir="dataset/processed_data",
        output_dir="models"
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
    
    # Save models
    print("Saving trained models...")
    saved_paths = trainer.save_all_models()
    print(f"Saved {len(saved_paths)} models.")
    
    return True

def evaluate_models(args):
    """Evaluate trained models"""
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)
    
    evaluator = ModelEvaluator(
        data_dir="dataset/processed_data",
        models_dir="models"
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
    print(f"\nBest model based on RMSE: {best_model}")
    
    if not args.no_plots:
        print("Generating plots for best model...")
        evaluator.plot_predictions(best_model)
        evaluator.plot_residuals(best_model)
        evaluator.feature_importance(best_model)
        
        # Save plots
        plt.savefig(f"results/{best_model}_feature_importance.png", dpi=300, bbox_inches='tight')

    return True

def select_best_model(args):
    """Perform hyperparameter tuning and select best model"""
    print("\n" + "="*50)
    print("STEP 4: MODEL SELECTION & HYPERPARAMETER TUNING")
    print("="*50)
    
    selector = ModelSelector(
        data_dir="dataset/processed_data",
        output_dir="models"
    )
    
    # Load data
    selector.load_data()
    
    # Tune models
    print("Performing hyperparameter tuning...")
    
    if args.fast:
        # Limited tuning for fast mode
        print("Fast mode: Limited hyperparameter tuning")
        selector.tune_random_forest(param_grid={
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        })
        selector.tune_gradient_boosting(param_grid={
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        })
    else:
        # Full tuning
        # selector.tune_random_forest()
        selector.tune_gradient_boosting()
        selector.tune_xgboost()
        selector.tune_lightgbm()
    
    # Compare models
    comparison = selector.compare_models()
    
    # Select and save best model
    best_model_name, best_model, best_score = selector.select_best_model()
    
    print(f"\nModel selection complete. Best model: {best_model_name} with RMSE: {best_score:.4f}")
    
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