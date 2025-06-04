import os
import pandas as pd
import joblib
import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.data_preprocessing.data_preprocessing import DataPreprocessor

def load_preprocessor():
    """Load or create a preprocessor"""
    # Define columns
    numerical_cols = ['temperature', 'irradiance', 'humidity', 'panel_age', 
                     'maintenance_count', 'soiling_ratio', 'voltage', 'current',
                     'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']
    categorical_cols = ['string_id', 'error_code', 'installation_type']
    target_col = 'efficiency'
    
    return DataPreprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target_col=target_col
    )

def main(input_file, output_file, model_file=None, verbose=False):
    """Make predictions on new data"""
    # Load model
    if model_file is None:
        model_file = os.path.join("models", "best_model.joblib")
    
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at {model_file}")
        return False
    
    model = joblib.load(model_file)
    if verbose:
        print(f"Model loaded from {model_file}")
    
    # Load data
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    input_data = pd.read_csv(input_file)
    original_shape = input_data.shape
    if verbose:
        print(f"Loaded input data with shape: {original_shape}")
    
    # Create preprocessor
    preprocessor = load_preprocessor()
    
    # Process input data (with same preprocessing as during training)
    if verbose:
        print("Preprocessing input data...")
    
    try:
        processed_data = preprocessor.preprocess(
            train_df=None,
            test_df=input_data,
            handle_outliers_method='iqr',
            handle_invalid=True
        )
        if isinstance(processed_data, tuple):
            processed_data = processed_data[1]  # Get test data from tuple
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False
    
    if verbose:
        print(f"Processed data shape: {processed_data.shape}")
    
    # Make predictions
    if verbose:
        print("Making predictions...")
    
    predictions = model.predict(processed_data)
    
    # Create output DataFrame
    if 'id' in input_data.columns:
        output_df = pd.DataFrame({
            'id': input_data['id'],
            'predicted_efficiency': predictions
        })
    else:
        output_df = pd.DataFrame({
            'predicted_efficiency': predictions
        })
    
    # Save predictions
    output_df.to_csv(output_file, index=False)
    if verbose:
        print(f"Predictions saved to {output_file}")
        print(f"Made {len(predictions)} predictions")
    
    return True

def predict_single_panel(panel_data, model_file=None):
    """
    Make a prediction for a single panel
    
    Parameters:
    -----------
    panel_data : dict
        Dictionary with panel feature values
    model_file : str, optional
        Path to model file
        
    Returns:
    --------
    float
        Predicted efficiency
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([panel_data])
    
    # Create temporary files
    temp_input = 'temp_input.csv'
    temp_output = 'temp_output.csv'
    
    # Save input
    df.to_csv(temp_input, index=False)
    
    # Make prediction
    success = main(temp_input, temp_output, model_file, verbose=False)
    
    if not success:
        return None
    
    # Load result
    result = pd.read_csv(temp_output)
    prediction = result['predicted_efficiency'].values[0]
    
    # Clean up
    os.remove(temp_input)
    os.remove(temp_output)
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solar Panel Efficiency Prediction')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='predictions.csv', help='Path to output CSV file')
    parser.add_argument('--model', help='Path to model file (default: models/best_model.joblib)')
    parser.add_argument('--verbose', action='store_true', help='Display detailed output')
    
    args = parser.parse_args()
    main(args.input, args.output, args.model, args.verbose)