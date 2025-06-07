import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path

# Import from project
from ..data_preprocessing.data_preprocessing import DataPreprocessor

class SolarEfficiencyPredictor:
    """Solar panel efficiency predictor for production use"""
    
    def __init__(self, model_path=None, preprocessor=None):
        """
        Initialize predictor with model and preprocessor
        
        Parameters:
        -----------
        model_path : str, optional
            Path to trained model file
        preprocessor : DataPreprocessor, optional
            Preprocessor instance for data preparation
        """
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        # Set default model path
        if model_path is None:
            model_path = os.path.join(base_dir, "models/best_model", "best_model.joblib")
            params_path= os.path.join(base_dir, "models", "preprocessor_params.json")
        # Load model
        self.model = joblib.load(model_path)
        
        # Load model info if available
        self.model_info = None
        model_info_path = os.path.join(base_dir, "models/best_model", "best_model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
        
        # Create preprocessor if not provided
        if preprocessor is None:
            # Define columns
            numerical_cols = ['temperature', 'irradiance', 'humidity', 'panel_age', 
                             'maintenance_count', 'soiling_ratio', 'voltage', 'current',
                             'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']
            categorical_cols = ['string_id', 'error_code', 'installation_type']
            target_col = 'efficiency'
            
            preprocessor = DataPreprocessor(
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                prediction_only=True  # Initialize for prediction only
            )
            
            # Try to load preprocessor parameters if available
            params_path = os.path.join(base_dir, "models", "preprocessor_params.json")
            if os.path.exists(params_path):
                try:
                    preprocessor.load_parameters(params_path)
                except Exception as e:
                    print(f"Warning: Could not load preprocessor parameters: {e}")
        
        self.preprocessor = preprocessor
        self.prediction_history = []
    
    def predict(self, data, preprocess=True):
        """
        Make efficiency predictions
        
        Parameters:
        -----------
        data : pandas DataFrame
            New data to predict on
        preprocess : bool, default=True
            Whether to preprocess the data
            
        Returns:
        --------
        predictions : numpy array
            Predicted efficiency values
        """
        # Store input data copy
        data_copy = data.copy()
        
        # Preprocess if needed
        if preprocess and self.preprocessor:
            try:
                # Mark preprocessor as fitted for prediction
                self.preprocessor.fitted = True
                
                # Use simplified preprocess method
                processed_data = self.preprocessor.preprocess(
                    df=data_copy,
                    is_train=False,
                    handle_outliers_method='percentile',
                    handle_invalid=True
                )
            except Exception as e:
                raise ValueError(f"Error during preprocessing: {str(e)}")
        else:
            processed_data = data_copy
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Log prediction event
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'data_shape': data.shape,
            'prediction_count': len(predictions),
            'prediction_mean': float(np.mean(predictions)),
            'prediction_min': float(np.min(predictions)),
            'prediction_max': float(np.max(predictions))
        })
        
        return predictions
    
    def predict_single(self, panel_data):
        """
        Predict efficiency for a single panel
        
        Parameters:
        -----------
        panel_data : dict
            Dictionary with panel features
            
        Returns:
        --------
        float
            Predicted efficiency value
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([panel_data])
        
        # Make prediction
        prediction = self.predict(df)[0]
        
        return prediction
    
    def predict_from_csv(self, csv_path, output_path=None, id_column=None):
        """
        Make predictions on data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to input CSV file
        output_path : str, optional
            Path to save predictions. If None, doesn't save.
        id_column : str, optional
            Name of ID column to include in output
            
        Returns:
        --------
        predictions : pandas DataFrame
            DataFrame with predictions and optional ID column
        """
        # Load data
        data = pd.read_csv(csv_path)
        
        # Extract IDs if needed
        ids = None
        if id_column and id_column in data.columns:
            ids = data[id_column].values
        
        # Make predictions
        preds = self.predict(data)
        
        # Create output DataFrame
        if ids is not None:
            output_df = pd.DataFrame({
                id_column: ids,
                'predicted_efficiency': preds
            })
        else:
            output_df = pd.DataFrame({'predicted_efficiency': preds})
        
        # Save if output path provided
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return output_df
    
    def batch_predict(self, data_list):
        """
        Make predictions on a batch of panels
        
        Parameters:
        -----------
        data_list : list of dict
            List of dictionaries with panel features
            
        Returns:
        --------
        list
            List of predicted efficiency values
        """
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data_list)
        
        # Make predictions
        predictions = self.predict(df).tolist()
        
        return predictions
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return self.model_info
    
    def get_prediction_history(self):
        """Get history of prediction operations"""
        return pd.DataFrame(self.prediction_history)
    
    def get_feature_importance(self):
        """
        Get feature importance if available in model
        
        Returns:
        --------
        DataFrame or None
            Feature importance data if available
        """
        if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
            return None
            
        # Get feature names from preprocessor or model
        feature_names = getattr(self.model, 'feature_names_in_', None)
        if feature_names is None and hasattr(self.preprocessor, 'train_columns'):
            feature_names = self.preprocessor.train_columns
        
        # If no feature names available, use indices
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(100)]  # Arbitrary large number
            
        # Get importance values
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
            if len(importances.shape) > 1:
                importances = importances[0]  # For multioutput models
        
        # Create DataFrame with available features
        n_features = min(len(feature_names), len(importances))
        importance_df = pd.DataFrame({
            'feature': feature_names[:n_features],
            'importance': importances[:n_features]
        })
        
        # Sort by importance
        return importance_df.sort_values('importance', ascending=False)