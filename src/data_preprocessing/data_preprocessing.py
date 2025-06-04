import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    def __init__(self, numerical_cols=None, categorical_cols=None, target_col=None, prediction_only=False):
        """
        Initialize the DataPreprocessor with column definitions
        
        Parameters:
        -----------
        numerical_cols : list
            List of numerical column names
        categorical_cols : list
            List of categorical column names
        target_col : str
            Name of the target column
        prediction_only : bool, default=False
            Whether to initialize for prediction only (skip fitting checks)
        """
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.fitted = prediction_only  # Set to True if prediction_only
        self.train_columns = None
        
        # Initialize containers for fitted parameters
        self.medians = {}
        self.modes = {}
        self.bounds = {}
        self.valid_medians = {}
    
    def convert_to_numeric(self, df):
        """Convert specified columns to numeric format"""
        df_copy = df.copy()
        for col in self.numerical_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        return df_copy
       
    def handle_invalid_values(self, df, is_train=True):
        """
        Handle physically impossible or inconsistent values in solar panel measurements
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing solar panel measurements
        is_train : bool, default=True
            Whether the data is training data
        
        Returns:
        --------
        df_copy : pandas DataFrame
            DataFrame with corrected values
        """
        df_copy = df.copy()
        
        if is_train:
            # Store valid medians for imputation
            self.valid_medians = {}
            
            # Calculate median of valid irradiance values (positive values)
            valid_irradiance = df_copy.loc[df_copy['irradiance'] > 0, 'irradiance']
            if not valid_irradiance.empty:
                self.valid_medians['irradiance'] = valid_irradiance.median()
            else:
                self.valid_medians['irradiance'] = 0
                
            # Calculate median of valid voltage values (positive values)
            valid_voltage = df_copy.loc[df_copy['voltage'] > 0, 'voltage']
            if not valid_voltage.empty:
                self.valid_medians['voltage'] = valid_voltage.median()
            else:
                self.valid_medians['voltage'] = 0
                
            # Calculate median of valid current values (positive values)
            valid_current = df_copy.loc[df_copy['current'] > 0, 'current']
            if not valid_current.empty:
                self.valid_medians['current'] = valid_current.median()
            else:
                self.valid_medians['current'] = 0
        
        # 1. Handle irradiance values
        # Replace negative irradiance with 0
        df_copy.loc[df_copy['irradiance'] < 0, 'irradiance'] = 0
        
        # Handle inconsistency: irradiance is 0 but current or voltage is positive
        inconsistent_irradiance = (df_copy['irradiance'] == 0) & ((df_copy['current'] > 0) | (df_copy['voltage'] > 0))
        df_copy.loc[inconsistent_irradiance, 'irradiance'] = self.valid_medians['irradiance']
        
        # 2. Handle voltage values
        # Replace negative voltage with 0
        df_copy.loc[df_copy['voltage'] < 0, 'voltage'] = 0
        
        # Handle inconsistency: voltage is 0 but irradiance and current are positive
        inconsistent_voltage = (df_copy['voltage'] == 0) & (df_copy['irradiance'] > 0) & (df_copy['current'] > 0)
        df_copy.loc[inconsistent_voltage, 'voltage'] = self.valid_medians['voltage']
        
        # 3. Handle current values
        # Replace negative current with 0
        df_copy.loc[df_copy['current'] < 0, 'current'] = 0
        
        # Handle inconsistency: current is 0 but irradiance and voltage are positive
        inconsistent_current = (df_copy['current'] == 0) & (df_copy['irradiance'] > 0) & (df_copy['voltage'] > 0)
        df_copy.loc[inconsistent_current, 'current'] = self.valid_medians['current']
        
        return df_copy    
    
    def handle_missing_values(self, df, is_train=True):
        """
        Fill missing values in the dataframe
        - Numerical columns: median
        - Categorical columns: mode
        """
        df_copy = df.copy()
        
        # Store medians and modes if it's training data
        if is_train:
            self.medians = {}
            self.modes = {}
            
        # Handle numerical columns
        for col in self.numerical_cols:
            if col in df_copy.columns:
                if is_train:
                    self.medians[col] = df_copy[col].median()

                df_copy[col] = df_copy[col].fillna(self.medians[col])
        
        # Handle categorical columns
        for col in self.categorical_cols:
            if col in df_copy.columns:
                if is_train:
                    self.modes[col] = df_copy[col].mode()[0]
                df_copy[col] = df_copy[col].fillna(self.modes[col])
        
        return df_copy
    
    def engineer_features(self, df):
        """Create new features based on domain knowledge of solar panel physics"""
        df_copy = df.copy()
        
        # Calculate power output
        if 'voltage' in df_copy.columns and 'current' in df_copy.columns:
            df_copy['power'] = df_copy['voltage'] * df_copy['current']
        
        # Calculate temperature differential
        if 'temperature' in df_copy.columns and 'module_temperature' in df_copy.columns:
            df_copy['temp_differential'] = df_copy['module_temperature'] - df_copy['temperature']
        
        # Calculate efficiency factors
        if 'irradiance' in df_copy.columns:
            # Calculate power to irradiance ratio (avoid division by zero)
            if 'power' in df_copy.columns:
                df_copy['power_irradiance_ratio'] = df_copy['power'] / df_copy['irradiance'].replace(0, 0.001)
            
            # Calculate soiling impact
            if 'soiling_ratio' in df_copy.columns:
                df_copy['soiling_impact'] = df_copy['soiling_ratio'] * df_copy['irradiance']
        
        # Panel age factors
        if 'panel_age' in df_copy.columns:
            if 'maintenance_count' in df_copy.columns:
                # Avoid division by zero
                df_copy['maintenance_frequency'] = df_copy['maintenance_count'] / df_copy['panel_age'].replace(0, 0.0001)
            
            if 'soiling_ratio' in df_copy.columns:
                df_copy['age_efficiency_factor'] = df_copy['panel_age'] * df_copy['soiling_ratio']
        
        # Environmental interaction features
        if 'humidity' in df_copy.columns and 'temperature' in df_copy.columns:
            df_copy['humidity_temperature_interaction'] = df_copy['humidity'] * df_copy['temperature']
        
        if 'wind_speed' in df_copy.columns and 'temp_differential' in df_copy.columns:
            df_copy['wind_cooling_effect'] = df_copy['wind_speed'] * df_copy['temp_differential']
        
        # Add engineered columns to numerical columns list for scaling
        new_numerical_cols = [
            'power', 'temp_differential', 'power_irradiance_ratio', 'soiling_impact',
            'maintenance_frequency', 'age_efficiency_factor', 
            'humidity_temperature_interaction', 'wind_cooling_effect'
        ]
        for col in new_numerical_cols:
            if col in df_copy.columns and col not in self.numerical_cols:
                self.numerical_cols.append(col)
        
        return df_copy    
    
    def handle_outliers(self, df, method='iqr', is_train=True):
        """
        Handle outliers in numerical columns
        
        Parameters:
        -----------
        df : pandas DataFrame
        method : str, default='iqr'
            Method to handle outliers ('iqr', 'zscore', 'percentile')
        is_train : bool, default=True
            Whether the data is training data
        """
        df_copy = df.copy()
        
        if is_train:
            self.bounds = {}
        
        if method == 'iqr':
            for col in self.numerical_cols:
                if col in df_copy.columns:
                    if is_train:
                        Q1 = df_copy[col].quantile(0.25)
                        Q3 = df_copy[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        self.bounds[col] = (lower_bound, upper_bound)
                    
                    # Apply bounds
                    lower, upper = self.bounds[col]
                    df_copy[col] = np.clip(df_copy[col], lower, upper)
        
        elif method == 'percentile':
            for col in self.numerical_cols:
                if col in df_copy.columns:
                    if is_train:
                        lower_bound = df_copy[col].quantile(0.01)
                        upper_bound = df_copy[col].quantile(0.99)
                        self.bounds[col] = (lower_bound, upper_bound)
                    
                    # Apply bounds
                    lower, upper = self.bounds[col]
                    df_copy[col] = np.clip(df_copy[col], lower, upper)
        
        return df_copy
    
    def encode_categorical(self, df, is_train=True):
        """One-hot encode categorical variables"""
        df_encoded = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        
        if is_train:
            self.train_columns = df_encoded.columns.tolist()
            if self.target_col and self.target_col in self.train_columns:
                self.train_columns.remove(self.target_col)
        
        return df_encoded
    
    def scale_features(self, df, is_train=True):
        """Scale numerical features using RobustScaler"""
        df_copy = df.copy()
        
        # Get columns that are actually in the dataframe
        cols_to_scale = [col for col in self.numerical_cols if col in df_copy.columns]
        
        if is_train:
            self.scaler.fit(df_copy[cols_to_scale])
            self.fitted = True
        
        if self.fitted:
            df_copy[cols_to_scale] = self.scaler.transform(df_copy[cols_to_scale])
        else:
            raise ValueError("Scaler not fitted yet. Process training data first.")
        
        return df_copy
    
    def align_test_columns(self, df):
        """Ensure test data has the same columns as training data"""
        if not self.train_columns:
            raise ValueError("Training columns not set. Process training data first.")
        
        # Add missing columns
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        return df[self.train_columns]
    
    def preprocess(self, df, is_train=True, handle_outliers_method=None, handle_invalid=True):
        """
        Process a single dataframe (training or prediction data)
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame to preprocess
        is_train : bool, default=True
            Whether this is training data (True) or prediction data (False)
        handle_outliers_method : str, optional
            Method to handle outliers ('iqr', 'percentile')
        handle_invalid : bool, default=True
            Whether to handle invalid/impossible values
            
        Returns:
        --------
        processed_df : pandas DataFrame
            Processed dataframe
        """
        # Check if the dataframe is None
        if df is None:
            return None
            
        # Check if we're in prediction mode but preprocessor is not fitted
        if not is_train and not self.fitted:
            raise ValueError("Preprocessor not fitted. Process training data first.")
        
        # Start preprocessing
        df_copy = df.copy()
        df_copy = self.convert_to_numeric(df_copy)

        if handle_invalid:
            df_copy = self.handle_invalid_values(df_copy, is_train=is_train)

        df_copy = self.handle_missing_values(df_copy, is_train=is_train)
        df_copy = self.engineer_features(df_copy)

        if handle_outliers_method:
            df_copy = self.handle_outliers(df_copy, method=handle_outliers_method, is_train=is_train)

        df_copy = self.encode_categorical(df_copy, is_train=is_train)
        df_copy = self.scale_features(df_copy, is_train=is_train)

        # For prediction data, ensure columns match training data
        if not is_train:
            df_copy = self.align_test_columns(df_copy)

        return df_copy

    def save_preprocessed_data(self, df, df_path=None, target_path=None):
        """
        Save preprocessed data to CSV files
        
        Parameters:
        -----------
        df : pandas DataFrame
            Processed training data
        df_path : str
            Path to save training data
        target_path : str, optional
            Path to save target variable separately
        """
        # Save training data (with or without target)
        if self.target_col and target_path:
            # Save X and y separately
            X = df.drop(columns=[self.target_col])
            Y = df[self.target_col]
            X.to_csv(df_path, index=False)
            Y.to_csv(target_path, index=False)
        else:
            # Save full training data
            df.to_csv(df_path, index=False)
        
        print(f"Data saved successfully to {df_path}")
    
    def save_parameters(self, path='models/preprocessor_params.json'):
        """
        Save preprocessing parameters to a JSON file
        
        Parameters:
        -----------
        path : str, default='models/preprocessor_params.json'
            Path to save parameters
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet. Process training data first.")
            
        # Convert numpy types to Python native types for JSON serialization
        params = {
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'target_col': self.target_col,
            'medians': {k: float(v) for k, v in self.medians.items()},
            'modes': self.modes,
            'bounds': {k: (float(v[0]), float(v[1])) for k, v in self.bounds.items()},
            'valid_medians': {k: float(v) for k, v in self.valid_medians.items()},
            'train_columns': self.train_columns,
            'fitted': self.fitted
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
            
        print(f"Preprocessor parameters saved to {path}")
        
    def load_parameters(self, path='models/preprocessor_params.json'):
        """
        Load preprocessing parameters from a JSON file
        
        Parameters:
        -----------
        path : str, default='models/preprocessor_params.json'
            Path to load parameters from
        """
        with open(path, 'r') as f:
            params = json.load(f)
        
        self.numerical_cols = params['numerical_cols']
        self.categorical_cols = params['categorical_cols']
        self.target_col = params['target_col']
        self.medians = params['medians']
        self.modes = params['modes']
        self.bounds = params['bounds']
        self.valid_medians = params['valid_medians']
        self.train_columns = params['train_columns']
        self.fitted = params['fitted']
        
        print(f"Preprocessor parameters loaded from {path}")

    def save_scaler(self, path='models/robust_scaler.joblib'):
        """
        Save the fitted RobustScaler to a file
        
        Parameters:
        -----------
        path : str, default='models/robust_scaler.joblib'
            Path to save the scaler
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted yet. Process training data first.")
        
        import joblib
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the scaler
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")

    def load_scaler(self, path='models/robust_scaler.joblib'):
        """
        Load a previously fitted RobustScaler
        
        Parameters:
        -----------
        path : str, default='models/robust_scaler.joblib'
            Path to load the scaler from
        """
        import joblib
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        
        self.scaler = joblib.load(path)
        self.fitted = True
        print(f"Scaler loaded from {path}")