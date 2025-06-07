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
        Handles invalid or inconsistent solar panel data using physics-aware rules.
        """
        df_cleaned = df.copy()

        # Drop duplicates
        df_cleaned.drop_duplicates(inplace=True)
                       
        # Handle irradiance
        if 'irradiance' in df_cleaned.columns:
            df_cleaned.loc[df_cleaned['irradiance'] < 0, 'irradiance'] = 0
            df_cleaned.loc[df_cleaned['irradiance'] > 1300, 'irradiance'] = 1300
            # Do NOT drop rows with irradiance < 20, just set to 20
            df_cleaned.loc[df_cleaned['irradiance'] < 20, 'irradiance'] = 20

        # Voltage and current (physics: must be non-negative)
        for col in ['voltage', 'current']:
            if col in df_cleaned.columns:
                # Fill negatives with median instead of NaN
                median_val = df_cleaned[col].median() if not df_cleaned[col].dropna().empty else 0
                df_cleaned.loc[df_cleaned[col] < 0, col] = median_val

        # Temperature cleaning
        if 'temperature' in df_cleaned.columns:
            median_val = df_cleaned['temperature'].median() if not df_cleaned['temperature'].dropna().empty else 20
            df_cleaned.loc[(df_cleaned['temperature'] < -40) | (df_cleaned['temperature'] > 60), 'temperature'] = median_val

        if 'module_temperature' in df_cleaned.columns:
            median_val = df_cleaned['module_temperature'].median() if not df_cleaned['module_temperature'].dropna().empty else 30
            df_cleaned.loc[(df_cleaned['module_temperature'] < -20) | (df_cleaned['module_temperature'] > 90), 'module_temperature'] = median_val
            # Panel temp should not be much lower than ambient (physics-informed check)
            if 'temperature' in df_cleaned.columns:
                too_low = df_cleaned['module_temperature'] < (df_cleaned['temperature'] - 5)
                df_cleaned.loc[too_low, 'module_temperature'] = median_val


        # Save medians for imputation (for test data)
        if is_train:
            for col in ['irradiance', 'voltage', 'current', 'temperature', 'module_temperature']:
                if col in df_cleaned.columns:
                    vals = df_cleaned[col].dropna()
                    self.valid_medians[col] = vals.median() if not vals.empty else 0

        return df_cleaned

    # temp disabled for now,
    def handle_invalid_(self, df, is_train=True):
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
                # Always use the stored median for imputation (from training)
                median = self.medians.get(col, 0)
                df_copy[col] = df_copy[col].fillna(median)
            else:
                # If column missing, add and fill with median
                median = self.medians.get(col, 0)
                df_copy[col] = median
        # Handle categorical columns
        for col in self.categorical_cols:
            if col in df_copy.columns:
                # Always use the stored mode for imputation (from training)
                mode = self.modes.get(col, "")
                df_copy[col] = df_copy[col].fillna(mode)
            else:
                # If column missing, add and fill with mode
                mode = self.modes.get(col, "")
                df_copy[col] = mode
        # Do NOT save medians/modes for test data (is_train=False)
        return df_copy
    
    def engineer_features(self, df):
        """Advanced feature engineering based on physics and data analysis"""
        df_copy = df.copy()

        # Step 1: Base features
        df_copy['power'] = df_copy['voltage'] * df_copy['current']
        df_copy['temp_differential'] = df_copy['module_temperature'] - df_copy['temperature']
        
        # Step 2: Derived physics-based features
        df_copy['power_irradiance_ratio'] = df_copy['power'] / df_copy['irradiance'].replace(0, 0.001)
        df_copy['soiling_impact'] = df_copy['soiling_ratio'] * df_copy['irradiance']
        df_copy['maintenance_frequency'] = df_copy['maintenance_count'] / df_copy['panel_age'].replace(0, 0.0001)
        df_copy['age_efficiency_factor'] = df_copy['panel_age'] * df_copy['soiling_ratio']
        df_copy['humidity_temperature_interaction'] = df_copy['humidity'] * df_copy['temperature']
        df_copy['wind_cooling_effect'] = df_copy['wind_speed'] * df_copy['temp_differential']

        # Step 3: Advanced interactions
        df_copy['performance_efficiency'] = df_copy['power'] / (
            df_copy['irradiance'].replace(0, 0.001) * (1 + df_copy['panel_age'])
        )

        df_copy['soiling_degradation'] = df_copy['soiling_ratio'] * df_copy['panel_age']

        df_copy['environmental_stress'] = (
            df_copy['humidity'] * df_copy['temperature'] * df_copy['cloud_coverage']
        )

        df_copy['wind_chill_effect'] = df_copy['wind_speed'] / (df_copy['module_temperature'] + 1)

        df_copy['pressure_normalized_power'] = df_copy['power'] / df_copy['pressure'].replace(0, 1)

        # Add to numerical columns for scaling if not present
        new_features = [
            'power', 'temp_differential', 'power_irradiance_ratio', 'soiling_impact',
            'maintenance_frequency', 'age_efficiency_factor',
            'humidity_temperature_interaction', 'wind_cooling_effect',
            'performance_efficiency', 'soiling_degradation', 'environmental_stress',
            'wind_chill_effect', 'pressure_normalized_power'
        ]

        for col in new_features:
            if col in df_copy.columns and col not in self.numerical_cols:
                self.numerical_cols.append(col)
        # Do NOT update self.train_columns here
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
                # If the column is missing, add it and fill with median/mode if possible, else raise error
                if col in self.numerical_cols:
                    median = self.medians.get(col, 0)
                    df[col] = median
                elif col in self.categorical_cols:
                    mode = self.modes.get(col, "")
                    df[col] = mode
                else:
                    raise ValueError(f"Missing required feature column '{col}' in test data after preprocessing. This indicates a bug in the preprocessing pipeline.")
        # Drop any extra columns not in train_columns
        extra_cols = [col for col in df.columns if col not in self.train_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols)
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

        # Drop low-importance features AFTER encoding (restore previous drop column names)
        drop_cols = [
            'maintenance_count', 'error_code_E01', 'error_code_E02',
            'installation_type_fixed', 'installation_type_tracking',
            'string_id_C3', 'string_id_B2', 'string_id_D4'
        ]
        to_drop = [col for col in drop_cols if col in df_copy.columns]
        df_copy = df_copy.drop(columns=to_drop)
        # Remove dropped columns from self.numerical_cols if present
        self.numerical_cols = [col for col in self.numerical_cols if col not in to_drop]

        # Set train_columns only once, after all feature engineering, encoding, and dropping (for training only)
        if is_train:
            self.train_columns = df_copy.columns.tolist()
            if self.target_col and self.target_col in self.train_columns:
                self.train_columns.remove(self.target_col)

        df_copy = self.scale_features(df_copy, is_train=is_train)

        # For prediction data, ensure columns match training data
        if not is_train:
            # Align columns: add missing, drop extra
            for col in self.train_columns:
                if col not in df_copy.columns:
                    # If the column is missing, add it and fill with median/mode if possible, else raise error
                    if col in self.numerical_cols:
                        median = self.medians.get(col, 0)
                        df_copy[col] = median
                    elif col in self.categorical_cols:
                        mode = self.modes.get(col, "")
                        df_copy[col] = mode
                    else:
                        raise ValueError(f"Missing required feature column '{col}' in test data after preprocessing. This indicates a bug in the preprocessing pipeline.")
            # Drop any extra columns not in train_columns
            extra_cols = [col for col in df_copy.columns if col not in self.train_columns]
            if extra_cols:
                df_copy = df_copy.drop(columns=extra_cols)
            # Reorder columns to match training data
            df_copy = df_copy[self.train_columns]
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