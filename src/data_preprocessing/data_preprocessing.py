import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    def __init__(self, numerical_cols=None, categorical_cols=None, target_col=None):
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
        """
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.fitted = False
        self.train_columns = None
    
    def convert_to_numeric(self, df):
        """Convert specified columns to numeric format"""
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
       
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
                df_copy['maintenance_frequency'] = df_copy['maintenance_count'] / df_copy['panel_age'].replace(0, 0.001)
            
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
    
    def preprocess(self, train_df, test_df=None, handle_outliers_method=None, handle_invalid=True):
        """
        Main preprocessing function to process both training and test data
        
        Parameters:
        -----------
        train_df : pandas DataFrame
            Training dataframe
        test_df : pandas DataFrame, optional
            Test dataframe
        handle_outliers_method : str, optional
            Method to handle outliers ('iqr', 'percentile')
        handle_invalid : bool, default=True
            Whether to handle invalid/impossible values in solar panel measurements
            
        Returns:
        --------
        train_df_processed : pandas DataFrame
            Processed training data
        test_df_processed : pandas DataFrame, optional
            Processed test data (if provided)
        """
        # Process training data
        train_copy = train_df.copy()
        
        # Convert to numeric
        train_copy = self.convert_to_numeric(train_copy)
        
        # Handle physically impossible values if requested
        if handle_invalid:
            train_copy = self.handle_invalid_values(train_copy, is_train=True)
        
        # Handle missing values
        train_copy = self.handle_missing_values(train_copy, is_train=True)
        
        # Engineer features (after handling missing values)
        train_copy = self.engineer_features(train_copy)  
             
        # Handle outliers if specified
        if handle_outliers_method:
            train_copy = self.handle_outliers(train_copy, method=handle_outliers_method, is_train=True)
        
        # Encode categorical variables
        train_copy = self.encode_categorical(train_copy, is_train=True)
        
        # Scale features
        train_copy = self.scale_features(train_copy, is_train=True)
        
        # If test data provided, process it too
        if test_df is not None:
            test_copy = test_df.copy()
            
            # Convert to numeric
            test_copy = self.convert_to_numeric(test_copy)
            
            # Handle physically impossible values if requested
            if handle_invalid:
                test_copy = self.handle_invalid_values(test_copy, is_train=False)
            
            # Handle missing values
            test_copy = self.handle_missing_values(test_copy, is_train=False)
            
            # Engineer features (after handling missing values)
            test_copy = self.engineer_features(test_copy)        
            
            # Handle outliers if specified
            if handle_outliers_method:
                test_copy = self.handle_outliers(test_copy, method=handle_outliers_method, is_train=False)
            
            # Encode categorical variables
            test_copy = self.encode_categorical(test_copy, is_train=False)
            
            # Scale features
            test_copy = self.scale_features(test_copy, is_train=False)
            
            # Ensure columns match training data
            test_copy = self.align_test_columns(test_copy)
            
            return train_copy, test_copy
        
        return train_copy
    
    def save_preprocessed_data(self, train_df, test_df=None, train_path="Clean_X_Train.csv", 
                               test_path="Clean_Test_Data.csv", target_path=None):
        """
        Save preprocessed data to CSV files
        
        Parameters:
        -----------
        train_df : pandas DataFrame
            Processed training data
        test_df : pandas DataFrame, optional
            Processed test data
        train_path : str
            Path to save training data
        test_path : str
            Path to save test data
        target_path : str, optional
            Path to save target variable separately
        """
        # Save training data (with or without target)
        if self.target_col and target_path:
            # Save X and y separately
            X_train = train_df.drop(columns=[self.target_col])
            y_train = train_df[self.target_col]
            X_train.to_csv(train_path, index=False)
            y_train.to_csv(target_path, index=False)
        else:
            # Save full training data
            train_df.to_csv(train_path, index=False)
        
        # Save test data if provided
        if test_df is not None:
            test_df.to_csv(test_path, index=False)
        
        print(f"Data saved successfully to {train_path}")