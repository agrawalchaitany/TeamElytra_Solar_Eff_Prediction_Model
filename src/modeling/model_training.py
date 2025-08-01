import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ModelTrainer:
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the model trainer
        
        Parameters:
        -----------
        data_dir : str
            Directory containing preprocessed data files
        output_dir : str
            Directory to save trained models
        """
        # Set default directories relative to project root if not provided
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        self.data_dir = data_dir or os.path.join(base_dir, "dataset/processed_data")
        self.output_dir = output_dir or os.path.join(base_dir, "models/all_models")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model dictionary
        self.models = {}
        self.trained_models = {}
    
    def load_data(self):
        """Load preprocessed training data"""
        X_train_path = os.path.join(self.data_dir, "X_Train.csv")
        y_train_path = os.path.join(self.data_dir, "y_train.csv")
        
        print(f"Loading training data from {self.data_dir}")
        self.X_train = pd.read_csv(X_train_path)
        self.y_train = pd.read_csv(y_train_path).values.ravel()  # Convert to 1D array
        
        print(f"Loaded X_train with shape: {self.X_train.shape}")
        print(f"Loaded y_train with shape: {self.y_train.shape}")
        
        return self.X_train, self.y_train
    
    def initialize_models(self, include_all=True, best_params=None):
        """Initialize regression models, optionally with fine-tuned parameters"""
        models = {}
        # Add more advanced models if requested
        if include_all:
            # XGBoost (less overfitting defaults)
            if best_params and 'xgboost' in best_params:
                models['xgboost'] = XGBRegressor(**best_params['xgboost'])
            else:
                models['xgboost'] = XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.008,
                    max_depth=5,
                    min_child_weight=4,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    gamma=0.25,
                    reg_alpha=2.0,
                    reg_lambda=4.0,
                    random_state=42
                )
            # LightGBM (less overfitting defaults)
            if best_params and 'lightgbm' in best_params:
                models['lightgbm'] = LGBMRegressor(**best_params['lightgbm'])
            else:
                models['lightgbm'] = LGBMRegressor(
                    n_estimators=6509,
                    learning_rate=0.007786302796586891,
                    max_depth=2,
                    num_leaves=39,
                    subsample=0.7412306270300187,
                    colsample_bytree=0.8643446505605613,
                    reg_alpha=1.462174878560702,
                    reg_lambda=3.962977085177355,
                    random_state=42
                )
        self.models = models
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
        return self.models
    
    def train_model(self, model_name, model=None, epochs=50, batch_size=32):
        """
        Train a specific model (no Keras NN support)
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        model : sklearn estimator, optional
            Model instance to train. If None, use the model from self.models
        
        Returns:
        --------
        trained_model : fitted estimator
            Trained model
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Initialize models first.")
            model = self.models[model_name]
        
        print(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        print(f"Finished training {model_name}")
        
        # Save the trained model
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self):
        """Train all initialized models"""
        if not self.models:
            raise ValueError("No models initialized. Call initialize_models() first.")
        
        for model_name, model in self.models.items():
            self.train_model(model_name, model)
        
        return self.trained_models
    
    def save_model(self, model_name):
        """Save a trained model to disk"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")
        
        model_path = os.path.join(self.output_dir, f"{model_name}.joblib")
        joblib.dump(self.trained_models[model_name], model_path)
        print(f"Model {model_name} saved to {model_path}")
        
        return model_path
    
    def save_all_models(self):
        """Save all trained models to disk"""
        saved_paths = {}
        for model_name in self.trained_models:
            path = self.save_model(model_name)
            saved_paths[model_name] = path
        
        return saved_paths

# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.load_data()
    trainer.initialize_models()
    trainer.train_all_models()
    trainer.save_all_models()
    print("All models trained and saved successfully!")
