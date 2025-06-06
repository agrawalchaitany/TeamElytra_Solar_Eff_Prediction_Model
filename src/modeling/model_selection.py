import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class ModelSelector:
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the model selector
        
        Parameters:
        -----------
        data_dir : str
            Directory containing preprocessed data files
        output_dir : str
            Directory to save the best model
        """
        # Set default directories relative to project root if not provided
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        self.data_dir = data_dir or os.path.join(base_dir, "dataset/processed_data")
        self.output_dir = output_dir or os.path.join(base_dir, "models")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize dict for best models
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}

    def load_data(self):
        """Load preprocessed training data"""
        X_train_path = os.path.join(self.data_dir, "X_Train.csv")
        y_train_path = os.path.join(self.data_dir, "y_train.csv")
        
        print(f"Loading training data from {self.data_dir}")
        self.X_train = pd.read_csv(X_train_path)
        self.y_train = pd.read_csv(y_train_path).values.ravel()
        
        print(f"Loaded X_train with shape: {self.X_train.shape}")
        print(f"Loaded y_train with shape: {self.y_train.shape}")
        
        return self.X_train, self.y_train

    
    def tune_xgboost(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Tune XGBRegressor hyperparameters
        
        Parameters:
        -----------
        param_grid : dict, optional
            Grid of parameters to search
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        best_model : fitted estimator
            Best XGBRegressor model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        model = XGBRegressor(random_state=42)
        
        print("Tuning XGBRegressor...")
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring='neg_root_mean_squared_error',
            random_state=42
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back to RMSE
        
        print(f"Best XGBoost RMSE: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        self.best_models['xgboost'] = best_model
        self.best_params['xgboost'] = best_params
        self.best_scores['xgboost'] = best_score
        
        return best_model, best_params, best_score
    
    def tune_lightgbm(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Tune LGBMRegressor hyperparameters
        
        Parameters:
        -----------
        param_grid : dict, optional
            Grid of parameters to search
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        best_model : fitted estimator
            Best LGBMRegressor model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 6, 9, 12],
                'num_leaves': [20, 31, 50],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        model = LGBMRegressor(random_state=42)
        
        print("Tuning LGBMRegressor...")
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring='neg_root_mean_squared_error',
            random_state=42
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back to RMSE
        
        print(f"Best LightGBM RMSE: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        self.best_models['lightgbm'] = best_model
        self.best_params['lightgbm'] = best_params
        self.best_scores['lightgbm'] = best_score
        
        return best_model, best_params, best_score
    
    def tune_gradient_boosting(self, param_grid=None, cv=5, n_jobs=-1, verbose=1):
        """
        Tune GradientBoostingRegressor hyperparameters
        
        Parameters:
        -----------
        param_grid : dict, optional
            Grid of parameters to search
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        best_model : fitted estimator
            Best GradientBoostingRegressor model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9]
            }
        
        model = GradientBoostingRegressor(random_state=42)
        
        print("Tuning GradientBoostingRegressor...")
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring='neg_root_mean_squared_error',
            random_state=42
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back to CV_RMSE

        print(f"Best GradientBoosting CV_RMSE: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        self.best_models['gradient_boosting'] = best_model
        self.best_params['gradient_boosting'] = best_params
        self.best_scores['gradient_boosting'] = best_score
        
        return best_model, best_params, best_score
    
    def select_best_model(self):
        """Select the best model based on lowest CV_RMSE score"""
        if not self.best_scores:
            raise ValueError("No models have been tuned yet.")
        
        # Find model with lowest CV RMSE
        best_model_name = min(self.best_scores, key=self.best_scores.get)
        best_model = self.best_models[best_model_name]
        best_score = self.best_scores[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best Cross-validation RMSE: {best_score:.4f}")
        
        # Save the best model
        best_model_path = os.path.join(self.output_dir, "best_model.joblib")
        joblib.dump(best_model, best_model_path)
        
        # Save model info
        model_info = {
            'model_name': best_model_name,
            'cv_rmse': best_score,
            'parameters': self.best_params[best_model_name]
        }
        
        # Save model info as JSON
        import json
        info_path = os.path.join(self.output_dir, "best_model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"Best model saved to {best_model_path}")
        print(f"Model info saved to {info_path}")
        
        return best_model_name, best_model, best_score
    
    def compare_models(self):
        """Compare all tuned models"""
        if not self.best_scores:
            raise ValueError("No models have been tuned yet.")
        
        # Create dataframe from results
        models = list(self.best_scores.keys())
        cv_rmse_values = [self.best_scores[model] for model in models]
        
        # Sort by CVRMSE (lower is better)
        df = pd.DataFrame({'Model': models, 'CVRMSE': cv_rmse_values})
        df_sorted = df.sort_values('CVRMSE')
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_sorted['Model'], df_sorted['CVRMSE'], color='skyblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom')

        plt.title('Model Comparison (CVRMSE, lower is better)')
        plt.ylabel('CVRMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig(f"plots/model_comparison.png", dpi=300, bbox_inches='tight')

        return df_sorted

# Example usage
if __name__ == "__main__":
    selector = ModelSelector()
    selector.load_data()
    
    # Tune models
    selector.tune_gradient_boosting()
    selector.tune_xgboost()
    selector.tune_lightgbm()
    
    # Compare and select best model
    selector.compare_models()
    best_model_name, best_model, best_score = selector.select_best_model()

    print(f"\nModel selection complete. Best model: {best_model_name} with CVRMSE: {best_score:.4f}")