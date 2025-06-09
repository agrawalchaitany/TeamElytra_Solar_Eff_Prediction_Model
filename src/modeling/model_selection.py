import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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
    
    
    def tune_xgboost_optuna(self, n_trials=50, cv=5, n_jobs=-1, verbose=1):
        """
        Hyperparameter tuning for XGBoost using Optuna to minimize cross-validated RMSE
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10),
                'random_state': 42,
                'tree_method': 'hist',
                'n_jobs': n_jobs
            }
            model = XGBRegressor(**params)
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=n_jobs)
            return -np.mean(cv_scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose > 0)
        best_params = study.best_params
        best_model = XGBRegressor(**best_params)
        best_model.fit(self.X_train, self.y_train)
        best_score = study.best_value
        print(f"Optuna XGBoost best RMSE: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        self.best_models['xgboost'] = best_model
        self.best_params['xgboost'] = best_params
        self.best_scores['xgboost'] = best_score
        return best_model, best_params, best_score

    def tune_lightgbm_optuna(self, n_trials=50, cv=5, n_jobs=-1, verbose=1):
        """
        Hyperparameter tuning for LightGBM using Optuna to minimize cross-validated RMSE
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 15),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10),
                'random_state': 42,
                'n_jobs': n_jobs
            }
            model = LGBMRegressor(**params)
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=n_jobs)
            return -np.mean(cv_scores)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose > 0)
        best_params = study.best_params
        best_model = LGBMRegressor(**best_params)
        best_model.fit(self.X_train, self.y_train)
        best_score = study.best_value
        print(f"Optuna LightGBM best RMSE: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        self.best_models['lightgbm'] = best_model
        self.best_params['lightgbm'] = best_params
        self.best_scores['lightgbm'] = best_score
        return best_model, best_params, best_score
        
    def train_and_evaluate_all_models(self):
        """Train and evaluate all candidate models with default parameters (no tuning)"""
        # Initialize models with default/fine-tuned defaults
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        # XGBoost
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        xgb_pred = xgb_model.predict(self.X_train)
        xgb_rmse = np.sqrt(mean_squared_error(self.y_train, xgb_pred))
        xgb_cv = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions={},
            n_iter=1,
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring='neg_root_mean_squared_error',
            random_state=42
        )
        xgb_cv.fit(self.X_train, self.y_train)
        xgb_cv_rmse = -xgb_cv.best_score_
        self.best_models['xgboost'] = xgb_model
        self.best_params['xgboost'] = xgb_model.get_params()
        self.best_scores['xgboost'] = xgb_cv_rmse
        # LightGBM
        lgbm_model = LGBMRegressor(
                    n_estimators=528,
                    learning_rate=0.02361368006658606,
                    max_depth=2,
                    num_leaves=101,
                    subsample=0.8535450231075647,
                    colsample_bytree=0.9723004818925387,
                    reg_alpha=0.9723004818925387,
                    reg_lambda=6.292691253538978,
                    random_state=42
        )
        lgbm_model.fit(self.X_train, self.y_train)
        lgbm_pred = lgbm_model.predict(self.X_train)
        lgbm_rmse = np.sqrt(mean_squared_error(self.y_train, lgbm_pred))
        lgbm_cv = RandomizedSearchCV(
            estimator=lgbm_model,
            param_distributions={},
            n_iter=1,
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring='neg_root_mean_squared_error',
            random_state=42
        )
        lgbm_cv.fit(self.X_train, self.y_train)
        lgbm_cv_rmse = -lgbm_cv.best_score_
        self.best_models['lightgbm'] = lgbm_model
        self.best_params['lightgbm'] = lgbm_model.get_params()
        self.best_scores['lightgbm'] = lgbm_cv_rmse

        print("Default models trained and evaluated (CV RMSE computed for each).")
        return self.best_models, self.best_params, self.best_scores
    
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
    selector.tune_xgboost()
    selector.tune_lightgbm()
    
    # Compare and select best model
    selector.compare_models()
    best_model_name, best_model, best_score = selector.select_best_model()

    print(f"\nModel selection complete. Best model: {best_model_name} with CVRMSE: {best_score:.4f}")