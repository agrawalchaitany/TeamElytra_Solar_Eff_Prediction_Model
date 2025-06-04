import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

class ModelEvaluator:
    def __init__(self, data_dir=None, models_dir=None):
        """
        Initialize the model evaluator
        
        Parameters:
        -----------
        data_dir : str
            Directory containing preprocessed data files
        models_dir : str
            Directory containing trained models
        """
        # Set default directories relative to project root if not provided
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        self.data_dir = data_dir or os.path.join(base_dir, "dataset/processed_data")
        self.models_dir = models_dir or os.path.join(base_dir, "models/all_models")
        
        # Initialize dictionaries for models and results
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load preprocessed training and test data"""
        X_train_path = os.path.join(self.data_dir, "X_Train.csv")
        y_train_path = os.path.join(self.data_dir, "y_train.csv")
        
        print(f"Loading data from {self.data_dir}")
        self.X_train = pd.read_csv(X_train_path)
        self.y_train = pd.read_csv(y_train_path).values.ravel()
        
        print(f"Loaded X_train with shape: {self.X_train.shape}")
        print(f"Loaded y_train with shape: {self.y_train.shape}")
        
        return self.X_train, self.y_train
    
    def load_models(self, model_names=None):
        """
        Load trained models from disk
        
        Parameters:
        -----------
        model_names : list, optional
            List of model names to load. If None, load all models in the directory.
        """
        if model_names is None:
            # Find all .joblib files in the models directory
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
            model_names = [os.path.splitext(f)[0] for f in model_files]
        
        for model_name in model_names:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_name}")
            else:
                print(f"Warning: Model file not found for {model_name}")
        
        return self.models
    
    def evaluate_model(self, model_name, model=None, cross_validate=True, cv=5):
        """
        Evaluate a single model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate
        model : sklearn estimator, optional
            Model instance to evaluate. If None, use the model from self.models
        cross_validate : bool, default=True
            Whether to perform cross-validation
        cv : int, default=5
            Number of cross-validation folds
        
        Returns:
        --------
        results : dict
            Dictionary containing evaluation metrics
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Load models first.")
            model = self.models[model_name]
        
        results = {}
        
        # Get predictions on training data
        y_train_pred = model.predict(self.X_train)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        results['train_rmse'] = train_rmse
        results['train_mae'] = train_mae
        results['train_r2'] = train_r2
        
        # Cross-validation if requested
        if cross_validate:
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            results['cv_rmse'] = cv_rmse
        
        print(f"--- Evaluation for {model_name} ---")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        if cross_validate:
            print(f"Cross-validation RMSE: {cv_rmse:.4f}")
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def evaluate_all_models(self, cross_validate=True, cv=5):
        """Evaluate all loaded models"""
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model, cross_validate, cv)
        
        return self.results
    
    def plot_predictions(self, model_name, n_samples=100):
        """
        Plot actual vs predicted values for a model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        n_samples : int, default=100
            Number of random samples to plot
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Load models first.")
        
        model = self.models[model_name]
        
        # Get predictions
        y_pred = model.predict(self.X_train)
        
        # Sample a subset of points if needed
        if len(self.y_train) > n_samples:
            idx = np.random.choice(len(self.y_train), n_samples, replace=False)
            y_actual = self.y_train[idx]
            y_predicted = y_pred[idx]
        else:
            y_actual = self.y_train
            y_predicted = y_pred
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_predicted, alpha=0.6)
        
        # Add the perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'Actual vs Predicted Values - {model_name}')
        plt.xlabel('Actual Efficiency')
        plt.ylabel('Predicted Efficiency')
        plt.grid(True)
        
        # Add metrics to the plot
        results = self.results.get(model_name, {})
        if 'cv_rmse' in results:
            plt.annotate(f"CV_RMSE: {results['cv_rmse']:.4f}\nR²: {results['train_r2']:.4f}", 
                         xy=(0.05, 0.95), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"plots/actual_vs_predicted_{model_name}.png", dpi=300, bbox_inches='tight')
        
    
    def plot_residuals(self, model_name):
        """
        Plot residuals for a model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot residuals for
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Load models first.")
        
        model = self.models[model_name]
        
        # Get predictions and calculate residuals
        y_pred = model.predict(self.X_train)
        residuals = self.y_train - y_pred
        residuals_df = pd.DataFrame({'Actual': self.y_train, 'Predicted': y_pred, 'Residuals': residuals})
        # Save residuals to CSV
        residuals_df.to_csv(f"plots/residuals_{model_name}.csv", index=False)
        # Create the plot
        plt.figure(figsize=(12, 5))
        
        # Residuals vs predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        
        # Histogram of residuals
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.6, color='skyblue')
        plt.title('Distribution of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.suptitle(f'Residual Analysis - {model_name}')
        plt.tight_layout()
        plt.savefig(f"plots/residual_analysis_{model_name}.png", dpi=300, bbox_inches='tight')
        
    
    def compare_models(self):
        """Compare all evaluated models based on metrics"""
        if not self.results:
            raise ValueError("No models evaluated yet. Call evaluate_all_models() first.")
        
        # Create dataframe from results
        metrics = ['train_rmse', 'train_mae', 'train_r2', 'cv_rmse']       
        df = pd.DataFrame(index=self.results.keys(), columns=metrics)
        
        for model_name, result in self.results.items():
            for metric in metrics:
                if metric in result:
                    df.loc[model_name, metric] = result[metric]
        
        # Sort by RMSE (lower is better)
        df_sorted = df.sort_values('cv_rmse')
        
        # Create bar plots for comparison
        plt.figure(figsize=(12, 6))
        
        # CV RMSE comparison
        plt.subplot(2, 2, 1)
        df_sorted['cv_rmse'].plot(kind='bar', color='indianred')
        plt.title('Cross-Validation RMSE Comparison (lower is better)')
        plt.ylabel('CV RMSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y')

        # Training RMSE comparison
        plt.subplot(2, 2, 2)
        df_sorted['train_rmse'].plot(kind='bar', color='skyblue')
        plt.title('Training RMSE')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y')

        # R² comparison
        plt.subplot(2, 2, 3)
        df_sorted['train_r2'].plot(kind='bar', color='lightgreen')
        plt.title('R² Comparison (higher is better)')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig("plots/model_comparison.png", dpi=300, bbox_inches='tight')
        
        print("\nModel comparison sorted by cross-validation RMSE (lower is better):")
        print(df_sorted[['cv_rmse', 'train_rmse', 'train_r2']].round(4))
        return df_sorted
    
    def feature_importance(self, model_name):
        """
        Plot feature importance for supported models
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot feature importance for
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Load models first.")
        
        model = self.models[model_name]
        
        # Check if the model supports feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print(f"Model {model_name} doesn't provide feature importance information.")
            return
        
        # Create dataframe with feature names and importance
        feature_names = self.X_train.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(f"plots/feature_importance_{model_name}.csv", index=False)
        # Plot top 20 features or all if less than 20
        n_features = min(20, len(importance_df))
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:n_features][::-1], 
                importance_df['importance'][:n_features][::-1])
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f"plots/feature_importance_{model_name}.png", dpi=300, bbox_inches='tight')
        
        
        return importance_df

# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_data()
    evaluator.load_models()  # Load all models in the directory
    evaluator.evaluate_all_models()
    
    # Compare all models
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison)
    
    # For the best model, show more detailed evaluations
    best_model = comparison.index[0]
    evaluator.plot_predictions(best_model)
    evaluator.plot_residuals(best_model)
    evaluator.feature_importance(best_model)
    
    print(f"\nBest model based on RMSE: {best_model}")