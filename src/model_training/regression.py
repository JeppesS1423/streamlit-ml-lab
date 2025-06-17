import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

class RegressorTrainerInitParams:
    model: Literal[
        "linear_regression",
        "random_forest_regressor",
        "support_vector_regressor",
        "gradient_boosting_regressor"
    ]
    hyperparams: dict[str, Any] | None = None

class RegressorTrainer:
    def __init__(self, params: RegressorTrainerInitParams):
        self.params = params
        self.model = self._get_model(params.model, params.hyperparams)
        self.is_trained = False
        self.feature_names = None

    def _get_model(self, model: str, hyperparams: dict[str, Any] | None = None):

        models = {
            "linear_regression": LinearRegression(**hyperparams),
            "random_forest_regressor": RandomForestRegressor(**hyperparams),
            "support_vector_regressor": SVR(**hyperparams),
            "gradient_boosting_regressor": GradientBoostingRegressor(**hyperparams)
        }

        return models[model]

    def train(self, X_train, y_train, X_test=None, y_test=None) -> dict[str, float]:
        """Train the model and return training metrics"""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Store test data for later evaluation if provided
        self.X_test, self.y_test = X_test, y_test
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train)
        }
        
        # Add test metrics if test data provided
        if X_test is not None and y_test is not None:
            y_pred_test = self.model.predict(X_test)
            metrics.update({
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "test_r2": r2_score(y_test, y_pred_test)
            })
            
        return metrics

    def predict(self, X):
        """Make predictions with the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X, y) -> dict[str, float]:
        """Evaluate the model on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        return {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred)
        }
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance for tree-based models"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importance)}
        return None