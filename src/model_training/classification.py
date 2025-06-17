import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)

class ClassifierTrainerInitParams(BaseModel):
    model: Literal[
        "logistic_regression",
        "random_forest_classifier", 
        "support_vector_classifier",
        "gradient_boosting_classifier"
    ]
    hyperparams: dict[str, Any] | None = None

class ClassifierTrainer:
    def __init__(self, params: ClassifierTrainerInitParams):
        self.params = params
        self.model = self._get_model(params.model, params.hyperparams)
        self.is_trained = False
        self.feature_names = None
        self.classes_ = None

    def _get_model(self, model: str, hyperparams: dict[str, Any] | None = None):
        if hyperparams is None:
            hyperparams = {}
            
        models = {
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "support_vector_classifier": SVC,
            "gradient_boosting_classifier": GradientBoostingClassifier
        }

        return models[model](**hyperparams)

    def train(self, X_train, y_train, X_test=None, y_test=None) -> dict[str, float]:
        """Train the model and return training metrics"""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Store test data for later evaluation if provided
        self.X_test, self.y_test = X_test, y_test
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred_train, prefix="train_")
        
        # Add test metrics if test data provided
        if X_test is not None and y_test is not None:
            y_pred_test = self.model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, y_pred_test, prefix="test_")
            metrics.update(test_metrics)
            
        return metrics

    def _calculate_metrics(self, y_true, y_pred, prefix="") -> dict[str, float]:
        """Calculate classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
        
        # Handle binary vs multiclass classification
        is_binary = len(np.unique(y_true)) == 2
        average = 'binary' if is_binary else 'weighted'
        
        metrics[f"{prefix}precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f"{prefix}recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f"{prefix}f1_score"] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Add macro averages for multiclass
        if not is_binary:
            metrics[f"{prefix}precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics[f"{prefix}recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics[f"{prefix}f1_score_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # ROC AUC for binary classification or if model has predict_proba
        try:
            if is_binary and hasattr(self.model, 'predict_proba'):
                if hasattr(self, 'X_test') and prefix == "test_":
                    y_proba = self.model.predict_proba(self.X_test)[:, 1]
                elif prefix == "train_":
                    # We need the training data for this, but we don't store it
                    # Skip ROC AUC for training metrics to avoid complexity
                    pass
                else:
                    y_proba = self.model.predict_proba(self._get_X_for_metrics())[:, 1]
                if 'y_proba' in locals():
                    metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            # Skip ROC AUC if there are issues
            pass
            
        return metrics

    def predict(self, X):
        """Make predictions with the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities (if supported by the model)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y) -> dict[str, float]:
        """Evaluate the model on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        return self._calculate_metrics(y, y_pred)
    
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
    
    def get_confusion_matrix(self, X, y) -> np.ndarray:
        """Get confusion matrix for the given data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def get_classification_report(self, X, y, output_dict=False) -> str | dict:
        """Get detailed classification report"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=output_dict)