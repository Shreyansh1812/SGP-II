import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False
    logging.warning("XGBoost not found, using RandomForestClassifier as fallback.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLModelTrainer:
    """Trainer class to handle model training and evaluation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.features = []

    def train_xgboost_model(self, historical_df: pd.DataFrame, model_output_path: str) -> Dict[str, Any]:
        """
        Trains an XGBoost model (or RandomForest fallback) using TimeSeriesSplit.

        Args:
            historical_df: DataFrame with features and target ('Target_Buy').
            model_output_path: Path to save the trained model.

        Returns:
            Dictionary containing evaluation metrics and metadata.
        """
        if 'Target_Buy' not in historical_df.columns:
            raise ValueError("Target column 'Target_Buy' not found in DataFrame.")

        # Separate features and target
        X = historical_df.drop(columns=['Target_Buy', 'Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')
        y = historical_df['Target_Buy']
        self.features = list(X.columns)

        if len(X) == 0:
            raise ValueError("No valid features found for training.")

        tscv = TimeSeriesSplit(n_splits=5)

        metrics = {
            'roc_auc': [],
            'precision': [],
            'recall': [],
            'brier_score': []
        }

        # Instantiate model
        if HAS_XGB:
            # XGBoost doesn't strictly need class_weight but scale_pos_weight can be used if imbalanced.
            # We'll use default parameters for this implementation.
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=self.random_state
            )

        # Cross-validation
        logging.info("Starting TimeSeries Cross-Validation...")
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Skip fold if only one class is present in y_true (can happen in small datasets or weird splits)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logging.warning(f"Fold {fold} skipped due to single class in target.")
                continue

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
            # zero_division=0 to prevent warnings if precision is undefined
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['brier_score'].append(brier_score_loss(y_test, y_prob))

        # Final training on ALL data
        logging.info("Training final model on all data...")
        model.fit(X, y)
        self.model = model

        # Calculate mean metrics across folds
        mean_metrics = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}
        logging.info(f"Cross-Validation Metrics: {mean_metrics}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(model_output_path) if os.path.dirname(model_output_path) else '.', exist_ok=True)

        # Save model pipeline and metadata
        saved_data = {
            'model': self.model,
            'features': self.features,
            'metrics': mean_metrics,
            'model_type': 'XGBClassifier' if HAS_XGB else 'RandomForestClassifier'
        }

        joblib.dump(saved_data, model_output_path)
        logging.info(f"Model and metadata saved to {model_output_path}")

        return {
            'metrics': mean_metrics,
            'features': self.features,
            'model_path': model_output_path
        }
