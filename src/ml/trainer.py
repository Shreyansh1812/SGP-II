"""
src/ml/trainer.py
==================

ML Model Training Pipeline with TimeSeriesSplit Cross-Validation.

Provides both single-ticker and multi-ticker (universal) training modes.
The universal model pools OHLCV data from all 40 mega-cap tickers to build
a single, sector-diversified classifier.

Training Pipeline
-----------------
1. Download OHLCV data per ticker (or receive pre-computed DataFrame).
2. Compute technical features via ``compute_technical_features()``.
3. Pool features across tickers (universal mode).
4. Handle class imbalance via ``scale_pos_weight``.
5. Train XGBoost (primary) or RandomForest (fallback) with
   ``TimeSeriesSplit(n_splits=5)`` cross-validation.
6. Evaluate: ROC-AUC, Precision, Recall, Brier Score per fold.
7. Serialize model + metadata + feature importances to ``.pkl``.

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier

    HAS_XGB = False
    logging.getLogger(__name__).warning(
        "XGBoost not found, using RandomForestClassifier as fallback."
    )

from src.ml.feature_engine import compute_technical_features

logger = logging.getLogger(__name__)

# Feature columns to exclude from the training matrix
_OHLCV_COLS = {"Open", "High", "Low", "Close", "Volume"}


class MLModelTrainer:
    """
    Trainer class for SGP-II ML signal model.

    Supports both single-ticker training (for quick experiments) and
    multi-ticker universal training (production mode that pools data
    across the entire 40-stock mega-cap universe).

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility (default 42).

    Attributes
    ----------
    model : object
        Trained sklearn/xgboost classifier (None until training).
    features : List[str]
        Ordered list of feature column names used in training.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.features: List[str] = []

    # ==================================================================
    # SINGLE-TICKER TRAINING (legacy / quick experiment)
    # ==================================================================

    def train_xgboost_model(
        self,
        historical_df: pd.DataFrame,
        model_output_path: str,
    ) -> Dict[str, Any]:
        """
        Train an XGBoost/RandomForest model on a single feature DataFrame.

        This is the original training method suitable for single-ticker
        experiments. For production use, prefer ``train_multi_ticker_model()``.

        Parameters
        ----------
        historical_df : pd.DataFrame
            DataFrame with computed features and ``Target_Buy`` column.
        model_output_path : str
            File path to save the serialized model artifact.

        Returns
        -------
        Dict[str, Any]
            Training result containing metrics, features, and model path.

        Raises
        ------
        ValueError
            If ``Target_Buy`` column is missing or no valid features found.
        """
        if "Target_Buy" not in historical_df.columns:
            raise ValueError("Target column 'Target_Buy' not found in DataFrame.")

        # Separate features and target
        drop_cols = list(_OHLCV_COLS | {"Target_Buy"})
        X = historical_df.drop(columns=drop_cols, errors="ignore")
        y = historical_df["Target_Buy"]
        self.features = list(X.columns)

        if len(X) == 0:
            raise ValueError("No valid features found for training.")

        # Train with cross-validation
        metrics, model = self._train_with_cv(X, y)

        # Save model artifact
        self.model = model
        self._save_model(
            model=model,
            features=self.features,
            metrics=metrics,
            model_output_path=model_output_path,
            tickers_used=["single_ticker"],
            training_date_range=(
                str(historical_df.index.min()),
                str(historical_df.index.max()),
            ),
            samples_per_class={
                "class_0": int((y == 0).sum()),
                "class_1": int((y == 1).sum()),
            },
        )

        return {
            "metrics": metrics["mean_metrics"],
            "features": self.features,
            "model_path": model_output_path,
        }

    # ==================================================================
    # MULTI-TICKER UNIVERSAL TRAINING (production mode)
    # ==================================================================

    def train_multi_ticker_model(
        self,
        tickers: List[str],
        years: int,
        model_output_path: str,
    ) -> Dict[str, Any]:
        """
        Train a universal model on pooled data from multiple tickers.

        Downloads OHLCV data for each ticker, computes features independently,
        then concatenates all feature matrices into a single training set.
        The model learns ticker-agnostic technical patterns.

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to download and train on.
        years : int
            Number of years of historical data per ticker.
        model_output_path : str
            File path to save the serialized model artifact.

        Returns
        -------
        Dict[str, Any]
            Training result containing metrics, features, model path,
            and per-ticker sample counts.

        Raises
        ------
        ValueError
            If no valid training data is produced from any ticker.
        """
        logger.info(
            f"Starting multi-ticker training: {len(tickers)} tickers × {years}y data."
        )

        all_feature_dfs: List[pd.DataFrame] = []
        ticker_sample_counts: Dict[str, int] = {}
        failed_tickers: List[str] = []
        date_min = None
        date_max = None

        for idx, ticker in enumerate(tickers, start=1):
            logger.info(f"[{idx}/{len(tickers)}] Downloading OHLCV for {ticker}...")
            try:
                ohlcv = yf.download(
                    ticker.upper(),
                    period=f"{years}y",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )

                if ohlcv is None or ohlcv.empty:
                    logger.warning(f"No data returned for {ticker}. Skipping.")
                    failed_tickers.append(ticker)
                    continue

                # Handle multi-level columns from yfinance
                if isinstance(ohlcv.columns, pd.MultiIndex):
                    ohlcv.columns = ohlcv.columns.get_level_values(0)

                # Compute features for this ticker
                feature_df = compute_technical_features(ohlcv)

                if feature_df.empty:
                    logger.warning(
                        f"Feature computation produced empty result for {ticker}."
                    )
                    failed_tickers.append(ticker)
                    continue

                # Track date range
                if date_min is None or feature_df.index.min() < date_min:
                    date_min = feature_df.index.min()
                if date_max is None or feature_df.index.max() > date_max:
                    date_max = feature_df.index.max()

                ticker_sample_counts[ticker] = len(feature_df)
                all_feature_dfs.append(feature_df)
                logger.info(f"  {ticker}: {len(feature_df)} samples extracted.")

            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                failed_tickers.append(ticker)

        if not all_feature_dfs:
            raise ValueError(
                "No valid training data produced from any ticker. "
                f"All {len(tickers)} tickers failed."
            )

        # Concatenate all ticker data
        combined_df = pd.concat(all_feature_dfs, axis=0, ignore_index=True)
        logger.info(
            f"Pooled training set: {len(combined_df)} total samples "
            f"from {len(all_feature_dfs)} tickers."
        )

        if failed_tickers:
            logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

        # Separate features and target
        drop_cols = list(_OHLCV_COLS | {"Target_Buy"})
        X = combined_df.drop(columns=drop_cols, errors="ignore")
        y = combined_df["Target_Buy"]
        self.features = list(X.columns)

        if len(X) == 0:
            raise ValueError("No valid features found after pooling.")

        # Train with cross-validation
        metrics, model = self._train_with_cv(X, y)
        self.model = model

        # Compute feature importances
        feature_importances = self._get_feature_importances(model, self.features)

        # Build samples_per_class
        samples_per_class = {
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
        }

        # Save model artifact
        self._save_model(
            model=model,
            features=self.features,
            metrics=metrics,
            model_output_path=model_output_path,
            tickers_used=list(ticker_sample_counts.keys()),
            training_date_range=(str(date_min), str(date_max)),
            samples_per_class=samples_per_class,
            ticker_sample_counts=ticker_sample_counts,
            failed_tickers=failed_tickers,
            feature_importances=feature_importances,
        )

        return {
            "metrics": metrics["mean_metrics"],
            "per_fold_metrics": metrics["per_fold_metrics"],
            "features": self.features,
            "feature_importances": feature_importances,
            "model_path": model_output_path,
            "tickers_trained": list(ticker_sample_counts.keys()),
            "tickers_failed": failed_tickers,
            "total_samples": len(combined_df),
            "samples_per_class": samples_per_class,
        }

    # ==================================================================
    # INTERNAL: Cross-validation training loop
    # ==================================================================

    def _train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple:
        """
        Train a model using TimeSeriesSplit cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target series.

        Returns
        -------
        tuple
            (metrics_dict, trained_model) where metrics_dict contains both
            per-fold and mean metrics.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        per_fold_metrics: List[Dict[str, float]] = []

        # Handle class imbalance: compute scale_pos_weight
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        scale_pos_weight = n_negative / max(n_positive, 1)
        logger.info(
            f"Class distribution: {n_negative} negative, {n_positive} positive. "
            f"scale_pos_weight = {scale_pos_weight:.2f}"
        )

        # Instantiate model
        if HAS_XGB:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                class_weight="balanced",
                random_state=self.random_state,
            )

        # Cross-validation loop
        logger.info("Starting TimeSeries Cross-Validation (5 folds)...")
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Skip fold if only one class is present
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning(f"Fold {fold} skipped: single class in target.")
                continue

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_metrics = {
                "fold": fold,
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "precision": float(
                    precision_score(y_test, y_pred, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, y_pred, zero_division=0)
                ),
                "brier_score": float(brier_score_loss(y_test, y_prob)),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
            per_fold_metrics.append(fold_metrics)
            logger.info(
                f"  Fold {fold}: ROC-AUC={fold_metrics['roc_auc']:.4f}, "
                f"Prec={fold_metrics['precision']:.4f}, "
                f"Rec={fold_metrics['recall']:.4f}, "
                f"Brier={fold_metrics['brier_score']:.4f}"
            )

        # Final training on ALL data
        logger.info("Training final model on all data...")
        model.fit(X, y)

        # Calculate mean metrics across folds
        if per_fold_metrics:
            mean_metrics = {
                "roc_auc": float(
                    np.mean([m["roc_auc"] for m in per_fold_metrics])
                ),
                "precision": float(
                    np.mean([m["precision"] for m in per_fold_metrics])
                ),
                "recall": float(
                    np.mean([m["recall"] for m in per_fold_metrics])
                ),
                "brier_score": float(
                    np.mean([m["brier_score"] for m in per_fold_metrics])
                ),
            }
        else:
            mean_metrics = {
                "roc_auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "brier_score": 0.0,
            }

        logger.info(f"Mean Cross-Validation Metrics: {mean_metrics}")

        return {
            "mean_metrics": mean_metrics,
            "per_fold_metrics": per_fold_metrics,
        }, model

    # ==================================================================
    # INTERNAL: Feature importance extraction
    # ==================================================================

    @staticmethod
    def _get_feature_importances(
        model: Any,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Extract and sort feature importances from the trained model.

        Parameters
        ----------
        model : Any
            Trained classifier with ``feature_importances_`` attribute.
        feature_names : List[str]
            Ordered list of feature names.

        Returns
        -------
        List[Dict[str, Any]]
            Sorted list of ``{feature, importance}`` dicts (descending).
        """
        try:
            importances = model.feature_importances_
            result = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importances)
            ]
            result.sort(key=lambda x: x["importance"], reverse=True)
            return result
        except AttributeError:
            logger.warning("Model does not have feature_importances_ attribute.")
            return []

    # ==================================================================
    # INTERNAL: Model serialization
    # ==================================================================

    @staticmethod
    def _save_model(
        model: Any,
        features: List[str],
        metrics: Dict[str, Any],
        model_output_path: str,
        tickers_used: List[str],
        training_date_range: tuple,
        samples_per_class: Dict[str, int],
        ticker_sample_counts: Optional[Dict[str, int]] = None,
        failed_tickers: Optional[List[str]] = None,
        feature_importances: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Serialize the trained model and all training metadata to disk.

        Parameters
        ----------
        model : Any
            Trained classifier.
        features : List[str]
            Feature column names.
        metrics : Dict[str, Any]
            Training metrics (mean and per-fold).
        model_output_path : str
            Output file path.
        tickers_used : List[str]
            Tickers included in training.
        training_date_range : tuple
            (start_date, end_date) strings.
        samples_per_class : Dict[str, int]
            Count of samples per target class.
        ticker_sample_counts : Optional[Dict[str, int]]
            Per-ticker sample counts.
        failed_tickers : Optional[List[str]]
            Tickers that failed during data download.
        feature_importances : Optional[List[Dict[str, Any]]]
            Sorted feature importances.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(model_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        saved_data = {
            "model": model,
            "features": features,
            "metrics": metrics.get("mean_metrics", metrics),
            "per_fold_metrics": metrics.get("per_fold_metrics", []),
            "model_type": type(model).__name__,
            "tickers_used": tickers_used,
            "training_date_range": training_date_range,
            "samples_per_class": samples_per_class,
            "ticker_sample_counts": ticker_sample_counts or {},
            "failed_tickers": failed_tickers or [],
            "feature_importances": feature_importances or [],
            "trained_at": datetime.now().isoformat(),
        }

        joblib.dump(saved_data, model_output_path)
        logger.info(f"Model and metadata saved to {model_output_path}")
