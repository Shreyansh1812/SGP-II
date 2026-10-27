import pytest
import pandas as pd
import numpy as np
import os
import joblib

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.feature_engine import compute_technical_features
from src.ml.trainer import MLModelTrainer

@pytest.fixture
def sample_ohlcv():
    """Generates synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')

    # Create somewhat realistic price data with a trend
    close_prices = 100 + np.cumsum(np.random.normal(0, 1, 300))
    open_prices = close_prices + np.random.normal(0, 0.5, 300)
    high_prices = np.maximum(close_prices, open_prices) + np.random.normal(0, 0.5, 300)
    low_prices = np.minimum(close_prices, open_prices) - np.random.normal(0, 0.5, 300)
    volume = np.random.randint(1000, 10000, 300)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)

    return df

class TestFeatureEngine:

    def test_feature_engine_output_shape(self, sample_ohlcv):
        """Test output shape and ensure NaNs are dropped."""
        df_features = compute_technical_features(sample_ohlcv)

        # Original rows = 300. Max window is SMA200 (200). Plus 5-day forward return.
        # So we expect around 300 - 200 - 5 = 95 rows.
        assert len(df_features) > 50, "Not enough rows after feature generation"
        assert not df_features.isna().any().any(), "NaNs found in output DataFrame"

    def test_feature_engine_columns(self, sample_ohlcv):
        """Test if all expected feature columns exist."""
        df_features = compute_technical_features(sample_ohlcv)

        expected_cols = [
            'RSI_14', 'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'BB_PctB', 'BB_Bandwidth', 'ATR_Norm',
            'Ratio_Close_SMA20', 'Ratio_Close_SMA50', 'Ratio_SMA50_SMA200',
            'Volatility_20d', 'Target_Buy'
        ]

        for col in expected_cols:
            assert col in df_features.columns, f"Missing expected column: {col}"

    def test_zero_lookahead_bias(self, sample_ohlcv):
        """
        Crucial Test: Ensure that features at index T are strictly derived from prices up to T-1.
        Since our code computes features on price at T, then shifts them by 1,
        the feature value at T should correspond to the unshifted indicator value at T-1.
        """
        # We will manually calculate a simple feature and compare
        # e.g. Ratio_Close_SMA20
        df = sample_ohlcv.copy()

        # Get features
        df_features = compute_technical_features(df)

        # Manual calc of unshifted SMA20 and Ratio for comparison
        sma_20_unshifted = df['Close'].rolling(window=20, min_periods=20).mean()
        ratio_unshifted = df['Close'] / sma_20_unshifted

        # The value of Ratio_Close_SMA20 in df_features at index `t`
        # MUST EQUAL the unshifted ratio at index `t-1`

        # Let's pick a random valid index from df_features
        test_idx = df_features.index[50]
        # Find integer location of this index in the original dataframe
        loc = df.index.get_loc(test_idx)

        # The previous day's timestamp
        prev_idx = df.index[loc - 1]

        # Assert equality (with small tolerance for float issues)
        assert np.isclose(
            df_features.loc[test_idx, 'Ratio_Close_SMA20'],
            ratio_unshifted.loc[prev_idx]
        ), "Lookahead bias detected! Feature at T does not match indicator value at T-1."

    def test_target_generation(self, sample_ohlcv):
        """Test if target is generated correctly without lookahead bias."""
        df = sample_ohlcv.copy()
        df_features = compute_technical_features(df)

        test_idx = df_features.index[50]
        loc = df.index.get_loc(test_idx)

        # Target for index `t` is return from `t` to `t+5`
        close_t = df.loc[test_idx, 'Close']
        close_t_plus_5 = df.iloc[loc + 5]['Close']

        forward_return = (close_t_plus_5 / close_t) - 1.0
        expected_target = 1 if forward_return > 0.02 else 0

        assert df_features.loc[test_idx, 'Target_Buy'] == expected_target, "Target logic incorrect."


class TestModelTrainer:

    def test_trainer_pipeline(self, sample_ohlcv, tmp_path):
        """Test model training pipeline."""
        df_features = compute_technical_features(sample_ohlcv)

        trainer = MLModelTrainer(random_state=42)
        model_path = os.path.join(tmp_path, "test_model.pkl")

        result = trainer.train_xgboost_model(df_features, model_path)

        # Check returned result
        assert 'metrics' in result
        assert 'features' in result
        assert 'model_path' in result

        # Ensure metrics are not NaN
        assert not np.isnan(result['metrics']['roc_auc'])
        assert not np.isnan(result['metrics']['precision'])

        # Ensure file was saved
        assert os.path.exists(model_path)

        # Load and verify saved file
        saved_data = joblib.load(model_path)
        assert 'model' in saved_data
        assert 'features' in saved_data
        assert 'metrics' in saved_data

        assert saved_data['features'] == result['features']
