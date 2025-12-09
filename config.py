import os

# Dynamic Path Determination
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# Settings
TICKER = "RELIANCE.NS"
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
INITIAL_CASH = 100000.0
COMMISSION = 0.001



