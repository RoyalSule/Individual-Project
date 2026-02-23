import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Loads historical data for ML models."""

    def __init__(self):
        self.ticker = "^GSPC"
        self.period = "10y"
        self.data = None

    def download(self):
        """Download data from Yahoo Finance."""
        logger.info(f"Get historical data for the S&P 500 ({self.ticker})...")
        ticker = yf.Ticker(self.ticker)
        self.data = ticker.history(period=self.period)
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        filename = f"sp500_{datetime.today().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(raw_dir, filename)
        self.data.to_csv(file_path)
        return self.data

    def clean(self):
        """Clean and prepare data."""
        if self.data is None:
            logger.error("There is no data to clean. Run download() first.")
            return None
        logger.info("Clean historical data...")
        del self.data["Dividends"]
        del self.data["Stock Splits"]
        self.data["Tomorrow"] = self.data["Close"].shift(-1)
        self.data["Target"] = (self.data["Tomorrow"] > self.data["Close"]).astype(int)
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        filename = f"sp500_{datetime.today().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(processed_dir, filename)
        self.data.to_csv(file_path)
        return self.data

if __name__ == "__main__":
    loader = DataLoader()

    data = loader.download()
    print(f"Download complete.")

    cleaned = loader.clean()
    print(f"Clean complete.")