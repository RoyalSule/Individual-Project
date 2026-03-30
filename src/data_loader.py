import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads historical data for ML models."""

    def __init__(self):
        self.ticker = "^GSPC"
        self.period = "max"
        self.data = None

    def download(self):
        """Download data from Yahoo Finance."""
        logger.info(f"Get historical data for the S&P 500 ({self.ticker})...")
        ticker = yf.Ticker(self.ticker)
        self.data = ticker.history(period=self.period)
        return self.data

    def clean(self):
        """Clean and prepare data."""
        logger.info(f"Clean historical data...")
        del self.data["Dividends"]
        del self.data["Stock Splits"]
        self.data["Tomorrow"] = self.data["Close"].shift(-1)
        self.data["Target"] = (self.data["Tomorrow"] > self.data["Close"]).astype(int)
        return self.data

    def add_features(self):
        """Add scale-invariant features."""
        logger.info(f"Adding features...")
        horizons = [2, 5, 60, 250, 1000]
        for horizon in horizons:
            # Close ratio
            self.data[f"Close_Ratio_{horizon}"] = self.data["Close"] / self.data["Close"].rolling(horizon).mean()
            # Volume ratio
            self.data[f"Volume_Ratio_{horizon}"] = self.data["Volume"] / self.data["Volume"].rolling(horizon).mean()
            # Trend
            self.data[f"Trend_{horizon}"] = self.data["Target"].shift(1).rolling(horizon).sum()
        # QStick Indicators
        # Body = |Close - Open|
        self.data["Body_Pct"] = (self.data["Close"] - self.data["Open"]) / self.data["Open"] * 100
        # Range = High - Low
        self.data["Range_Pct"] = (self.data["High"] - self.data["Low"]) / self.data["High"] * 100
        return self.data

    def add_technical_indicators(self):
        """Add EMA, MACD and ATR indicators."""
        logger.info(f"Adding technical indicators...")
        # 1. EMA (Exponential Moving Average)
        self.data["EMA_9"] = self.data["Close"].ewm(span=9, adjust=False).mean()
        # 2. MACD (Moving Average Convergence/Divergence)
        ema_12 = self.data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = self.data["Close"].ewm(span=26, adjust=False).mean()
        self.data["EMA_12"] = ema_12
        self.data["EMA_26"] = ema_26
        # MACD Line = EMA(12) - EMA(26)
        self.data["MACD"] = ema_12 - ema_26
        # Signal Line = EMA(9) of the MACD Line
        self.data["MACD_Signal"] = self.data["MACD"].ewm(span=9, adjust=False).mean()
        # MACD Histogram = MACD Line - MACD Signal
        self.data["MACD_Histogram"] = self.data["MACD"] - self.data["MACD_Signal"]
        # 3. ATR (Average True Range)
        high = self.data["High"]
        low = self.data["Low"]
        closep = self.data["Close"].shift(1)
        # TR = MAX[(H - L), |H - Cp|, |L - Cp|]
        tr = pd.concat([high - low, abs(high - closep), abs(low - closep)], axis=1).max(axis=1)
        self.data["ATR_14"] = tr.rolling(window=14).mean()
        return self.data
    