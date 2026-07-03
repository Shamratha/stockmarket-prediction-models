"""Price data loading (yfinance with local cache, bundled CSV fallback) and
feature engineering."""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, 'data_cache')
DATASET_DIR = os.path.join(ROOT, 'dataset')

# bundled offline fallbacks (from the original repo's dataset, Apache-2.0)
FALLBACK_CSV = {
    'GOOG': 'GOOG-year.csv',
    'TSLA': 'TSLA.csv',
    'AMD': 'AMD.csv',
    'FB': 'FB.csv',
    'TWTR': 'TWTR.csv',
}


def load_prices(ticker='GOOG', period='5y', interval='1d', use_cache=True):
    """Return an OHLCV DataFrame indexed by date.

    Tries yfinance first (cached to data_cache/), falls back to the bundled
    CSVs so everything still runs offline.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{ticker}_{period}_{interval}.csv')
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _clean(df)

    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 50:
            raise RuntimeError(f'yfinance returned only {len(df)} rows')
        df.to_csv(cache_path)
        return _clean(df)
    except Exception as e:
        if ticker in FALLBACK_CSV:
            print(f'[data] yfinance failed ({e}); using bundled CSV for {ticker}')
            df = pd.read_csv(os.path.join(DATASET_DIR, FALLBACK_CSV[ticker]),
                             index_col=0, parse_dates=True)
            return _clean(df)
        raise


def _clean(df):
    df = df.rename(columns=str.title)
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[cols].dropna()
    df.index.name = 'Date'
    return df


def log_returns(close):
    return np.log(close / close.shift(1)).dropna()


def add_features(df):
    """Technical-indicator feature matrix for tree/boosting models."""
    out = pd.DataFrame(index=df.index)
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    out['ret_1'] = np.log(close / close.shift(1))
    for lag in (2, 3, 5, 10, 21):
        out[f'ret_{lag}'] = np.log(close / close.shift(lag))
    for win in (5, 10, 21):
        out[f'vol_{win}'] = out['ret_1'].rolling(win).std()
        out[f'sma_ratio_{win}'] = close / close.rolling(win).mean() - 1

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out['rsi_14'] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out['macd'] = macd / close
    out['macd_signal'] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    # Bollinger position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out['bb_pos'] = (close - sma20) / (2 * std20 + 1e-12)

    # intraday range + volume trend
    out['hl_range'] = (high - low) / close
    out['volume_z'] = (volume - volume.rolling(21).mean()) / (volume.rolling(21).std() + 1e-12)

    out['target_ret'] = out['ret_1'].shift(-1)  # next-day log return
    return out.dropna()


def train_test_split_series(series_or_df, test_size=100):
    """Chronological split: everything except the last `test_size` rows is train."""
    train = series_or_df.iloc[:-test_size]
    test = series_or_df.iloc[-test_size:]
    return train, test
