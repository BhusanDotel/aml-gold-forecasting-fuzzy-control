"""
Data preprocessing for XAUUSD M15 dataset.
Loads raw OHLC data, computes technical indicators used as GP and BN features.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'XAUUSD_M15.csv')


def load_data(n_rows=5000):
    """Load and clean XAUUSD M15 data."""
    df = pd.read_csv(
        DATA_PATH, sep='\t', skiprows=1, header=None,
        names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Ticks', 'Volume']
    )
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df.dropna()
    if n_rows:
        df = df.iloc[:n_rows]
    return df


def add_technical_indicators(df):
    """Compute technical indicators from OHLC data."""
    # Returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Moving averages
    df['SMA_14'] = df['Close'].rolling(14).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Ratio of close to SMA (normalised)
    df['SMA_Ratio'] = df['Close'] / df['SMA_14'] - 1.0

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # Average True Range (ATR)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # Bollinger Band width
    rolling_std = df['Close'].rolling(14).std()
    df['BB_Width'] = (2 * rolling_std) / df['SMA_14']

    # Price range features
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['OC_Range'] = (df['Close'] - df['Open']) / df['Open']

    # Volume normalised over rolling window
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].rolling(14).mean()) / (
        df['Volume'].rolling(14).std() + 1e-9
    )

    # Time features
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    # Trading session flag (0=Asian 1=European 2=American)
    def session(h):
        if 0 <= h < 8:
            return 0
        elif 8 <= h < 16:
            return 1
        else:
            return 2

    df['Session'] = df['Hour'].apply(session)

    # Next-bar target (for supervised learning)
    df['Target_Return'] = df['Log_Return'].shift(-1)
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)

    df = df.dropna()
    return df


# GP feature set (4 inputs + 1 output) --------------------------------
GP_FEATURES = ['RSI', 'SMA_Ratio', 'HL_Range', 'Volume_Norm', 'ATR']
GP_TARGET = 'Target_Return'

# Bayesian Network variable list (10 variables) ------------------------
BN_VARS = [
    'Open_cat', 'High_cat', 'Low_cat', 'Close_cat',
    'Volume_cat', 'RSI_cat', 'Trend_cat',
    'Session', 'Volatility_cat', 'Direction'
]


def discretize_for_bn(df, n_bins=3):
    """Discretize continuous variables into categories for Bayesian Network."""
    out = pd.DataFrame(index=df.index)

    # Price levels relative to rolling mean → Low / Mid / High
    for col in ['Open', 'High', 'Low', 'Close']:
        mu = df[col].rolling(100).mean().fillna(df[col].mean())
        sig = df[col].rolling(100).std().fillna(df[col].std())
        z = (df[col] - mu) / (sig + 1e-9)
        out[f'{col}_cat'] = pd.cut(z, bins=[-np.inf, -0.5, 0.5, np.inf],
                                   labels=[0, 1, 2]).astype(int)

    # Volume
    out['Volume_cat'] = pd.cut(df['Volume'], bins=n_bins,
                               labels=list(range(n_bins))).astype(int)

    # RSI zones: oversold / neutral / overbought
    out['RSI_cat'] = pd.cut(df['RSI'], bins=[0, 35, 65, 100],
                             labels=[0, 1, 2]).astype(int)

    # Trend: price above / below SMA
    out['Trend_cat'] = (df['Close'] > df['SMA_14']).astype(int)

    # Session (already integer 0/1/2)
    out['Session'] = df['Session'].astype(int)

    # Volatility based on ATR quartiles
    out['Volatility_cat'] = pd.cut(df['ATR'], bins=2,
                                   labels=[0, 1]).astype(int)

    # Target direction
    out['Direction'] = df['Target_Direction'].astype(int)

    out = out.dropna()
    return out


if __name__ == '__main__':
    df_raw = load_data()
    df = add_technical_indicators(df_raw)
    print(f"Dataset shape after indicators: {df.shape}")
    print(df[GP_FEATURES + [GP_TARGET]].describe())
    print("\nBN discrete vars sample:")
    print(discretize_for_bn(df).head())
