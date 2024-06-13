import pandas as pd
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
import config

# Project description & conceptual design
"""
This module is designed to fetch historical cryptocurrency data using Alpaca API,
process the data, and prepare it for further analysis. The fetched data includes
timestamp, open, high, low, close prices, and volume for specified cryptocurrencies.
"""

# Data model design
# The data model involves using a DataFrame to store the fetched cryptocurrency data.
# The DataFrame will have columns for timestamp, open, high, low, close prices, and volume.

# Function to create a connection to the Alpaca API and fetch data
def get_crypto_data(symbol, timeframe, start_date, end_date):
    """
    Fetch historical cryptocurrency data from Alpaca API.

    Parameters:
    - symbol: str, cryptocurrency symbol (e.g., "BTC/USD")
    - timeframe: TimeFrame, the timeframe for the data (e.g., TimeFrame.Hour)
    - start_date: str, the start date for fetching data in "YYYY-MM-DD" format
    - end_date: str, the end date for fetching data in "YYYY-MM-DD" format

    Returns:
    - crypto_bars: DataFrame, the fetched cryptocurrency data
    """
    # Create a client to fetch historical data
    client = CryptoHistoricalDataClient(config.API_KEY, config.SECRET_KEY)

    # Create request parameters
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        # timeframe=timeframe.Minute,
        timeframe=timeframe.Hour,
        # timeframe=timeframe.Day,
        start=start_date,
        end=end_date
    )

    # Fetch data
    crypto_bars = client.get_crypto_bars(request_params).df

    # Reset index to make 'timestamp' a separate column
    crypto_bars = crypto_bars.reset_index()

    # Clean and process data
    crypto_bars = clean_data(crypto_bars)

    print(crypto_bars)
    return crypto_bars


# ETL / Data processing e.g. cleaning
def clean_data(df):
    """
    Clean and process the fetched cryptocurrency data.

    Parameters:
    - df: DataFrame, the raw cryptocurrency data

    Returns:
    - df: DataFrame, the cleaned and processed cryptocurrency data
    """
    # Drop any rows with missing values
    df = df.dropna()

    # Convert the timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure that numerical columns are of correct dtype
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    return df