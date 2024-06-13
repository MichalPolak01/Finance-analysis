import pandas as pd
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.historical import CryptoHistoricalDataClient
import config

# Function to create a connection to the Alpaca API and fetch data
def get_crypto_data(symbol, timeframe, start_date, end_date):
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


def clean_data(df):
    # Drop any rows with missing values
    df = df.dropna()

    # Convert the timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure that numerical columns are of correct dtype
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    return df