import pandas as pd
from typing import List, Optional
import logging
from xbbg import blp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_single_ticker_data(
    ticker: str, 
    fields: List[str], 
    start_date: str, 
    end_date: str, 
    freq: str = 'DAILY', 
    fill_method: str = 'ffill', 
    clean: bool = True, 
    column_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieves and cleans data for a single ticker from Bloomberg using XBBG library.

    :param ticker: The ticker to retrieve data for.
    :param fields: List of fields to retrieve for the ticker.
    :param start_date: Start date for data retrieval (YYYY-MM-DD).
    :param end_date: End date for data retrieval (YYYY-MM-DD).
    :param freq: Frequency of data retrieval (e.g., 'DAILY', 'MONTHLY').
    :param fill_method: Method to fill missing data ('ffill' for forward fill, 'bfill' for backward fill).
    :param clean: Boolean indicating whether to clean data.
    :param column_names: List of new column names to override the default ones.
    :return: Cleaned DataFrame with data for specified ticker and fields, with single-level column names.
    """
    
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and fill missing data in the DataFrame."""
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        else:
            raise ValueError("Invalid fill_method. Use 'ffill' or 'bfill'.")
        # Additional cleaning steps can be added here
        return df
    
    def normalize_frequency(frequency: str) -> str:
        """Normalize frequency input to the expected Bloomberg format."""
        freq_map = {
            'DAILY': 'DAILY', 'D': 'DAILY',
            'WEEKLY': 'WEEKLY', 'W': 'WEEKLY',
            'MONTHLY': 'MONTHLY', 'M': 'MONTHLY',
            'QUARTERLY': 'QUARTERLY', 'Q': 'QUARTERLY',
            'YEARLY': 'YEARLY', 'Y': 'YEARLY',
            'BM': 'MONTHLY', 'BME': 'MONTHLY', 'BMONTHLY': 'MONTHLY'  # Treat business month-end as monthly
        }
        return freq_map.get(frequency.upper(), 'DAILY')  # Default to 'DAILY' if not found
    
    try:
        # Normalize frequency
        normalized_freq = normalize_frequency(freq)
        
        logging.info(f"Retrieving data for ticker: {ticker} with frequency: {normalized_freq}")
        
        # Retrieve data for the single ticker
        data = blp.bdh(tickers=ticker, flds=fields, start_date=start_date, end_date=end_date, Per=normalized_freq)
        
        logging.info(f"Retrieved data shape for {ticker}: {data.shape}")

        if clean:
            data = clean_data(data)
            logging.info(f"Cleaned data shape for {ticker}: {data.shape}")

        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Force frequency to be monthly if specified and handle business month-end frequencies
        if normalized_freq == 'MONTHLY':
            data = data.asfreq('M', method='ffill')

        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]

        # Override column names if provided
        if column_names:
            if len(column_names) != len(fields):
                raise ValueError("Length of column_names must match length of fields.")
            data.columns = column_names
        else:
            data.columns = [f"{ticker}_{field}" for field in fields]

        # Perform data integrity checks
        assert not data.isnull().values.any(), f"Data for {ticker} contains missing values"
        assert data.index.is_monotonic_increasing, f"Date index for {ticker} is not sorted"
        
        logging.info(f"Successfully retrieved data for ticker: {ticker}")
        
        return data

    except Exception as e:
        logging.error(f"Error retrieving data for ticker {ticker}: {e}")
        raise

def merge_dataframes(dataframes: List[pd.DataFrame], method: str = 'outer') -> pd.DataFrame:
    """
    Merges a list of DataFrames on their datetime index.

    :param dataframes: List of DataFrames to merge.
    :param method: Method of merging ('inner', 'outer', 'left', 'right').
    :return: Merged DataFrame with aligned date index.
    """
    if not dataframes:
        raise ValueError("No dataframes to merge.")
    
    # Merge all DataFrames on their index
    merged_df = pd.concat(dataframes, axis=1, join=method)
    
    # Fill missing values after merging
    merged_df = merged_df.ffill().bfill()

    # Perform data integrity checks
    assert not merged_df.isnull().values.any(), "Merged data contains missing values"
    assert merged_df.index.is_monotonic_increasing, "Date index is not sorted"
    
    logging.info(f"Merged {len(dataframes)} dataframes using {method} method.")
    
    return merged_df


def merge_dataframes_latest_start_date(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
  """
  Merges a list of DataFrames on their datetime index, ensuring the earliest start date
  of the returned DataFrame is the latest start date among the input DataFrames.

  :param dataframes: List of DataFrames to merge.
  :return: Merged DataFrame with aligned date index starting from the latest start date.
  """
  if not dataframes:
      raise ValueError("No dataframes to merge.")
  
  # Perform an inner join to ensure the earliest start date is the latest among the DataFrames
  merged_df = pd.concat(dataframes, axis=1, join='inner')
  
  # Remove the fill methods to keep only complete data
  # merged_df = merged_df.ffill().bfill()

  # Perform data integrity checks
  assert not merged_df.isnull().values.any(), "Merged data contains missing values"
  assert merged_df.index.is_monotonic_increasing, "Date index is not sorted"
  
  logging.info(f"Merged {len(dataframes)} dataframes using inner join to align on latest start date.")
  
  return merged_df