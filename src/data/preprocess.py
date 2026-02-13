import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing:
    - Parse week column as datetime
    - Sort data for time series
    """
    df = df.copy()

    # Convert week to datetime
    df["week"] = pd.to_datetime(df["week"], format="%d-%m-%Y")

    # Sort by store, sku, and time
    df = df.sort_values(by=["store_id", "sku_id", "week"])

    return df

def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """
    Time-based train-validation split using percentage.
    """
    df = df.sort_values("week")

    split_idx = int(len(df) * (1 - val_ratio))

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    return train_df, val_df

