import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure week is datetime
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

    df["day"] = df["week"].dt.day
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month

    return df



def create_lag_features(df: pd.DataFrame, lags=[1, 2, 3, 4, 8]) -> pd.DataFrame:
    df = df.copy()

    for lag in lags:
        df[f"units_sold_lag_{lag}"] = (
            df.groupby(["store_id", "sku_id"])["units_sold"]
            .shift(lag)
        )

    return df


def create_rolling_features(df: pd.DataFrame, windows=[2, 4, 8]) -> pd.DataFrame:
    df = df.copy()

    for window in windows:
        df[f"units_sold_roll_mean_{window}"] = (
            df.groupby(["store_id", "sku_id"])["units_sold"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["discount"] = df["base_price"] - df["total_price"]
    df["discount_pct"] = df["discount"] / df["base_price"]

    return df
