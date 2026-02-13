import pandas as pd
from src.features.build_features import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_price_features
)

PROCESSED_DATA_DIR = "data/processed"


def main():
    train_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/train_processed.csv", parse_dates=["week"])
    val_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/val_processed.csv", parse_dates=["week"])
    test_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/test_processed.csv", parse_dates=["week"])

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = create_time_features(df)
        df = create_price_features(df)

        if "units_sold" in df.columns:
            df = create_lag_features(df)
            df = create_rolling_features(df)

        df.dropna(inplace=True)

        df.to_csv(f"{PROCESSED_DATA_DIR}/{df_name}_features.csv", index=False)
        print(f"{df_name} features saved.")


if __name__ == "__main__":
    main()
