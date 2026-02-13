from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data, train_val_split


RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def main():
    train_df, test_df = load_raw_data(RAW_DATA_DIR)

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_df, val_df = train_val_split(train_df)

    train_df.to_csv(f"{PROCESSED_DATA_DIR}/train_processed.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DATA_DIR}/val_processed.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DATA_DIR}/test_processed.csv", index=False)

    print("Phase 1 complete: Processed data saved.")


if __name__ == "__main__":
    main()
