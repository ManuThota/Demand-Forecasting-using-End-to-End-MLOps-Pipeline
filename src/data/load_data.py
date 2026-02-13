import pandas as pd
from pathlib import Path

def load_raw_data(data_dir: str):
    """
    Loading the RAW TRAIN and TEST datasets
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    return train_df, test_df