import pandas as pd
import joblib
from xgboost import XGBRegressor


FEATURE_DATA_DIR = "data/processed"
MODEL_DIR = "models"
TARGET = "units_sold"

def prepare_features(df):
    df = df.copy()

    # Drop identifiers & datetime
    drop_cols = ["record_ID", "week"]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Convert everything else to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.fillna(0)

    return df

def load_data():
    train_df = pd.read_csv(f"{FEATURE_DATA_DIR}/train_features.csv")
    val_df = pd.read_csv(f"{FEATURE_DATA_DIR}/val_features.csv")
    return train_df, val_df


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main():
    train_df, val_df = load_data()

    X_train = prepare_features(train_df.drop(columns=[TARGET]))
    y_train = train_df[TARGET]

    model = train_model(X_train, y_train)

    joblib.dump(model, f"{MODEL_DIR}/xgb_demand_model.pkl")
    print("Model trained and saved.")



if __name__ == "__main__":
    main()
