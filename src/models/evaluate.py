import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


FEATURE_DATA_DIR = "data/processed"
MODEL_DIR = "models"
TARGET = "units_sold"


def main():
    val_df = pd.read_csv(f"{FEATURE_DATA_DIR}/val_features.csv")

    # Separate target
    y_val = val_df[TARGET].values

    # Drop non-feature columns
    drop_cols = ["record_ID", "week", TARGET]
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])

    # FORCE numeric numpy array
    X_val = X_val.to_numpy(dtype=np.float32)

    model = joblib.load(f"{MODEL_DIR}/xgb_demand_model.pkl")

    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mape = mean_absolute_percentage_error(y_val, preds)

    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAPE: {mape:.2%}")


if __name__ == "__main__":
    main()
