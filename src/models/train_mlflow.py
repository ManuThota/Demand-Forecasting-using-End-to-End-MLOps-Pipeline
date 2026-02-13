import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


FEATURE_DATA_DIR = "data/processed"
TARGET = "units_sold"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Demand_Forecasting")

def main():
    train_df = pd.read_csv(f"{FEATURE_DATA_DIR}/train_features.csv")
    val_df = pd.read_csv(f"{FEATURE_DATA_DIR}/val_features.csv")

    DROP_COLS = [TARGET, "week", "record_ID"]

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = np.log1p(train_df[TARGET])

    X_val = val_df.drop(columns=DROP_COLS)
    y_val = np.log1p(val_df[TARGET])


    with mlflow.start_run():
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds_log = model.predict(X_val)

        # Convert back to original scale
        preds = np.expm1(preds_log)
        y_true = np.expm1(y_val)

        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = mean_absolute_percentage_error(y_true, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 6)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="DemandForecastingModel"
        )


        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2%}")

        print("Feature order used for training:")
        print(list(X_train.columns))



if __name__ == "__main__":
    main()
