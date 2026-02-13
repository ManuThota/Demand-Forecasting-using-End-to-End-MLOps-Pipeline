import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -------------------------
# Model Path
# -------------------------
MODEL_PATH = "model/model.pkl"
model = joblib.load(MODEL_PATH)


# -------------------------
# Historical Data
# -------------------------
HISTORICAL_DATA_PATH = "data/processed/train_features.csv"
historical_df = pd.read_csv(HISTORICAL_DATA_PATH)

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------
# Home Route
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#--------------------------
# Health Check Route
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction Route
# -------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    store_id: int = Form(...),
    sku_id: int = Form(...),
    total_price: float = Form(...),
    base_price: float = Form(...),
    is_featured_sku: int = Form(...),
    is_display_sku: int = Form(...),
):

    sku_history = historical_df[
        (historical_df["store_id"] == store_id) &
        (historical_df["sku_id"] == sku_id)
    ]

    if sku_history.empty:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": "No historical data found."}
        )

    latest_row = sku_history.sort_values("week").iloc[-1]

    input_dict = {
        "store_id": store_id,
        "sku_id": sku_id,
        "total_price": total_price,
        "base_price": base_price,
        "is_featured_sku": is_featured_sku,
        "is_display_sku": is_display_sku,
        "day": latest_row["day"],
        "week_of_year": latest_row["week_of_year"],
        "month": latest_row["month"],
        "discount": base_price - total_price,
        "discount_pct": (base_price - total_price) / base_price,
        "units_sold_lag_1": latest_row["units_sold_lag_1"],
        "units_sold_lag_2": latest_row["units_sold_lag_2"],
        "units_sold_lag_3": latest_row["units_sold_lag_3"],
        "units_sold_lag_4": latest_row["units_sold_lag_4"],
        "units_sold_lag_8": latest_row["units_sold_lag_8"],
        "units_sold_roll_mean_2": latest_row["units_sold_roll_mean_2"],
        "units_sold_roll_mean_4": latest_row["units_sold_roll_mean_4"],
        "units_sold_roll_mean_8": latest_row["units_sold_roll_mean_8"],
    }

    input_df = pd.DataFrame([input_dict])

    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log[0])

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": round(prediction, 2)}
    )
