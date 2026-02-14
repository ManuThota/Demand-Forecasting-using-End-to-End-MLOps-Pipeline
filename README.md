# Demand Forecasting using End-to-End MLOps Pipeline

This project is a **production-grade end-to-end Machine Learning & MLOps pipeline** to predict **weekly product demand (units sold)** for retail stores using historical sales, pricing, and promotion data.

The system is built with a **real-world MLOps mindset**, covering everything from feature engineering and model training to **MLflow model registry, Dockerized FastAPI deployment, CI/CD automation, and cloud deployment**.

---

## 📌 Problem Statement

Accurately forecast **future demand (units sold)** for each **Store–SKU combination** to help businesses:

- Optimize inventory levels
- Reduce stock-outs and overstock
- Improve pricing and promotion strategies

The model learns from historical patterns such as:

- Store ID & SKU ID
- Price and discount information
- Promotion flags
- Time-based features
- Lagged sales behavior

---

## 💡 Solution Overview

- Performed **EDA & experimentation** in VS code
- Engineered **time-series lag features and rolling statistics**
- Trained and evaluated models using **XGBoost**
- Applied **log transformation** to stabilize demand variance
- Tracked experiments using **MLflow**
- Registered the best model in **MLflow Model Registry**
- Built a **FastAPI web application** with form-based input
- Containerized the application using **Docker**
- Automated **CI/CD using GitHub Actions**
- Deployed the service to **Render (cloud)**

---
## 📊 Dataset

One of the largest retail chains in the world wants to use their vast data source to build an efficient forecasting model to predict the sales for each SKU in its portfolio at its 76 different stores using historical sales data for the past 3 years on a week-on-week basis. Sales and promotional information is also available for each week - product and store wise.

Link : [https://www.kaggle.com/datasets/aswathrao/demand-forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting)

---

## 📂 Project Structure

```text
End-to-End-MLOps-Pipeline-for-Demand-Forecasting/
│
├──.github/
|   └──workflows/
|      └──ci-cd.yml
├──api/
│  └──main.py
│
├──data/
│  ├──raw/
│  └──processed/
|
├──mlartifacts/
│  └──(MLflow model artifacts)
|
├──model/
|  └──(Place the final model here)
|
├──models/
|  └──xgb_demand_model.pkl
|
├──src/
│  ├──data/
|  |  ├──load_data.py
|  |  ├──preprocess.py
|  |  └──run_preprocessing.py   
│  ├──features/
|  |  ├──build_features.py
|  |  └──run_feature_pipeline.py
│  ├──models/
|  |  ├──evaluate.py
|  |  ├──train_mlflow.py
|  |  └──train.py
│  ├──pipelines/
|  |  └──train_pipeline.py
│  └──__init__.py
|
├──static/
|  └──styles.css
|
├──template/
|  └──index.html
|
├──requirements.txt
├──Dockerfile
├──.dockerignore
├──.gitignore
├──.setup.py
└──README.md
```

---
## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- MLflow (Experiment Tracking & Model Registry)
- FastAPI
- Docker
- GitHub Actions (CI/CD)
- Render (Cloud Deployment)

---
## 🤖 Machine Learning & MLOps Pipeline
1. Data Ingestion
   - Raw sales data loaded from data/raw/

2. Feature Engineering
   - Time-based features (day, week, month)
   - Discount and discount percentage
   - Lag features:
        - units_sold_lag_1
        - units_sold_lag_2
        - units_sold_lag_4
        - units_sold_lag_8
   - Rolling mean features:
        - units_sold_roll_mean_2
        - units_sold_roll_mean_4
        - units_sold_roll_mean_8

3. Model Training
   - Algorithm: XGBoost Regressor
   - Log transformation applied to target
   - Time-based train-validation split
   - Metrics tracked:
        - RMSE
        - MAPE

4. Experiment Tracking
   - All experiments logged using MLflow
   - Parameters, metrics, and artifacts stored
   - Best model promoted to Production stage

5. Model Serving
   - Model loaded from MLflow artifacts
   - Served via FastAPI
   - HTML form-based input (no JSON required)

6. Containerization
   - Dockerized FastAPI application
   - Environment-variable-based configuration

7. CI/CD Automation
   - Auto test on GitHub push
   - Auto Docker image build
   - Auto push to Docker Hub
   - Auto redeploy on model/code updates
---
## 🚀 How to Run the Project

1. Clone the repository
   
   ```
    git clone <my-repo-url>
    cd End-to-End-MLOps-Pipeline-for-Demand-Forecasting
   ```
2. Create & activate virtual environment

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies

   ```
   pip install -r requirements.txt
   ```

4. Run pipeline
  
   ```
   python -m src.pipelines.train_pipeline
   ```

5. Check the MLflow and make the model stage to production

   ```
   mlflow ui
   ```

6. Run FastAPI app

   ```
   uvicorn api.main:app --reload
   ```

6. Open in browser

   ```
   http://127.0.0.1:8000
   ```

---
## 🌟 Model Performance
     
     - Model Used: XGBoost Regressor
     - Metrics:
        - RMSE: ~25
        - MAPE: ~37%
     - Performance improved using:
        - Log transformation
        - Lag and rolling window features
---

## 📌 Future Improvements
    
    - Add model monitoring & drift detection
    - Scheduled retraining pipeline
    - Remote MLflow tracking server
    - Authentication & rate limiting for API
    - Advanced UI dashboards
---
## 👹Author

    Aspiring Machine Learning Engineer / Data Scientist
                                            - Mad_titaN 

⭐ If you like this project, consider giving it a star!