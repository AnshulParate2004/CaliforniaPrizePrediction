from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Step 0: Load saved models

lightgbm_model = joblib.load(r"D:\MLLabProject\tuned_lightgbm_regressor_model.pkl")
xgboost_model = joblib.load(r"D:\MLLabProject\tuned_xgboost_regressor_model.pkl")

# Step 1: Initialize FastAPI

app = FastAPI(title="California Housing Price Prediction")

origins = [
    "http://localhost",
    "http://localhost:8080",  # if you have a frontend running on port 3000
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allow these origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow GET, POST, etc.
    allow_headers=["*"],          # Allow custom headers
)


# Step 2: Define input schema with defaults, validation, and example

class HouseFeatures(BaseModel):
    longitude: float = Field(-122.23, description="Longitude of the house")
    latitude: float = Field(37.88, description="Latitude of the house")
    housing_median_age: float = Field(41, ge=0, description="Median age of houses in the block")
    total_rooms: float = Field(880, ge=1, description="Total rooms in the block")
    total_bedrooms: float = Field(129, ge=1, description="Total bedrooms in the block")
    population: float = Field(322, ge=1, description="Population of the block")
    households: float = Field(126, ge=1, description="Number of households")
    median_income: float = Field(8.3252, ge=0.01, description="Median income in tens of thousands")
    ocean_proximity_1H_OCEAN: float = Field(0, ge=0, le=1)
    ocean_proximity_INLAND: float = Field(1, ge=0, le=1)
    ocean_proximity_ISLAND: float = Field(0, ge=0, le=1)
    ocean_proximity_NEAR_BAY: float = Field(0, ge=0, le=1)
    ocean_proximity_NEAR_OCEAN: float = Field(0, ge=0, le=1)
    bedroom_ratio: float = Field(0.1466, ge=0, le=1)
    household_rooms: float = Field(6.9841, ge=0, description="Average rooms per household")

# Swagger UI input
    class Config:
        schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252,
                "ocean_proximity_1H_OCEAN": 0,
                "ocean_proximity_INLAND": 1,
                "ocean_proximity_ISLAND": 0,
                "ocean_proximity_NEAR_BAY": 0,
                "ocean_proximity_NEAR_OCEAN": 0,
                "bedroom_ratio": 0.1466,
                "household_rooms": 6.9841
            }
        }

# Step 3: Prediction helper

def predict(model, features: HouseFeatures):
    # Convert input to numpy array
    input_data = np.array([list(features.dict().values())])
    # Make prediction
    prediction = model.predict(input_data)[0]
    return round(float(prediction), 2)


# Step 4: LightGBM endpoint

@app.post("/predict/lightgbm")
def predict_lightgbm(features: HouseFeatures):
    if features.median_income <= 0:
        raise HTTPException(status_code=400, detail="Median income must be greater than 0.")
    return {"predicted_median_house_value": predict(lightgbm_model, features)}


# Step 5: XGBoost endpoint

@app.post("/predict/xgboost")
def predict_xgboost(features: HouseFeatures):
    if features.median_income <= 0:
        raise HTTPException(status_code=400, detail="Median income must be greater than 0.")
    return {"predicted_median_house_value": predict(xgboost_model, features)}

# Ensemble endpoint: mean of LightGBM + XGBoost

@app.post("/predict/ensemble")
def predict_ensemble(features: HouseFeatures):
    if features.median_income <= 0:
        raise HTTPException(status_code=400, detail="Median income must be greater than 0.")

    pred_lgbm = predict(lightgbm_model, features)
    pred_xgb = predict(xgboost_model, features)
    mean_prediction = round((pred_lgbm + pred_xgb) / 2, 2)

    return {
        "lightgbm_prediction": round(pred_lgbm, 2),
        "xgboost_prediction": round(pred_xgb, 2),
        "mean_prediction": mean_prediction
    }