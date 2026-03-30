import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json
from predict import predict

app = FastAPI(
    title="House Price Predictor API",
    description="Prédit le prix d'une maison basé sur ses caractéristiques.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseFeatures(BaseModel):
    OverallQual: int = Field(5, ge=1, le=10, description="Qualité générale (1-10)")
    GrLivArea: int = Field(1500, description="Surface habitable (sqft)")
    TotalBsmtSF: int = Field(800, description="Surface sous-sol (sqft)")
    GarageCars: int = Field(2, description="Capacité garage (voitures)")
    YearBuilt: int = Field(1990, description="Année de construction")
    YearRemodAdd: int = Field(1990, description="Année de rénovation")
    Neighborhood: str = Field("NAmes", description="Quartier")
    FullBath: int = Field(2, description="Nb salles de bain complètes")
    BedroomAbvGr: int = Field(3, description="Nb chambres")
    KitchenQual: str = Field("TA", description="Qualité cuisine (Ex/Gd/TA/Fa/Po)")
    MSZoning: str = Field("RL", description="Zonage")
    LotArea: int = Field(8500, description="Surface terrain (sqft)")

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1800,
                "TotalBsmtSF": 900,
                "GarageCars": 2,
                "YearBuilt": 2000,
                "YearRemodAdd": 2005,
                "Neighborhood": "NAmes",
                "FullBath": 2,
                "BedroomAbvGr": 3,
                "KitchenQual": "Gd",
                "MSZoning": "RL",
                "LotArea": 9000
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    price_range: list
    currency: str


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "House Price Predictor API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_price(features: HouseFeatures):
    try:
        result = predict(features.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Model"])
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Modèle pas encore entraîné")
    with open(metrics_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
