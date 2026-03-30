import numpy as np
import joblib
import os
import pandas as pd
from preprocessing import clean_and_encode, CATEGORICAL_COLS, NUMERICAL_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_model = None
_encoders = None
_feature_cols = None


def load_artifacts():
    global _model, _encoders, _feature_cols
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
        _encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
        _feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))


def predict(input_data: dict) -> dict:
    """
    input_data : dict avec les features de la maison
    Retourne : {'predicted_price': float, 'price_range': [low, high]}
    """
    load_artifacts()
    df = pd.DataFrame([input_data])

    # Colonnes manquantes → valeur par défaut
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            df[col] = 0
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = 'None'

    X, _ = clean_and_encode(df, encoders=_encoders, fit=False)

    # Aligner sur les colonnes d'entraînement
    for col in _feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[_feature_cols]

    log_pred = _model.predict(X)[0]
    price = float(np.expm1(log_pred))

    return {
        'predicted_price': round(price, 2),
        'price_range': [round(price * 0.9, 2), round(price * 1.1, 2)],
        'currency': 'USD'
    }
