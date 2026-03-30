import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from preprocessing import clean_and_encode

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))))


def train():
    print("Chargement du dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    y = df['SalePrice']
    df = df.drop(columns=['SalePrice', 'Id'], errors='ignore')

    print("Preprocessing...")
    X, encoders = clean_and_encode(df, fit=True)

    # Log-transform de la cible (réduit le skew)
    y_log = np.log1p(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Ridge': Ridge(alpha=10)
    }

    results = {}
    trained = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y_log, cv=kf,
                                 scoring='neg_root_mean_squared_error')
        rmse_cv = -scores.mean()
        print(f"{name:25s} — CV RMSLE: {rmse_cv:.4f} ± {scores.std():.4f}")
        model.fit(X, y_log)
        trained[name] = model
        results[name] = round(float(rmse_cv), 4)

    # Sauvegarde du meilleur modèle
    best_name = min(results, key=results.get)
    print(f"\nMeilleur modèle : {best_name} (RMSLE={results[best_name]})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(trained[best_name], os.path.join(MODELS_DIR, 'model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_cols.pkl'))

    metrics = {
        'best_model': best_name,
        'cv_rmsle': results[best_name],
        'all_scores': results,
        'n_features': len(X.columns)
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nModèle sauvegardé dans models/")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    train()
