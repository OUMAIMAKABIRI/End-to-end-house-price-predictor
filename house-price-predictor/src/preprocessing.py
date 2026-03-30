import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

CATEGORICAL_COLS = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
]

NUMERICAL_COLS = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
    'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
    'MoSold', 'YrSold'
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features supplémentaires."""
    df = df.copy()
    # Surface totale
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    # Âge de la maison
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    # Rénovée récemment ?
    df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    # Qualité x Surface
    df['QualSF'] = df['OverallQual'] * df['GrLivArea']
    return df


def clean_and_encode(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    """Nettoyage + encodage des variables catégorielles."""
    df = df.copy()
    df = feature_engineering(df)

    # Remplissage des valeurs manquantes numériques
    for col in NUMERICAL_COLS + ['TotalSF', 'HouseAge', 'Remodeled', 'QualSF']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if fit else 0)

    # Remplissage des valeurs manquantes catégorielles
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Encodage label
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns and encoders and col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

    feature_cols = (
        NUMERICAL_COLS +
        CATEGORICAL_COLS +
        ['TotalSF', 'HouseAge', 'Remodeled', 'QualSF']
    )
    available = [c for c in feature_cols if c in df.columns]
    return df[available], encoders
