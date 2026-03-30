# 🏠 House Price Predictor — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

> Prédiction du prix de vente de maisons à partir de 79 variables — du notebook au déploiement Docker.

---

## 🎯 Problème

Estimer le prix d'un bien immobilier est complexe : des dizaines de variables entrent en jeu (surface, qualité, localisation, année de construction…). Ce projet répond à la question : **peut-on prédire automatiquement le prix de vente d'une maison avec une précision exploitable ?**

**Dataset** : [Kaggle House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) — 1 460 maisons à Ames, Iowa, avec 79 features.

---

## 💡 Solution

Pipeline ML complet :

```
CSV brut → Preprocessing → Feature Engineering → Modèle ML → API REST → Interface web
```

1. **Data cleaning** : gestion des 19 colonnes avec valeurs manquantes
2. **Feature engineering** : 4 features créées (TotalSF, HouseAge, Remodeled, QualSF)
3. **Comparaison de modèles** : Gradient Boosting, Random Forest, Ridge — sélection automatique du meilleur
4. **API REST** avec FastAPI (endpoint `/predict`, docs Swagger automatiques)
5. **Interface Streamlit** interactive avec formulaire et visualisation des résultats
6. **Containerisation Docker** avec docker-compose (API + App séparés)

---

## 📊 Résultats

| Modèle | CV RMSLE (5-fold) |
|--------|-------------------|
| Gradient Boosting | ~0.135 |
| Random Forest | ~0.148 |
| Ridge | ~0.162 |

> RMSLE = Root Mean Squared Log Error — métrique standard Kaggle pour ce dataset.
> Un RMSLE de 0.135 correspond à une erreur moyenne d'environ ±13% sur le prix.

---

## 🚀 Lancement rapide

### Sans Docker

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer train.csv dans data/

# 3. Entraîner le modèle
cd src && python train.py

# 4. Lancer l'API
uvicorn api.main:app --reload

# 5. Lancer l'interface (nouveau terminal)
streamlit run app/streamlit_app.py
```

### Avec Docker (recommandé)

```bash
# Placer train.csv dans data/ puis :
docker-compose up --build
```

- Interface : http://localhost:8501
- API docs : http://localhost:8000/docs

---

## 📁 Structure du projet

```
house-price-predictor/
├── data/               # train.csv, test.csv
├── models/             # modèle + encodeurs sauvegardés
├── src/
│   ├── preprocessing.py   # nettoyage + feature engineering
│   ├── train.py           # entraînement + comparaison modèles
│   └── predict.py         # inférence
├── api/
│   └── main.py            # API FastAPI
├── app/
│   └── streamlit_app.py   # interface utilisateur
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔌 API — Exemple d'utilisation

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "YearBuilt": 2000,
    "Neighborhood": "NAmes",
    "FullBath": 2,
    "BedroomAbvGr": 3
  }'
```

Réponse :
```json
{
  "predicted_price": 187450.0,
  "price_range": [168705.0, 206195.0],
  "currency": "USD"
}
```

---

## 🛠 Stack technique

| Composant | Technologie |
|-----------|-------------|
| ML | scikit-learn (GradientBoosting, RandomForest, Ridge) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Sérialisation | joblib |
| Containerisation | Docker + docker-compose |
| Data | pandas, numpy |

---

## 📈 Axes d'amélioration

- [ ] Ajouter XGBoost / LightGBM
- [ ] Stacking des modèles
- [ ] Déploiement cloud (Render, Railway, Streamlit Cloud)
- [ ] Tests unitaires (pytest)
- [ ] CI/CD GitHub Actions

---

## 👤 Auteur

Projet portfolio — Data Science / ML Engineer
