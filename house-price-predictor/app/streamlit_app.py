import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.markdown("Estimez le prix d'une maison en quelques secondes grâce au Machine Learning.")

# ── Sidebar : infos modèle ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Infos modèle")
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            m = r.json()
            st.success("Modèle chargé ✅")
            st.metric("Meilleur modèle", m.get("best_model", "N/A"))
            st.metric("CV RMSLE", f"{m.get('cv_rmsle', 0):.4f}")
            st.metric("Nb features", m.get("n_features", "N/A"))
        else:
            st.warning("API accessible mais modèle non chargé")
    except Exception:
        st.error("API non accessible — lancez `uvicorn api.main:app`")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("- scikit-learn\n- FastAPI\n- Streamlit\n- Docker")

# ── Formulaire principal ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏗 Structure")
    overall_qual = st.slider("Qualité générale (1-10)", 1, 10, 6)
    year_built = st.number_input("Année de construction", 1870, 2024, 1995)
    year_remod = st.number_input("Année rénovation", 1870, 2024, 1995)
    gr_liv_area = st.number_input("Surface habitable (sqft)", 500, 6000, 1500)

with col2:
    st.subheader("🛁 Intérieur")
    full_bath = st.selectbox("Salles de bain complètes", [1, 2, 3, 4], index=1)
    bedroom = st.selectbox("Chambres", [1, 2, 3, 4, 5, 6], index=2)
    kitchen_qual = st.selectbox("Qualité cuisine", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    total_bsmt = st.number_input("Surface sous-sol (sqft)", 0, 3000, 800)

with col3:
    st.subheader("🌍 Localisation")
    neighborhood = st.selectbox("Quartier", [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
        "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
        "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber"
    ])
    ms_zoning = st.selectbox("Zonage", ["RL", "RM", "FV", "RH", "C (all)"])
    lot_area = st.number_input("Surface terrain (sqft)", 1000, 50000, 8500)
    garage_cars = st.selectbox("Places de garage", [0, 1, 2, 3, 4], index=2)

st.markdown("---")

# ── Prédiction ──────────────────────────────────────────────────────────────
if st.button("🔮 Estimer le prix", type="primary", use_container_width=True):
    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "Neighborhood": neighborhood,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom,
        "KitchenQual": kitchen_qual,
        "MSZoning": ms_zoning,
        "LotArea": lot_area
    }

    with st.spinner("Calcul en cours..."):
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                result = r.json()
                price = result["predicted_price"]
                low, high = result["price_range"]

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("💰 Prix estimé", f"${price:,.0f}")
                with col_r2:
                    st.metric("📉 Fourchette basse", f"${low:,.0f}")
                with col_r3:
                    st.metric("📈 Fourchette haute", f"${high:,.0f}")

                st.info(f"Intervalle de confiance : **${low:,.0f}** — **${high:,.0f}** (±10%)")

                with st.expander("📋 Données envoyées à l'API"):
                    st.json(payload)
            else:
                st.error(f"Erreur API : {r.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de joindre l'API. Lancez d'abord : `uvicorn api.main:app --reload`")
        except Exception as e:
            st.error(f"Erreur : {e}")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Projet portfolio data science · Kaggle House Prices Dataset · scikit-learn + FastAPI + Streamlit")
