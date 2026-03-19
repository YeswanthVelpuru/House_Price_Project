import os
import pickle
import numpy as np
import torch
import streamlit as st
import pandas as pd
import shap
import geohash2
from geopy.distance import geodesic
import altair as alt

from image_features import get_image_features
from market_features import scrape_market_trends
from graph_features import estimate_gnn_price
from rl_price_trend import RLPriceAgent
from model_training import HousePricePredictor


# ================================
# PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():

    def check_file(path, name):
        if not os.path.exists(path):
            st.error(f"❌ Missing file: {name} at {path}")
            st.stop()

    check_file(ARTIFACTS_PATH, "artifacts.pkl")
    check_file(SCALER_PATH, "scaler.pkl")
    check_file(MODEL_PATH, "house_price_model.pth")

    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = HousePricePredictor(input_dim=2685, image_dim=2560)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    rl_agent = RLPriceAgent()

    return artifacts, scaler, model, rl_agent


# ================================
# 🔥 STATE LAYER
# ================================
artifacts, scaler, model, rl_agent = load_models()

APP_STATE = {
    "artifacts": artifacts,
    "scaler": scaler,
    "model": model,
    "rl_agent": rl_agent
}


# ================================
# PRICING
# ================================
def calculate_indian_tiered_valuation(lat, lon, sqft, raw_nn_output):

    acre_factor = sqft / 43560.0

    metros = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Hyderabad": (17.385, 78.486),
        "Bangalore": (12.971, 77.594),
        "Chennai": (13.082, 80.270),
        "Kolkata": (22.5726, 88.3639),
    }

    dist_to_metro = min([geodesic((lat, lon), m).km for m in metros.values()])
    urban_intensity = np.clip(abs(raw_nn_output) / 1e6, 0.1, 1.0)

    if dist_to_metro < 25 or urban_intensity > 0.8:
        zone, min_p, max_p = "Metropolitan", 10e7, 130e7
    elif dist_to_metro < 50 or urban_intensity > 0.5:
        zone, min_p, max_p = "Urban", 2e7, 10e7
    elif dist_to_metro < 80 or urban_intensity > 0.3:
        zone, min_p, max_p = "Semi-Urban", 0.3e7, 1.5e7
    else:
        zone, min_p, max_p = "Rural Village", 0.1e7, 0.25e7

    price_per_acre = min_p + (urban_intensity * (max_p - min_p))
    return price_per_acre * acre_factor, zone


# ================================
# FEATURE EXTRACTION
# ================================
def extract_live_features(lat, lon, sqft, beds, baths, state):

    scaler = state["scaler"]

    sat = get_image_features(lat, lon, style="satellite-v9")
    street = get_image_features(lat, lon, style="streets-v11")

    image_block = np.hstack([sat, street])

    gh = geohash2.encode(lat, lon, precision=6)
    vals = [lat, lon, sqft, beds, baths, 2026, int(gh.encode().hex(), 16) % 100000]

    tabular_vector = np.array(vals, dtype=np.float32)

    if len(tabular_vector) > 125:
        tabular_vector = tabular_vector[:125]
    else:
        tabular_vector = np.pad(tabular_vector, (0, 125 - len(tabular_vector)))

    full_features = np.hstack([image_block, tabular_vector]).reshape(1, -1)

    return torch.tensor(scaler.transform(full_features), dtype=torch.float32)


# ================================
# SHAP
# ================================
def compute_shap_explanations(features, state):

    model = state["model"]

    explainer = shap.GradientExplainer(model, torch.zeros((5, features.shape[1])))
    shap_vals = np.array(explainer.shap_values(features)).flatten()

    # Total absolute importance
    total = np.sum(np.abs(shap_vals)) + 1e-8

    # Feature names (same as before)
    feature_names = [
        "Property Size (sqft)", "Bedrooms", "Bathrooms", "Property Age", "Floor Level", "Parking",
        "Latitude", "Longitude", "Distance to City Center", "Distance to Metro", "Distance to Airport",
        "Nearby Schools", "Nearby Hospitals", "Nearby Malls", "Nearby Restaurants", "Nearby Parks",
        "Road Connectivity", "Traffic Density", "Public Transport", "Urban Index",
        "Greenery Score", "Air Quality", "Water Proximity", "Noise Level",
        "Building Quality", "Street Condition", "Neighborhood Aesthetics", "Urban Density", "Road Layout",
        "Market Demand", "Price Trend", "Economic Activity",
        "GNN Neighbor Price", "Cluster Effect", "RL Adjustment"
    ]

    shap_vals = shap_vals[:len(feature_names)]

    data = []
    for name, val in zip(feature_names, shap_vals):
        pct = (abs(val) / total) * 100

        direction = "Increase ↑" if val > 0 else "Decrease ↓"

        data.append({
            "Feature": name,
            "Impact %": round(pct, 2),
            "Effect": direction
        })

    df = pd.DataFrame(data).sort_values(by="Impact %", ascending=False)

    return df

# ================================
# UI
# ================================
st.title("Multimodal Geo-Spatial Price Prediction Framework")

c1, c2 = st.columns([1, 1.2])

with c1:
    sqft = st.slider("Area (sqft)", 500, 20000, 2000)
    beds = st.slider("Bedrooms", 1, 12, 3)
    baths = st.slider("Bathrooms", 1.0, 10.0, 2.0)

    lat = st.number_input("Latitude", value=19.0760)
    lon = st.number_input("Longitude", value=72.8777)

    if st.button("Predict Market Value"):

        features = extract_live_features(lat, lon, sqft, beds, baths, APP_STATE)

        model = APP_STATE["model"]
        rl_agent = APP_STATE["rl_agent"]

        with torch.no_grad():
            base_pred = model(features).numpy().reshape(-1)[0]

        gnn_val = estimate_gnn_price(lat, lon)
        market = scrape_market_trends()

        tiered_price, zone = calculate_indian_tiered_valuation(lat, lon, sqft, base_pred)
        final_price = rl_agent.adjust_price(tiered_price, market["demand_index"])

        st.success(f"💰 Price: ₹{abs(final_price):,.2f}")
        st.info(f"Zone: {zone}")

        shap_df = compute_shap_explanations(features, APP_STATE)

        chart = alt.Chart(shap_df).mark_bar().encode(
            x='Impact %',
            y=alt.Y('Feature', sort='-x'),
            color=alt.condition(
                alt.datum.Effect == "Increase ↑",
                alt.value("green"),
                alt.value("red")
            ),
            tooltip=["Feature", "Impact %", "Effect"]
        )

        st.altair_chart(chart, use_container_width=True)
    
with c2:
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=12)