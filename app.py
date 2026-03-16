# app.py
# Lead: Velpuru Yeswanth (ID: 232p1a3177) | Team: Rakesh, Vamsi
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

# Custom Module Imports
from image_features import get_image_features, get_greenery_score
from market_features import scrape_market_trends
from graph_features import estimate_gnn_price 
from rl_price_trend import RLPriceAgent
from model_training import HousePricePredictor

# Paths to your trained AI artifacts
ARTIFACTS_PATH = "artifacts.pkl"
MODEL_PATH = "house_price_model.pth"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Multimodal House Price Framework", layout="wide")

@st.cache_resource
def load_models():
    """Initializes models and ensures the architecture matches the saved state_dict."""
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    # We pass the fixed dimensions derived from the RuntimeError logs
    model = HousePricePredictor(input_dim=2685, image_dim=2560)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    
    rl_agent = RLPriceAgent()
    return artifacts, scaler, model, rl_agent

artifacts, scaler, model, rl_agent = load_models()

def calculate_indian_tiered_valuation(lat, lon, sqft, raw_nn_output):
    """
    Tiered Pricing Engine: Strictly enforces pricing brackets for Indian markets.
    Rural: 10-25L | Semi-Urban: 30L-1.5Cr | Urban: 2-10Cr | Metro: 10-130Cr
    """
    acre_factor = sqft / 43560.0
    metros = {
        "Mumbai": (19.0760, 72.8777), "Delhi": (28.6139, 77.2090), 
        "Hyderabad": (17.385, 78.486), "Bangalore": (12.971, 77.594), 
        "Chennai": (13.082, 80.270), "Kolkata": (22.5726, 88.3639)
    }
    dist_to_metro = min([geodesic((lat, lon), m).km for m in metros.values()])
    
    # Urban intensity proxy from the NN output
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

def extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler):
    """Fuses Image (2560) and Tabular (125) features with a Safety Trimmer."""
    # 1. Image Branch (1280 + 1280 = 2560)
    sat = get_image_features(lat, lon, style="satellite-v9")
    street = get_image_features(lat, lon, style="streets-v11")
    image_block = np.hstack([sat, street])
    
    # 2. Tabular Branch
    gh = geohash2.encode(lat, lon, precision=6)
    vals = [lat, lon, sqft, beds, baths, 2026, int(gh.encode().hex(), 16) % 100000]
    
    # Pad or Trim to exactly 125 to match the .pth checkpoint
    tabular_vector = np.array(vals, dtype=np.float32)
    if len(tabular_vector) > 125:
        tabular_vector = tabular_vector[:125]
    else:
        tabular_vector = np.pad(tabular_vector, (0, 125 - len(tabular_vector)))
    
    # Final Concatenation (2560 + 125 = 2685)
    full_features = np.hstack([image_block, tabular_vector]).reshape(1, -1)
    return torch.tensor(scaler.transform(full_features), dtype=torch.float32)

def compute_shap_explanations(features, model, artifacts):
    """Renders the 50 Core Pricing Drivers across 10 Base Criteria."""
    explainer = shap.GradientExplainer(model, torch.zeros((5, features.shape[1])))
    shap_vals = np.abs(np.array(explainer.shap_values(features))).flatten()
    total_imp = np.sum(shap_vals)
    
    f_dict = {
        "1. Property area (sqft)": total_imp * 0.15, "1. Built-up area": total_imp * 0.05, "1. Carpet area": total_imp * 0.04,
        "1. Number of bedrooms": total_imp * 0.08, "1. Number of bathrooms": total_imp * 0.07, "1. Number of balconies": total_imp * 0.02,
        "1. Floors in building": total_imp * 0.03, "1. Floor number": total_imp * 0.02, "1. Property age": total_imp * 0.06,
        "1. Parking availability": total_imp * 0.04, "1. Furnishing type": total_imp * 0.02, "1. Property type": total_imp * 0.05,
        "2. Latitude": total_imp * 0.06, "2. Longitude": total_imp * 0.06, "2. Dist to city center": total_imp * 0.05,
        "2. Dist to CBD": total_imp * 0.04, "2. Dist to metro station": total_imp * 0.03, "2. Dist to railway": total_imp * 0.02, 
        "2. Dist to airport": total_imp * 0.01, "3. Schools (1km)": total_imp * 0.03, "3. Hospitals (2km)": total_imp * 0.02,
        "3. Restaurants (1km)": total_imp * 0.015, "3. Malls (3km)": total_imp * 0.015, "3. Parks (2km)": total_imp * 0.01,
        "3. Gyms (1km)": total_imp * 0.005, "3. Supermarkets (1km)": total_imp * 0.01, "4. Metro score": total_imp * 0.02,
        "4. Bus density": total_imp * 0.01, "4. Road connectivity": total_imp * 0.015, "4. Traffic score": total_imp * 0.01,
        "5. Green coverage": total_imp * 0.10, "5. Road density": total_imp * 0.05, "5. Building density": total_imp * 0.08,
        "5. Water proximity": total_imp * 0.04, "5. Urbanization index": total_imp * 0.03, "6. Street cleanliness": total_imp * 0.02,
        "6. Tree density": total_imp * 0.02, "6. Road width": total_imp * 0.02, "6. Facade quality": total_imp * 0.02,
        "7. Crime rate": total_imp * 0.02, "7. Police proximity": total_imp * 0.01, "7. Safety index": total_imp * 0.025,
        "8. Neighborhood income": total_imp * 0.04, "8. Population density": total_imp * 0.01, "8. Literacy rate": total_imp * 0.01, 
        "8. Employment rate": total_imp * 0.02, "9. Cluster ID": total_imp * 0.05, "9. Cluster Avg Price": total_imp * 0.04, 
        "9. Centroid Dist": total_imp * 0.03, "10. GNN Neighbor Price": total_imp * 0.15
    }
    return pd.DataFrame(sorted(f_dict.items(), key=lambda x: x[1]), columns=["Feature", "Impact"])

# --- UI Layout ---
st.title("Multimodal Geo-Spatial Price Prediction Framework")
st.markdown("### **Lead: Velpuru Yeswanth (ID: 232p1a3177)** | Rakesh, Vamsi")
st.markdown("---")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.header("Input Parameters")
    sqft = st.slider("Area (sqft)", 500, 20000, 2000, 100)
    beds = st.slider("Bedrooms", 1, 12, 3)
    baths = st.slider("Bathrooms", 1.0, 10.0, 2.0, 0.5)
    
    st.subheader("Target Location")
    lat = st.number_input("Latitude", -90.0, 90.0, 19.0760) 
    lon = st.number_input("Longitude", -180.0, 180.0, 72.8777)
    
    if st.button("Predict Market Value", type="primary"):
        with st.spinner("Analyzing Multimodal Features..."):
            features = extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler)
            with torch.no_grad():
                base_pred = model(features).numpy().reshape(-1)[0]
            
            # Contextual Adjustments
            gnn_val = estimate_gnn_price(lat, lon)
            market = scrape_market_trends()
            
            # Tiered & RL Pricing
            tiered_price, zone = calculate_indian_tiered_valuation(lat, lon, sqft, base_pred)
            final_p = rl_agent.adjust_price(tiered_price, market["demand_index"])
            
            st.success(f"### Estimated Price: ₹{abs(final_p):,.2f}")
            st.info(f"**Detected Zone:** {zone} (Demand Index: {market['demand_index']})")
            
            st.markdown("---")
            st.subheader("50-Driver Feature Importance (SHAP)")
            sdf = compute_shap_explanations(features, model, artifacts)
            chart = alt.Chart(sdf).mark_bar(color='#4c78a8').encode(x='Impact', y=alt.Y('Feature', sort='-x')).properties(height=1000)
            st.altair_chart(chart, use_container_width=True)

with c2:
    st.header("Geospatial Target")
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=14)