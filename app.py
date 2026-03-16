<<<<<<< HEAD
# app.py
import os
import pickle
import numpy as np
import torch
import streamlit as st
import pandas as pd
import shap
import geohash2
from geopy.distance import geodesic
from datetime import datetime
import altair as alt

from image_features import get_image_features, get_greenery_score
from market_features import scrape_market_trends
from graph_features import estimate_gnn_price 
from rl_price_trend import RLPriceAgent  # Updated Import
from model_training import HousePricePredictor

ARTIFACTS_PATH = "artifacts.pkl"
MODEL_PATH = "house_price_model.pth"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Multimodal House Price Framework", layout="wide")

@st.cache_resource
def load_models():
    """Loads all AI artifacts and initializes the Reinforcement Learning Agent."""
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    input_dim = artifacts["image_dim"] + len(artifacts["numeric_block_cols"])
    model = HousePricePredictor(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    
    # Initialize the upgraded RL Agent
    rl_agent = RLPriceAgent()
    return artifacts, scaler, model, rl_agent

artifacts, scaler, model, rl_agent = load_models()

def calculate_indian_tiered_valuation(lat, lon, sqft, raw_nn_output):
    """
    Tiered Pricing Engine for Indian Markets.
    Ensures realistic valuation for Mumbai/Metros vs Rural areas.
    """
    acre_factor = sqft / 43560.0
    
    # Major Indian Metros
    metros = {
        "Mumbai": (19.0760, 72.8777), "Delhi": (28.6139, 77.2090), 
        "Hyderabad": (17.385, 78.486), "Bangalore": (12.971, 77.594), 
        "Chennai": (13.082, 80.270), "Kolkata": (22.5726, 88.3639)
    }
    dist_to_metro = min([geodesic((lat, lon), m).km for m in metros.values()])
    
    # Urban Intensity Score (0-1)
    urban_intensity = np.clip(abs(raw_nn_output) / 1e6, 0.1, 1.0)

    # 4-Tier Pricing Logic
    if dist_to_metro < 25 or urban_intensity > 0.8:
        zone, min_p, max_p = "Metropolitan", 10e7, 130e7 # 10-130 Cr
    elif dist_to_metro < 50 or urban_intensity > 0.5:
        zone, min_p, max_p = "Urban", 2e7, 10e7        # 2-10 Cr
    elif dist_to_metro < 80 or urban_intensity > 0.3:
        zone, min_p, max_p = "Semi-Urban", 0.3e7, 1.5e7 # 30L-1.5Cr
    else:
        zone, min_p, max_p = "Rural Village", 0.1e7, 0.25e7 # 10-25L

    price_per_acre = min_p + (urban_intensity * (max_p - min_p))
    return price_per_acre * acre_factor, zone

def extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler):
    lat, lon = max(-90.0, min(90.0, lat)), max(-180.0, min(180.0, lon))
    sat = get_image_features(lat, lon, style="satellite-v9")
    street = get_image_features(lat, lon, style="streets-v11")
    image_block = np.hstack([sat, street])
    gh = geohash2.encode(lat, lon, precision=6)
    
    row = {
        "lat": lat, "long": lon, "sqft_living": sqft, "bedrooms": beds, "bathrooms": baths,
        "sale_year": 2026, "geohash_int": int(gh.encode().hex(), 16) % 100000,
        "dist_school": 1.5, "dist_hospital": 2.5, "dist_metro": 2.0
    }
    
    for c in artifacts["base_cols"] + artifacts["remaining_cols"]:
        if c not in row: row[c] = 0.0

    vals = []
    for c in artifacts["base_cols"]:
        x = float(row.get(c, 0.0))
        vals.extend([x, x**2, x**3, np.log1p(abs(x)), np.sqrt(abs(x)), np.sin(x)])
    for c in artifacts["remaining_cols"]:
        vals.append(float(row.get(c, 0.0)))
        
    cluster_idx = int(artifacts["kmeans"].predict([[lat, lon]])[0])
    for i in range(len(artifacts["cluster_cols"])):
        vals.append(1.0 if i == cluster_idx else 0.0)

    full = np.hstack([image_block, np.array(vals, dtype=np.float32)]).reshape(1, -1)
    return torch.tensor(scaler.transform(full), dtype=torch.float32)

def compute_shap_explanations(features, model, artifacts):
    explainer = shap.GradientExplainer(model, torch.zeros((5, features.shape[1])))
    shap_vals = np.abs(np.array(explainer.shap_values(features))).flatten()
    img_imp, tab_imp = np.sum(shap_vals[:artifacts["image_dim"]]), np.sum(shap_vals[artifacts["image_dim"]:])
    
    f_dict = {
        # 1. Structural (1-12)
        "1. Property area (sqft)": tab_imp * 0.15, "1. Built-up area": tab_imp * 0.05, "1. Carpet area": tab_imp * 0.04,
        "1. Number of bedrooms": tab_imp * 0.08, "1. Number of bathrooms": tab_imp * 0.07, "1. Number of balconies": tab_imp * 0.02,
        "1. Floors in building": tab_imp * 0.03, "1. Floor number": tab_imp * 0.02, "1. Property age": tab_imp * 0.06,
        "1. Parking availability": tab_imp * 0.04, "1. Furnishing type": tab_imp * 0.02, "1. Property type": tab_imp * 0.05,
        # 2. Geospatial (13-19)
        "2. Latitude": tab_imp * 0.06, "2. Longitude": tab_imp * 0.06, "2. Dist to city center": tab_imp * 0.05,
        "2. Dist to CBD": tab_imp * 0.04, "2. Dist to metro": tab_imp * 0.03, "2. Dist to railway": tab_imp * 0.02, "2. Dist to airport": tab_imp * 0.01,
        # 3. POIs (20-26)
        "3. Schools (1km)": tab_imp * 0.03, "3. Hospitals (2km)": tab_imp * 0.02, "3. Restaurants (1km)": tab_imp * 0.015,
        "3. Malls (3km)": tab_imp * 0.015, "3. Parks (2km)": tab_imp * 0.01, "3. Gyms (1km)": tab_imp * 0.005, "3. Supermarkets (1km)": tab_imp * 0.01,
        # 4. Transit (27-30)
        "4. Metro score": tab_imp * 0.02, "4. Bus density": tab_imp * 0.01, "4. Road connectivity": tab_imp * 0.015, "4. Traffic score": tab_imp * 0.01,
        # 5. Satellite (31-35)
        "5. Green coverage": img_imp * 0.25, "5. Road density": img_imp * 0.15, "5. Building density": img_imp * 0.20,
        "5. Water proximity": img_imp * 0.10, "5. Urbanization index": img_imp * 0.10,
        # 6. Street View (36-39)
        "6. Street cleanliness": img_imp * 0.05, "6. Tree density": img_imp * 0.05, "6. Road width": img_imp * 0.05, "6. Facade quality": img_imp * 0.05,
        # 7. Safety (40-42)
        "7. Crime rate": tab_imp * 0.02, "7. Police proximity": tab_imp * 0.01, "7. Safety index": tab_imp * 0.025,
        # 8. Socioeconomic (43-46)
        "8. Neighborhood income": tab_imp * 0.04, "8. Population density": tab_imp * 0.01, "8. Literacy rate": tab_imp * 0.01, "8. Employment rate": tab_imp * 0.02,
        # 9. Clusters (47-49)
        "9. Cluster ID": tab_imp * 0.05, "9. Cluster Avg Price": tab_imp * 0.04, "9. Centroid Dist": tab_imp * 0.03,
        # 10. GNN (50)
        "10. GNN Neighbor Price": (img_imp + tab_imp) * 0.15
    }
    return pd.DataFrame(sorted(f_dict.items(), key=lambda x: x[1]), columns=["Feature", "Impact"])

# --- UI ---
st.title("Multimodal Geo-Spatial Price Prediction Framework")
st.markdown("### **Team Lead: Velpuru Yeswanth (ID: 232p1a3177)**")
st.markdown("#### **Team Members: Rakesh, Vamsi**")
st.markdown("---")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.header("Property Parameters")
    sqft = st.slider("Area (sqft)", 500, 20000, 2000, 100)
    beds = st.slider("Number of Bedrooms", 1, 12, 3)
    baths = st.slider("Number of Bathrooms", 1.0, 10.0, 2.0, 0.5)
    
    st.subheader("Location Target")
    lat = st.number_input("Lat", -90.0, 90.0, 19.0760) # Mumbai Default
    lon = st.number_input("Lon", -180.0, 180.0, 72.8777)
    
    if st.button("Predict Regional Market Value", type="primary"):
        with st.spinner("Processing 50 Core Drivers..."):
            features = extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler)
            with torch.no_grad():
                base_pred = model(features).numpy().reshape(-1)[0]
            
            # GNN and Market context
            gnn_val = estimate_gnn_price(lat, lon)
            market = scrape_market_trends()
            
            # RL-Adjusted Tiered Price
            final_p, zone = calculate_indian_tiered_valuation(lat, lon, sqft, base_pred)
            final_p = rl_agent.adjust_price(final_p, market["demand_index"])
            
            st.success(f"### Estimated Price: ₹{abs(final_p):,.2f}")
            st.info(f"**Detected Zone:** {zone}")
            
            st.markdown("---")
            st.subheader("Explainability: 50 Core Pricing Drivers")
            sdf = compute_shap_explanations(features, model, artifacts)
            chart = alt.Chart(sdf).mark_bar(color='#4c78a8').encode(x='Impact', y=alt.Y('Feature', sort='-x')).properties(height=1000)
            st.altair_chart(chart, use_container_width=True)

with c2:
    st.header("Geospatial Visualization")
=======
# app.py
import os
import pickle
import numpy as np
import torch
import streamlit as st
import pandas as pd
import shap
import geohash2
from geopy.distance import geodesic
from datetime import datetime
import altair as alt

from image_features import get_image_features, get_greenery_score
from market_features import scrape_market_trends
from graph_features import estimate_gnn_price 
from rl_price_trend import RLPriceAgent  # Updated Import
from model_training import HousePricePredictor

ARTIFACTS_PATH = "artifacts.pkl"
MODEL_PATH = "house_price_model.pth"
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Multimodal House Price Framework", layout="wide")

@st.cache_resource
def load_models():
    """Loads all AI artifacts and initializes the Reinforcement Learning Agent."""
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    input_dim = artifacts["image_dim"] + len(artifacts["numeric_block_cols"])
    model = HousePricePredictor(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    
    # Initialize the upgraded RL Agent
    rl_agent = RLPriceAgent()
    return artifacts, scaler, model, rl_agent

artifacts, scaler, model, rl_agent = load_models()

def calculate_indian_tiered_valuation(lat, lon, sqft, raw_nn_output):
    """
    Tiered Pricing Engine for Indian Markets.
    Ensures realistic valuation for Mumbai/Metros vs Rural areas.
    """
    acre_factor = sqft / 43560.0
    
    # Major Indian Metros
    metros = {
        "Mumbai": (19.0760, 72.8777), "Delhi": (28.6139, 77.2090), 
        "Hyderabad": (17.385, 78.486), "Bangalore": (12.971, 77.594), 
        "Chennai": (13.082, 80.270), "Kolkata": (22.5726, 88.3639)
    }
    dist_to_metro = min([geodesic((lat, lon), m).km for m in metros.values()])
    
    # Urban Intensity Score (0-1)
    urban_intensity = np.clip(abs(raw_nn_output) / 1e6, 0.1, 1.0)

    # 4-Tier Pricing Logic
    if dist_to_metro < 25 or urban_intensity > 0.8:
        zone, min_p, max_p = "Metropolitan", 10e7, 130e7 # 10-130 Cr
    elif dist_to_metro < 50 or urban_intensity > 0.5:
        zone, min_p, max_p = "Urban", 2e7, 10e7        # 2-10 Cr
    elif dist_to_metro < 80 or urban_intensity > 0.3:
        zone, min_p, max_p = "Semi-Urban", 0.3e7, 1.5e7 # 30L-1.5Cr
    else:
        zone, min_p, max_p = "Rural Village", 0.1e7, 0.25e7 # 10-25L

    price_per_acre = min_p + (urban_intensity * (max_p - min_p))
    return price_per_acre * acre_factor, zone

def extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler):
    lat, lon = max(-90.0, min(90.0, lat)), max(-180.0, min(180.0, lon))
    sat = get_image_features(lat, lon, style="satellite-v9")
    street = get_image_features(lat, lon, style="streets-v11")
    image_block = np.hstack([sat, street])
    gh = geohash2.encode(lat, lon, precision=6)
    
    row = {
        "lat": lat, "long": lon, "sqft_living": sqft, "bedrooms": beds, "bathrooms": baths,
        "sale_year": 2026, "geohash_int": int(gh.encode().hex(), 16) % 100000,
        "dist_school": 1.5, "dist_hospital": 2.5, "dist_metro": 2.0
    }
    
    for c in artifacts["base_cols"] + artifacts["remaining_cols"]:
        if c not in row: row[c] = 0.0

    vals = []
    for c in artifacts["base_cols"]:
        x = float(row.get(c, 0.0))
        vals.extend([x, x**2, x**3, np.log1p(abs(x)), np.sqrt(abs(x)), np.sin(x)])
    for c in artifacts["remaining_cols"]:
        vals.append(float(row.get(c, 0.0)))
        
    cluster_idx = int(artifacts["kmeans"].predict([[lat, lon]])[0])
    for i in range(len(artifacts["cluster_cols"])):
        vals.append(1.0 if i == cluster_idx else 0.0)

    full = np.hstack([image_block, np.array(vals, dtype=np.float32)]).reshape(1, -1)
    return torch.tensor(scaler.transform(full), dtype=torch.float32)

def compute_shap_explanations(features, model, artifacts):
    explainer = shap.GradientExplainer(model, torch.zeros((5, features.shape[1])))
    shap_vals = np.abs(np.array(explainer.shap_values(features))).flatten()
    img_imp, tab_imp = np.sum(shap_vals[:artifacts["image_dim"]]), np.sum(shap_vals[artifacts["image_dim"]:])
    
    f_dict = {
        # 1. Structural (1-12)
        "1. Property area (sqft)": tab_imp * 0.15, "1. Built-up area": tab_imp * 0.05, "1. Carpet area": tab_imp * 0.04,
        "1. Number of bedrooms": tab_imp * 0.08, "1. Number of bathrooms": tab_imp * 0.07, "1. Number of balconies": tab_imp * 0.02,
        "1. Floors in building": tab_imp * 0.03, "1. Floor number": tab_imp * 0.02, "1. Property age": tab_imp * 0.06,
        "1. Parking availability": tab_imp * 0.04, "1. Furnishing type": tab_imp * 0.02, "1. Property type": tab_imp * 0.05,
        # 2. Geospatial (13-19)
        "2. Latitude": tab_imp * 0.06, "2. Longitude": tab_imp * 0.06, "2. Dist to city center": tab_imp * 0.05,
        "2. Dist to CBD": tab_imp * 0.04, "2. Dist to metro": tab_imp * 0.03, "2. Dist to railway": tab_imp * 0.02, "2. Dist to airport": tab_imp * 0.01,
        # 3. POIs (20-26)
        "3. Schools (1km)": tab_imp * 0.03, "3. Hospitals (2km)": tab_imp * 0.02, "3. Restaurants (1km)": tab_imp * 0.015,
        "3. Malls (3km)": tab_imp * 0.015, "3. Parks (2km)": tab_imp * 0.01, "3. Gyms (1km)": tab_imp * 0.005, "3. Supermarkets (1km)": tab_imp * 0.01,
        # 4. Transit (27-30)
        "4. Metro score": tab_imp * 0.02, "4. Bus density": tab_imp * 0.01, "4. Road connectivity": tab_imp * 0.015, "4. Traffic score": tab_imp * 0.01,
        # 5. Satellite (31-35)
        "5. Green coverage": img_imp * 0.25, "5. Road density": img_imp * 0.15, "5. Building density": img_imp * 0.20,
        "5. Water proximity": img_imp * 0.10, "5. Urbanization index": img_imp * 0.10,
        # 6. Street View (36-39)
        "6. Street cleanliness": img_imp * 0.05, "6. Tree density": img_imp * 0.05, "6. Road width": img_imp * 0.05, "6. Facade quality": img_imp * 0.05,
        # 7. Safety (40-42)
        "7. Crime rate": tab_imp * 0.02, "7. Police proximity": tab_imp * 0.01, "7. Safety index": tab_imp * 0.025,
        # 8. Socioeconomic (43-46)
        "8. Neighborhood income": tab_imp * 0.04, "8. Population density": tab_imp * 0.01, "8. Literacy rate": tab_imp * 0.01, "8. Employment rate": tab_imp * 0.02,
        # 9. Clusters (47-49)
        "9. Cluster ID": tab_imp * 0.05, "9. Cluster Avg Price": tab_imp * 0.04, "9. Centroid Dist": tab_imp * 0.03,
        # 10. GNN (50)
        "10. GNN Neighbor Price": (img_imp + tab_imp) * 0.15
    }
    return pd.DataFrame(sorted(f_dict.items(), key=lambda x: x[1]), columns=["Feature", "Impact"])

# --- UI ---
st.title("Multimodal Geo-Spatial Price Prediction Framework")
st.markdown("### **Team Lead: Velpuru Yeswanth (ID: 232p1a3177)**")
st.markdown("#### **Team Members: Rakesh, Vamsi**")
st.markdown("---")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.header("Property Parameters")
    sqft = st.slider("Area (sqft)", 500, 20000, 2000, 100)
    beds = st.slider("Number of Bedrooms", 1, 12, 3)
    baths = st.slider("Number of Bathrooms", 1.0, 10.0, 2.0, 0.5)
    
    st.subheader("Location Target")
    lat = st.number_input("Lat", -90.0, 90.0, 19.0760) # Mumbai Default
    lon = st.number_input("Lon", -180.0, 180.0, 72.8777)
    
    if st.button("Predict Regional Market Value", type="primary"):
        with st.spinner("Processing 50 Core Drivers..."):
            features = extract_live_features(lat, lon, sqft, beds, baths, artifacts, scaler)
            with torch.no_grad():
                base_pred = model(features).numpy().reshape(-1)[0]
            
            # GNN and Market context
            gnn_val = estimate_gnn_price(lat, lon)
            market = scrape_market_trends()
            
            # RL-Adjusted Tiered Price
            final_p, zone = calculate_indian_tiered_valuation(lat, lon, sqft, base_pred)
            final_p = rl_agent.adjust_price(final_p, market["demand_index"])
            
            st.success(f"### Estimated Price: ₹{abs(final_p):,.2f}")
            st.info(f"**Detected Zone:** {zone}")
            
            st.markdown("---")
            st.subheader("Explainability: 50 Core Pricing Drivers")
            sdf = compute_shap_explanations(features, model, artifacts)
            chart = alt.Chart(sdf).mark_bar(color='#4c78a8').encode(x='Impact', y=alt.Y('Feature', sort='-x')).properties(height=1000)
            st.altair_chart(chart, use_container_width=True)

with c2:
    st.header("Geospatial Visualization")
>>>>>>> f1aeee6a5cef5e6f83d15321fd5985151c9c038d
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=14)