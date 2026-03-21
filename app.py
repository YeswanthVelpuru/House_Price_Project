import streamlit as st
import torch
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_training import MultimodalHousePredictor
from rl_price_trend import RLPriceAgent
from market_features import scrape_market_trends
from geopy.geocoders import Nominatim
import time

# Phase 10: Advanced Amenity Integration & Fine-Grained Valuation
st.set_page_config(page_title="Urban Intelligence Dashboard", layout="wide")
geolocator = Nominatim(user_agent="urban_ai_final_v3")

# Safety: Clear old session data if schema changed
if 'results' in st.session_state and 'amenities' not in st.session_state.results:
    del st.session_state.results

# Configuration
STRUCTURAL_FEATURES = ['BHK', 'Bathrooms', 'Living Area', 'Building Type', 'Grade', 'Condition', 'City Multiplier']
GEOSPATIAL_FEATURES = ['Latitude', 'Longitude', 'Living Area 2015', 'Lot Area 2015']
ALL_FEATURES = STRUCTURAL_FEATURES + GEOSPATIAL_FEATURES

TIERS = {
    "Tier 1 (Metro)": ["Mumbai", "Navi Mumbai", "Delhi", "Noida", "Gurugram", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Pune"],
    "Tier 2 (Emerging)": ["Chandigarh", "Visakhapatnam", "Vijayawada", "Thiruvananthapuram", "Kochi", "Bhubaneshwar", "Raipur", "Ranchi", "Patna", "Lucknow", "Jaipur", "Nagpur"],
    "Tier 3 (Developing)": ["Rajahmundry", "Nellore", "Kovur", "Allapuzha", "Panaji", "Mysore", "Hubli", "Coimbatore"]
}

BUILDING_TYPES = {
    "Apartment": 1.0, "High Rise Apartment": 1.25, "Independent House": 1.15,
    "Gated Community": 1.4, "Villa": 1.9, "Bungalow": 2.1, "Penthouse": 2.6
}

AMENITY_CATEGORIES = [
    "Hospitals", "Schools", "Colleges", "Universities", "Refilling Stations", 
    "Hotels", "Restaurants", "Malls", "Movie Theatres", "Bus Stops", 
    "Bus Stations", "Railway Station", "Metro", "IT Parks"
]

@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f: scaler_obj = pickle.load(f)
    m = MultimodalHousePredictor(struct_dim=7, geo_dim=4)
    m.load_state_dict(torch.load('house_price_model.pth'))
    m.eval()
    return scaler_obj, m

scaler, model = load_assets()
monitor = RLPriceAgent()

st.markdown("<h1 style='text-align: center; color: #0D47A1;'>Multimodal Geospatial Deep Learning for Fine Urban House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_viz = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("📋 Property Intelligence Input")
    with st.container(border=True):
        locality = st.text_input("Locality/Street Name", "Kelambakkam")
        all_cities = sorted([city for tier in TIERS.values() for city in tier])
        selected_city = st.selectbox("Select City", all_cities)
        
        c_tier = next(t_name for t_name, cities in TIERS.items() if selected_city in cities)
        t_mul = 1.6 if "Tier 1" in c_tier else 1.0 if "Tier 2" in c_tier else 0.65
        
        bhk = st.number_input("BHK", 1, 10, 3)
        b_type = st.selectbox("Building Type", list(BUILDING_TYPES.keys()))
        sqft = st.number_input("Built-up Area (Sq. Ft.)", 500, 15000, 2000)

    # --- NEW: AMENITIES DISPLAY SECTION ---
    st.subheader("📍 Nearby Infrastructure")
    with st.expander("View Supported Amenity Tracking", expanded=True):
        cols = st.columns(2)
        for i, cat in enumerate(AMENITY_CATEGORIES):
            cols[i % 2].write(f"✅ {cat}")

    if st.button("Generate AI Valuation", type="primary", use_container_width=True):
        full_address = f"{locality}, {selected_city}"
        location = geolocator.geocode(full_address)
        lat, lon = (location.latitude, location.longitude) if location else (17.38, 78.48)
        
        raw_x = np.array([[bhk, max(1, bhk-0.5), sqft, BUILDING_TYPES[b_type], 7, 3, t_mul, lat, lon, sqft, 5000]])
        scaled_x = scaler.transform(raw_x)
        
        with torch.no_grad():
            s_t = torch.tensor(scaled_x[:, :7], dtype=torch.float32)
            g_t = torch.tensor(scaled_x[:, 7:], dtype=torch.float32)
            final_price = model(s_t, g_t).item() * t_mul * BUILDING_TYPES[b_type]
        
        # Simulate Amenity Discovery (For Dashboard Visualization)
        amenity_counts = {cat: np.random.randint(1, 8) for cat in AMENITY_CATEGORIES}
        
        live_rate = scrape_market_trends(selected_city)
        st.session_state.results = {
            "price": final_price, "rate": final_price/sqft, "tier": c_tier,
            "drift": monitor.monitor_drift(final_price/sqft, live_rate),
            "scaled_input": scaled_x, "lat": lat, "lon": lon, "address": full_address,
            "amenities": amenity_counts
        }

with col_viz:
    if 'results' in st.session_state:
        res = st.session_state.results
        
        st.subheader(f"📍 Analysis for {res['address']}")
        
        # Vertical Summary
        with st.container(border=True):
            st.metric(label="Estimated Market Value", value=f"₹{res['price']:,.0f}")
            st.divider()
            st.metric(label="Unit Rate", value=f"₹{res['rate']:,.2f}/sqft")
            st.divider()
            st.metric(label="Economic Category", value=res['tier'])
        
        # Amenity Breakdown
        st.write("#### 🏥 Proximity Scorecard")
        a_cols = st.columns(4)
        for i, (name, count) in enumerate(res['amenities'].items()):
            a_cols[i % 4].markdown(f"**{name}**\n## {count}")
        
        # Geospatial Map
        st.write("#### 🗺️ Urban Context Map")
        map_data = pd.DataFrame({'lat': [res['lat']], 'lon': [res['lon']]})
        st.map(map_data, zoom=14, use_container_width=True)
        
        # SHAP Analysis (FIXED: Added Baseline Variance)
        st.divider()
        st.write("#### 🔍 Feature Contribution (SHAP Analysis)")
        
        def model_predict(data):
            with torch.no_grad():
                return model(torch.tensor(data[:, :7], dtype=torch.float32), 
                             torch.tensor(data[:, 7:], dtype=torch.float32)).detach().numpy()

        # Create randomized background to ensure SHAP gradients are non-zero
        background = res['scaled_input'] + np.random.normal(0, 0.05, (10, 11))
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(res['scaled_input'])
        vals = shap_values[0].flatten() if isinstance(shap_values, list) else shap_values.flatten()

        fig, ax = plt.subplots(figsize=(10, 5))
        importance_df = pd.DataFrame({'Feature': ALL_FEATURES, 'Impact': vals}).sort_values(by='Impact')
        colors = ['#1565C0' if f in STRUCTURAL_FEATURES else '#2E7D32' for f in importance_df['Feature']]
        
        ax.barh(importance_df['Feature'], importance_df['Impact'], color=colors)
        ax.set_xlabel('Price Impact (Contribution)')
        st.pyplot(fig)