import streamlit as st
import torch
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
from model_training import HousePricePredictor
from rl_price_trend import RLPriceAgent
from market_features import scrape_market_trends

# Step 9: Deployment Configuration
st.set_page_config(page_title="Urban AI Valuation", layout="wide")

@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
    m = HousePricePredictor(12)
    m.load_state_dict(torch.load('house_price_model.pth'))
    m.eval()
    return sc, m

scaler, model = load_assets()
monitor = RLPriceAgent()

st.title("🏡 Fine-Grained Urban AI Predictor")

# Input simulation
with st.sidebar:
    st.header("Property Inputs")
    sqft = st.number_input("Sqft Living", value=2000)
    grade = st.slider("Grade", 1, 13, 7)
    city = st.selectbox("City", ["Delhi", "Mumbai", "Hyderabad", "Visakhapatnam"])

if st.button("Predict Market Value"):
    # Create raw input matching feature list
    raw_input = np.array([[3, 2, sqft, 5000, 1, 0, 0, 3, grade, 2015, 17.38, 78.48]])
    scaled_input = scaler.transform(raw_input)
    
    # Prediction
    pred = model(torch.tensor(scaled_input, dtype=torch.float32)).item()
    
    # Step 7: Explain (SHAP)
    st.subheader("🔍 Prediction Breakdown (SHAP)")
    # We use a kernel explainer for the PyTorch model
    explainer = shap.Explainer(lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(), scaled_input)
    shap_values = explainer(scaled_input)
    
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

    # Step 10: Monitoring
    live_rate = scrape_market_trends(city)
    drift = monitor.monitor_drift(pred/sqft, live_rate)
    st.metric("Valuation", f"₹{pred:,.0f}", f"Drift: {drift}")