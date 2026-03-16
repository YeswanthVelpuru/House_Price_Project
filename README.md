# Multimodal Geo-Spatial House Price Prediction Framework
**Team Lead:** Velpuru Yeswanth (ID: 232p1a3177)  
**Team Members:** Rakesh, Vamsi

## 🚀 Overview
A Multimodal Deep Learning framework that predicts residential property values by fusing Satellite Imagery (CNN), Neighborhood Graphs (GNN), and Tabular Data.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch (EfficientNet-B0 for Vision)
- **Explainability:** SHAP (50-Driver Taxonomy)
- **UI:** Streamlit
- **Geospatial:** Geopy, Mapbox API, Graph Neural Networks

## 📊 10 Base Criteria
1. Property Structural Features
2. Geospatial Intelligence
3. POI Density Features
4. Transportation Accessibility
5. Satellite Image Features
6. Street View Features
7. Crime & Safety Features
8. Socioeconomic Features
9. Spatial Clustering
10. GNN Neighborhood Features

## 🏃 How to Run
1. `pip install -r requirements.txt`
2. `python setup_data.py`
3. `python model_training.py`
4. `streamlit run app.py`
