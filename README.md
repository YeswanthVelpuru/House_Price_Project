# 🏡 Multi-Modal Geospatial House Price Prediction AI
### *A Fine-Grained Urban Valuation System using the 10-Phase ML Lifecycle*

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B.svg)

This project implements a state-of-the-art house price prediction engine that integrates **Tabular Data**, **Computer Vision (Architectural Features)**, and **Geospatial Graphs (Neighborhood Intelligence)**. It follows a rigorous professional machine learning lifecycle from problem definition to real-time monitoring.
## 🚀 The 10-Phase ML Lifecycle

This repository is structured to demonstrate the complete end-to-end pipeline required in production AI environments:
1.  **Define Problem & Metric**: Targeted 2026 urban house price regression using **RMSE** and **R² Score** as primary success metrics (`setup_data.py`).
2.  **Split Train/Val/Test**: Implemented a 70/15/15 deterministic split to ensure model generalization and prevent data leakage (`data_processing.py`).
3.  **Baseline**: Established a performance floor using a **Random Forest Regressor** before moving to Deep Learning (`model_training.py`).
4.  **Feature Engineering**:
    * **Vision**: Extraction of architectural features using **ResNet-18** (`image_features.py`).
    * **Graphs**: Neighborhood context via **BallTree Haversine** spatial indexing (`graph-features.py`).
5.  **Train DL/ML**: A multi-branch Neural Network built in **PyTorch** designed for multi-modal fusion (`model_training.py`).
6.  **Evaluate**: Comprehensive testing against unseen data, tracking MSE and error distribution (`model_training.py`).
7.  **Explain (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** to provide feature-level transparency for every prediction (`app.py`).
8.  **Package**: Model serialized via **ONNX** for cross-platform high-performance inference (`house_price_model.onnx`).
9.  **Deploy**: Interactive dashboard built with **Streamlit** for real-time geospatial valuation (`app.py`).
10. **Monitor**: Built-in **Drift Detection** agent that compares AI predictions against live 2026 market trends (`rl_price_trend.py`).
