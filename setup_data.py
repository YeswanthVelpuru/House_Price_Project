import pandas as pd
import os

def initialize_project():
    print("--- Phase 1: Problem Definition ---")
    print("Regression: Predicting house prices based on structural and geospatial features.")
    print("Primary Metric: Root Mean Squared Error (RMSE)")
    print("Secondary Metric: R-Squared (R2) Score")
    
    if os.path.exists('kc_house_data.csv'):
        df = pd.read_csv('kc_house_data.csv')
        print(f"Dataset Loaded: {len(df)} records available.")
    else:
        print("CRITICAL: kc_house_data.csv not found.")

if __name__ == "__main__":
    initialize_project()