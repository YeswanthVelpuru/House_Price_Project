import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_processed_data():
    """
    Phase 2 & 4: Splitting & Feature Engineering
    Returns: X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, feature_names
    """
    print("--- Phase 2 & 4: Processing Data ---")
    df = pd.read_csv('kc_house_data.csv')
    
    # 7 Structural Features + 4 Geospatial Features = 11 Total
    structural_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'condition', 'yr_built']
    geospatial_cols = ['lat', 'long', 'sqft_living15', 'sqft_lot15']
    features = structural_cols + geospatial_cols
    
    X = df[features]
    y = df['price']

    # Step 2: Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Step 4: Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Save scaler for app.py
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, features

if __name__ == "__main__":
    get_processed_data()