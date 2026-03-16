# data_processing.py
import os
import pickle
import numpy as np
import pandas as pd
import geohash2
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from image_features import get_image_features

ARTIFACTS_PATH = "artifacts.pkl"

def load_dataset(path="kc_house_data.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["sale_year"] = df["date"].dt.year.fillna(0).astype(int)
        df["sale_month"] = df["date"].dt.month.fillna(0).astype(int)
        df["sale_day"] = df["date"].dt.day.fillna(0).astype(int)
    return df

def add_geohash_features(df, precision=6):
    df["geohash"] = df.apply(
        lambda r: geohash2.encode(r["lat"], r["long"], precision=precision), axis=1)
    df["geohash_int"] = df["geohash"].apply(lambda x: int(x.encode().hex(), 16) % 100000)
    return df

def create_spatial_clusters(df, n_clusters=20):
    coords = df[["lat", "long"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    df["cluster_label"] = labels

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cluster_ohe = ohe.fit_transform(labels.reshape(-1, 1))
    cluster_cols = [f"cluster_{i}" for i in range(cluster_ohe.shape[1])]

    df_cluster = pd.DataFrame(cluster_ohe, columns=cluster_cols, index=df.index)
    df = pd.concat([df, df_cluster], axis=1)

    return df, kmeans, ohe, cluster_cols

def compute_distance_features(df):
    """Calculates distance to key Points of Interest (POIs)."""
    # Define primary POIs (Seattle context as default)
    pois = {
        "school": (47.6062, -122.3321),
        "hospital": (47.6205, -122.3493),
        "metro": (47.5998, -122.3343) # Metro distance feature
    }
    
    for poi_name, poi_coords in pois.items():
        distances = []
        for _, row in df.iterrows():
            loc = (row["lat"], row["long"])
            distances.append(geodesic(loc, poi_coords).km)
        df[f"dist_{poi_name}"] = distances
        
    return df

def _engineer_numeric_expansions(df, base_cols, expansions_per_col=6):
    """
    For each base col produce expansions:
      [x, x^2, x^3, log1p(x), sqrt(x_pos), sin(x)]
    Ensure deterministic, fill NaNs with 0.
    """
    out = []
    out_cols = []
    for c in base_cols:
        col = df[c].fillna(0).astype(float)
        a = col.values
        out.append(a)
        out_cols.append(f"{c}_x1")
        out.append(a ** 2)
        out_cols.append(f"{c}_x2")
        out.append(a ** 3)
        out_cols.append(f"{c}_x3")
        out.append(np.log1p(np.abs(a)))
        out_cols.append(f"{c}_log1p")
        out.append(np.sqrt(np.abs(a)))
        out_cols.append(f"{c}_sqrt")
        out.append(np.sin(a))
        out_cols.append(f"{c}_sin")
    arr = np.vstack(out).T 
    return arr, out_cols

def build_training_data(path="kc_house_data.csv", use_image_cache=True, limit=None):
    """
    Build full feature matrix including image features (2560 dims) + numeric engineered features.
    Saves artifacts.pkl with scaler, kmeans, ohe, numeric column metadata.
    """
    print("Loading dataset...")
    df = load_dataset(path)
    if limit:
        df = df.head(limit).copy()

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Adding geohash...")
    df = add_geohash_features(df)

    print("Creating spatial clusters and OHE...")
    df, kmeans, ohe, cluster_cols = create_spatial_clusters(df, n_clusters=20)

    print("Computing distance features...")
    df = compute_distance_features(df)

    target_col = "price"
    drop_cols = {"id", "date", "geohash", "cluster_label"}
    numeric_cols = [c for c in df.columns
                    if (c not in drop_cols and c != target_col and
                        pd.api.types.is_numeric_dtype(df[c]))]

    base_cols = numeric_cols[:12] if len(numeric_cols) >= 12 else numeric_cols
    print(f"Numeric base columns used for expansion (count {len(base_cols)}): {base_cols}")

    expanded_arr, expanded_cols = _engineer_numeric_expansions(df, base_cols)

    remaining_cols = [c for c in numeric_cols if c not in base_cols]
    remaining_arr = df[remaining_cols].fillna(0).astype(float).values if remaining_cols else np.zeros((len(df), 0))

    cluster_ohe_arr = df[cluster_cols].values

    numeric_block = np.hstack([expanded_arr, remaining_arr, cluster_ohe_arr])
    numeric_block_cols = expanded_cols + remaining_cols + cluster_cols

    print("Extracting image features (this may take time)...")
    sat_feats = []
    street_feats = []
    for idx, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["long"])
        sat = get_image_features(lat, lon, style="satellite-v9", use_cache=use_image_cache)
        street = get_image_features(lat, lon, style="streets-v11", use_cache=use_image_cache)
        if sat is None:
            sat = np.zeros(1280, dtype=np.float32)
        if street is None:
            street = np.zeros(1280, dtype=np.float32)
        sat_feats.append(sat)
        street_feats.append(street)

    sat_feats = np.vstack(sat_feats)  
    street_feats = np.vstack(street_feats)  
    image_block = np.hstack([sat_feats, street_feats])  

    X_full = np.hstack([image_block, numeric_block])

    print("Final feature shape:", X_full.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X = pd.DataFrame(X_scaled, columns=[f"img_{i}" for i in range(image_block.shape[1])] + numeric_block_cols)

    y = df[target_col].copy()

    artifacts = {
        "kmeans": kmeans,
        "ohe": ohe,
        "cluster_cols": cluster_cols,
        "base_cols": base_cols,
        "remaining_cols": remaining_cols,
        "numeric_block_cols": numeric_block_cols,
        "scaler": scaler,
        "image_dim": image_block.shape[1],
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Artifacts saved to {ARTIFACTS_PATH}")

    return X, y, scaler