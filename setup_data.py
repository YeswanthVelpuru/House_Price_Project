<<<<<<< HEAD
# setup_data.py
import os
import pandas as pd
import numpy as np

def init_project():
    print("Initializing project structure...")
    
    # Create required directories
    os.makedirs("image_cache", exist_ok=True)
    print("Created 'image_cache' directory.")
    
    # Create a mock kc_house_data.csv if it doesn't exist
    dataset_path = "kc_house_data.csv"
    if not os.path.exists(dataset_path):
        print(f"Generating sample dataset at {dataset_path}...")
        np.random.seed(42)
        n_samples = 500
        
        # Seattle area coordinates
        lats = np.random.uniform(47.5, 47.7, n_samples)
        longs = np.random.uniform(-122.4, -122.2, n_samples)
        
        # Base price calculation with some spatial logic
        prices = 500000 + (lats - 47.5) * 1000000 + np.random.normal(0, 50000, n_samples)
        
        df = pd.DataFrame({
            "id": range(1, n_samples + 1),
            "date": pd.date_range(start="2023-01-01", periods=n_samples).astype(str),
            "price": prices,
            "bedrooms": np.random.randint(1, 6, n_samples),
            "bathrooms": np.random.uniform(1, 4, n_samples),
            "sqft_living": np.random.randint(800, 5000, n_samples),
            "lat": lats,
            "long": longs
        })
        df.to_csv(dataset_path, index=False)
        print("Sample dataset created successfully!")
    else:
        print(f"Dataset {dataset_path} already exists. Skipping generation.")

if __name__ == "__main__":
=======
# setup_data.py
import os
import pandas as pd
import numpy as np

def init_project():
    print("Initializing project structure...")
    
    # Create required directories
    os.makedirs("image_cache", exist_ok=True)
    print("Created 'image_cache' directory.")
    
    # Create a mock kc_house_data.csv if it doesn't exist
    dataset_path = "kc_house_data.csv"
    if not os.path.exists(dataset_path):
        print(f"Generating sample dataset at {dataset_path}...")
        np.random.seed(42)
        n_samples = 500
        
        # Seattle area coordinates
        lats = np.random.uniform(47.5, 47.7, n_samples)
        longs = np.random.uniform(-122.4, -122.2, n_samples)
        
        # Base price calculation with some spatial logic
        prices = 500000 + (lats - 47.5) * 1000000 + np.random.normal(0, 50000, n_samples)
        
        df = pd.DataFrame({
            "id": range(1, n_samples + 1),
            "date": pd.date_range(start="2023-01-01", periods=n_samples).astype(str),
            "price": prices,
            "bedrooms": np.random.randint(1, 6, n_samples),
            "bathrooms": np.random.uniform(1, 4, n_samples),
            "sqft_living": np.random.randint(800, 5000, n_samples),
            "lat": lats,
            "long": longs
        })
        df.to_csv(dataset_path, index=False)
        print("Sample dataset created successfully!")
    else:
        print(f"Dataset {dataset_path} already exists. Skipping generation.")

if __name__ == "__main__":
>>>>>>> f1aeee6a5cef5e6f83d15321fd5985151c9c038d
    init_project()