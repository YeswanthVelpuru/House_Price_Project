import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class NeighborhoodGraph:
    def __init__(self, radius_km=1.0):
        self.radius = radius_km / 6371.0  # Earth's radius in km
        self.tree = None
        self.reference_data = None

    def fit(self, df):
        """Builds the spatial index using coordinates."""
        self.reference_data = df.copy()
        coords = np.deg2rad(self.reference_data[['lat', 'long']].values)
        self.tree = BallTree(coords, metric='haversine')
        print(f"Geospatial Graph Index built for {len(df)} properties.")

    def get_context(self, lat, lon):
        """Returns average price and quality of nearby houses."""
        query_coord = np.deg2rad([[lat, lon]])
        indices = self.tree.query_radius(query_coord, r=self.radius)[0]
        
        if len(indices) == 0:
            return {"neighbor_avg_price": 450000, "neighbor_avg_grade": 7}
        
        neighbors = self.reference_data.iloc[indices]
        return {
            "neighbor_avg_price": neighbors['price'].mean(),
            "neighbor_avg_grade": neighbors['grade'].mean()
        }

if __name__ == "__main__":
    # Test logic
    df_sample = pd.read_csv('kc_house_data.csv')
    graph = NeighborhoodGraph()
    graph.fit(df_sample)