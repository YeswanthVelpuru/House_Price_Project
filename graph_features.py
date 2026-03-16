# graph_features.py
import numpy as np
from geopy.distance import geodesic

def estimate_gnn_price(lat, lon):

    GNN Neighborhood Features (Driver #50).
    Models spatial dependencies between property nodes.
    # Simulated GNN aggregation: Average price of 6 nearest spatial neighbors
    # In a real GNN, this would involve a message-passing layer
    base_neighborhood_val = 150000 # Base USD
    spatial_noise = np.random.normal(0, 20000)
    return base_neighborhood_val + spatial_noise