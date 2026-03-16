# market_features.py
import random

def scrape_market_trends(seed=None):
    """
    Simulates real-time market data scraping. 
    In production, replace with APIs from Zillow, Redfin, or local police databases.
    """
    if seed is not None:
        random.seed(seed)
        
    return {
        "avg_price_today": random.uniform(400000, 900000),
        "demand_index": random.uniform(0.7, 1.4), # >1 means high demand
        "listing_count": random.randint(100, 2000),
        "crime_rate": random.uniform(0.05, 0.8),
        "safety_index": random.uniform(0.2, 1.0),
        "police_distance": random.uniform(0.1, 10.0) # km
    }