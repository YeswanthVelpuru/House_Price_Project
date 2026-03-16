# market_features.py
import numpy as np

def scrape_market_trends():
    """
    Simulates real-time market data scraping.
    Provides the 'Demand Index' for the RL Agent to adjust pricing.
    """
    # In production, this would use BeautifulSoup or an API to check 
    # inventory levels and interest rates in the specific region.
    
    # Generate a demand index: 1.0 is stable, >1.0 is high demand, <1.0 is low
    demand_index = np.random.uniform(0.8, 1.3)
    
    # Mock data for local market sentiment
    market_data = {
        "demand_index": round(demand_index, 2),
        "inventory_status": "Low" if demand_index > 1.1 else "Normal",
        "avg_days_on_market": int(30 / demand_index)
    }
    
    return market_data