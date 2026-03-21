import numpy as np

def scrape_market_trends(city_name):
    """
    Simulates a live API call to fetch current 2026 market rates.
    Used for Phase 10: Monitoring.
    """
    # 2026 Urban Benchmarks (Avg rate per sqft)
    market_benchmarks = {
        "Delhi": 18500,
        "Mumbai": 34200,
        "Hyderabad": 8700,
        "Visakhapatnam": 7100
    }
    
    base_rate = market_benchmarks.get(city_name, 6000)
    # Add a random daily market fluctuation (+/- 3%)
    live_rate = base_rate * np.random.uniform(0.97, 1.03)
    
    return round(live_rate, 2)