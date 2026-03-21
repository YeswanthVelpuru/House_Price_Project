import logging

class RLPriceAgent:
    def __init__(self, alert_threshold=0.15):
        self.threshold = alert_threshold

    def monitor_drift(self, predicted_unit_rate, live_market_rate):
        """
        Detects 'Model Drift' by comparing model predictions vs live market.
        Metric: Absolute Percentage Error.
        """
        drift = abs(predicted_unit_rate - live_market_rate) / live_market_rate
        
        if drift > self.threshold:
            status = f"⚠️ HIGH DRIFT ({drift:.2%})"
            logging.warning(f"Retraining Triggered: Model is off by {drift:.2%}")
        else:
            status = f"✅ STABLE ({drift:.2%})"
            
        return status

if __name__ == "__main__":
    agent = RLPriceAgent()
    # Example: Model predicts 8000/sqft, Market is 9500/sqft
    print(f"Monitoring Status: {agent.monitor_drift(8000, 9500)}")