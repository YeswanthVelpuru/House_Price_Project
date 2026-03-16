# rl_price_trend.py
import numpy as np

class RLPriceAgent:
    """
    Reinforcement Learning Agent that adjusts the base neural network 
    prediction based on real-time market volatility.
    """
    def __init__(self):
        # In a fully trained state, these would be Q-table values or weights
        self.learning_rate = 0.01

    def adjust_price(self, base_price, demand_index):
        """
        Action: Adjust price based on Demand 'State'.
        If demand is high (>1.1), the agent learns to nudge price up.
        """
        # Simple policy simulation for the demo
        adjustment_factor = 1.0
        if demand_index > 1.1:
            adjustment_factor = 1.05  # 5% Premium
        elif demand_index < 0.9:
            adjustment_factor = 0.95  # 5% Discount
            
        return base_price * adjustment_factor