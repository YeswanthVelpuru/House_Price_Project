# rl_price_trend.py
import numpy as np

class RLPriceAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        # State: Demand levels (Low, Medium, High)
        # Actions: Adjustments (-5%, 0%, +5%)
        self.q_table = np.zeros((3, 3)) 
        self.lr = learning_rate
        self.gamma = discount_factor
        
    def _get_state(self, demand_index):
        if demand_index < 0.8: return 0 # Low
        if demand_index < 1.2: return 1 # Normal
        return 2 # High
        
    def adjust_price(self, base_price, demand_index):
        state = self._get_state(demand_index)
        
        # Choose action (Exploit Q-table)
        action_idx = np.argmax(self.q_table[state])
        
        adjustments = [-0.05, 0.0, 0.05]
        chosen_adj = adjustments[action_idx]
        
        final_price = base_price * (1.0 + chosen_adj)
        
        # Simulate reward (In a real system, this comes from user interactions/sales)
        reward = 1.0 if (demand_index > 1.0 and chosen_adj > 0) or (demand_index < 1.0 and chosen_adj < 0) else -1.0
        
        # Update Q-Table
        best_next_action = np.max(self.q_table[state])
        self.q_table[state, action_idx] = self.q_table[state, action_idx] + \
            self.lr * (reward + self.gamma * best_next_action - self.q_table[state, action_idx])
            
        return final_price