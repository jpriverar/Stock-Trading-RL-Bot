from stock_trading_model import StockTradingModel
import numpy as np

class StockTradingAgent:
    def __init__(self, state_features, output_actions, samples, gamma=0.9):
        self.model = StockTradingModel(state_features, output_actions, samples)
        self.gamma = gamma

    def get_action(self, state):
        # Returns the best action known so far by the agent
        action_values = self.model.predict(state)
        return np.argmax(action_values)

    def train(self, state, action, next_state, reward, done):
        # Updates the model weights based on the environment observation
        # Checking what's the target
        if done:
            target = reward
        else:
            target = reward + self.gamma*max(self.model.predict(next_state))

        # Updating the model based on the input and target
        self.model.fit(state, action, target)