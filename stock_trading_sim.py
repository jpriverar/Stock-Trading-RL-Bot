import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class StockTradingSimEnvironment:
    def __init__(self, data, initial_investment):
        # Pointer to the data point we are currently in
        self.pointer = 0

        # Saving the initial investment for reset and to the current budget
        self.initial_investment = initial_investment
        self.budget = initial_investment

        # Saving the stock market historical data
        self.data = data
        self.n_steps, self.n_stocks = self.data.shape

        # Getting the action space based in the dataset
        # Possible actions for each stock ('buy', 'sell', 'hold')
        actions = ["buy", "sell", "hold"]
        self.action_space = np.arange(len(actions)**self.n_stocks)
        
        # Mapping from the action space to the visual representation with cartesian product
        self.action_list = np.asarray(list(itertools.product(actions, repeat=self.n_stocks)))

        # To store how many stock shares we currently have
        self.stock_shares = np.zeros(self.n_stocks)

        # State observation size
        self.state_size = self.n_stocks*2 + 1

    def reset(self):
        # Resets all the variables and returns the initial state
        self.pointer = 0
        self.budget = self.initial_investment
        self.stock_shares = np.zeros(self.n_stocks)

        # [share_value, n_shares, budget]
        return self.__current_state()

    def __stock_prices(self):
        return self.data.iloc[self.pointer].values

    def __current_state(self):
        return np.concatenate((self.stock_shares, self.__stock_prices(), self.budget), axis=None)

    def __get_portfolio(self):
        return np.matmul(self.__stock_prices(), self.stock_shares) + self.budget

    def step(self, action):
        # Performs the given action in the environment
        # Map the integer action to the action list
        action = self.action_list[action]
        # Our current portfolio to compute the reward later
        current_portfolio = self.__get_portfolio()

        # Current stock market prices
        prices = self.__stock_prices()

        # First selling all the stock shares
        self.budget += np.matmul(prices[action == "sell"], self.stock_shares[action == "sell"])
        self.stock_shares[action == "sell"] = 0

        # The buying as many stock shares as possible
        buying = True
        while buying:
            buying = False
            for i, stock_action in enumerate(action):
                # If we want to buy for that stock and we have enough money, then buy it
                if stock_action == "buy" and self.budget >= prices[i]:
                    self.stock_shares[i] += 1
                    self.budget -= prices[i]
                    # Set the buying flag to true for we are still buying
                    buying = True

        # If we want to hold, then there's nothing to do
        # Finally increase the data pointer and and compute next step info
        self.pointer += 1
        next_portfolio = self.__get_portfolio()
        next_state = self.__current_state()
        reward = next_portfolio - current_portfolio # Change in portfolio
        done = self.pointer >= (self.n_steps-1) # Done if we reached the last time step
        info = {"portfolio_value":next_portfolio} # Other additional information we might want to add

        return next_state, reward, done, info