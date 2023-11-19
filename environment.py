import numpy as np
import random

class StockTradingEnv:
    def __init__(self, initial_balance, max_stocks):
        """
        self.initial_balance: Amount of money that the model starts with. We assume that we have no stocks at the starting state.
        self.max_stocks: The maximum number of units of any stock we can own.
        self.n_tickers: Number of stocks our model will trade on. 
        self.actions_per_ticker: The number of actions we can take per ticker. For each ticker, we can hold, or buy/sell up to self.max_stocks units. 
        """
        self.initial_balance = initial_balance
        self.max_stocks = max_stocks
        self.n_tickers = 2
        self.actions_per_ticker = 2*self.max_stocks + 1 

# Parameters for the Q-learning algorithm
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
epochs = 1000  # Number of epochs for training