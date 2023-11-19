import numpy as np
import random

from data_preprocessing import Data
from state import State
from utils import generate_combinations

class StockTradingEnv:
    def __init__(self, initial_balance: float, max_stocks: int, n_tickers: int, gamma: float, alpha: float, epsilon: float):
        """
        self.initial_balance: Amount of money that the model starts with. We assume that we have no stocks at the starting state.
        self.max_stocks: The maximum number of units of any stock we can own.
        self.n_tickers: Number of stocks our model will trade on. 
        self.actions_per_ticker: The number of actions we can take per ticker. For each ticker, we can hold, or buy/sell up to self.max_stocks units. 
        self.Q: Dictionary of (state, action) pairs to utility. Populated dynamically.
        """
        self.initial_balance = initial_balance
        self.max_stocks = max_stocks
        self.n_tickers = 2
        self.actions_per_ticker = 2*self.max_stocks + 1 
        self.possible_actions = generate_combinations([self.max_stocks]*self.n_tickers)
        self.Q = dict()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon 

    def get_next_action(self, s: State):

        if random.uniform(0, 1) >= self.epsilon: ## select best action
            best_action = None
            best_Qval = -1
            for action in self.possible_actions:
                if s.is_valid_action(action):
                    if self.Q[(s,action)] > best_Qval:
                        best_action = action
            action = best_action

        else: ## select random action
            valid_actions = []
            for action in self.possible_actions:
                if s.is_valid_action(action):
                    valid_actions.append(action)
            action = random.sample(valid_actions)
        
        return action
            
    def update(self, s: int, a: int, r: float, s_prime: int):
        # TODO: update
        self.Q[(s, a)] += self.alpha*(r + (self.gamma*self.Q[(s_prime, self.get_next_action(s_prime))]) - self.Q[(s, a)])
        
    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon


# Parameters for the Q-learning algorithm
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
epochs = 1000  # Number of epochs for training

filenames = ['archive/etfs/VOO.csv', 'archive/etfs/VTI.csv']
data = Data(filenames)

