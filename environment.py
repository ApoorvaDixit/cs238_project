import numpy as np
import random

from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict

from data_preprocessing import Data
from state import State
from utils import generate_combinations

class StockTradingEnv:
    def __init__(self, initial_balance: float, max_stocks: int, n_tickers: int, gamma: float, alpha: float, epsilon: float):
        """
        self.initial_balance: Amount of money that the model starts with. We assume that we have no stocks at the starting state.
        self.max_stocks: The maximum number of units of any stock we can buy or sell in an action.
        self.n_tickers: Number of stocks our model will trade on. 
        self.Q: Dictionary of (state, action) pairs to utility. Populated dynamically.
        """
        self.initial_balance = initial_balance
        self.max_stocks = max_stocks
        self.n_tickers = n_tickers
        self.possible_actions = generate_combinations([self.max_stocks]*self.n_tickers)
        self.Q = defaultdict(float)
        self.state_count = defaultdict(int)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon 

    def reset_for_rollout(self):
        self.Q = defaultdict(float)
        self.state_count = defaultdict(int)

    def get_next_action(self, s: State):

        if random.uniform(0, 1) > self.epsilon: ## select best action
            best_action = None
            best_Qval = -1e12
            for action in self.possible_actions:
                if s.is_valid_action(action):
                    if (s.get_tup(), action) not in self.Q: 
                        self.Q[(s.get_tup(),action)] = 0
                    if self.Q[(s.get_tup(),action)] > best_Qval:
                        best_action = action
            action = best_action

        else: ## select random action
            valid_actions = []
            for action in self.possible_actions:
                if s.is_valid_action(action):
                    valid_actions.append(action)
            action = random.sample(valid_actions, 1)[0]
        
        return action

    def get_next_state(self, s: State, a: tuple[int], prices: np.ndarray):
        # This must be a valid action.
        new_p = prices
        new_h = s.h + np.array(a)
        new_b = s.b*s.balance_bucket_size - np.sum(s.p*s.prices_bucket_size*np.array(a))
        return State(new_b, new_p, new_h, s.max_stocks)
            
    def update(self, s: State, a: tuple[int], r: float, s_prime: State):
        s_tuple = s.get_tup()
        s_prime_tuple = s_prime.get_tup()
        self.state_count[s_tuple] += 1
        x_hat = self.Q[(s_tuple, a)]
        next_action = self.get_next_action(s_prime)
        x_new = r + (self.gamma*self.Q[(s_prime_tuple, next_action)])
        # (6, 106, 59, 84.0, 0.0) (0, 0) 600
        # if s.b == 10000 and s.p[0] == 115 and s.p[1] == 65 and s.h[0] == 0 and s.h[1] == 0:
            # print("reward: " + str(r))
            # print("Q before update: " + str(self.Q[(s_tuple, a)]))
            # print("Next action Q: " + str(self.Q[(s_prime_tuple, next_action)]))
        self.Q[(s_tuple, a)] += (x_hat*(self.state_count[s_tuple] - 1) + x_new)/self.state_count[s_tuple]
        # if s.b == 10000 and s.p[0] == 115 and s.p[1] == 65 and s.h[0] == 0 and s.h[1] == 0:
            # print("Q after update: " + str(self.Q[(s_tuple, a)]))

# Parameters for the Q-learning algorithm
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
rollouts_per_date = 3000  # Number of epochs for training
start_dates = ['1/3/11', '1/4/11', '1/5/11', '1/6/11', '1/7/11']
date_format = '%m/%d/%y'
n_tickers = 2
initial_balance = 10000.0
max_stocks = 5
rollout_depth = 50

env = StockTradingEnv(initial_balance, max_stocks, n_tickers, gamma, alpha, epsilon)

filenames = ['archive/etfs/VOO.csv', 'archive/etfs/VTI.csv']
data = Data(filenames)

def rollout_helper(state, depth, date):
    if depth == 0:
        return
    else:
        action = env.get_next_action(state)
        new_date = date + timedelta(days=7)
        new_prices = np.zeros(n_tickers)
        idx = 0
        for ticker_dict in data.ticker_data:
            if new_date not in ticker_dict:
                new_prices[idx] = state.p[idx]*state.prices_bucket_size
            else:
                new_prices[idx] = ticker_dict[new_date]
            idx += 1 
        s_prime = env.get_next_state(state, action, new_prices)
        rollout_helper(s_prime, depth-1, new_date)
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        env.update(state, action, reward, s_prime)

if __name__ == '__main__':

    Q_vals = defaultdict(float)

    for start_date in tqdm(start_dates):
        init_state = None 
        init_date = datetime.strptime(start_date, date_format)
        init_b = initial_balance
        init_h = np.zeros(n_tickers)
        init_p = np.zeros(n_tickers)
        idx = 0
        for ticker_dict in data.ticker_data:
            init_p[idx] = ticker_dict[init_date]
            idx += 1 
        init_state = State(init_b, init_p, init_h, max_stocks)
        for rollout in tqdm(range(rollouts_per_date)):
            rollout_helper(init_state, rollout_depth, init_date)
            for k, v in env.Q.items():
                Q_vals[k] += v/rollouts_per_date
            env.reset_for_rollout()
            
    learned_q_vals = Q_vals
    # State -> (Best Action, Utility)
    best_actions = dict()
    utilities = dict()
    for k, v in learned_q_vals.items():
        s, a = k
        utilities[(s,a)] = (v,)
        if s in best_actions:
            if v > best_actions[s][1]:
                best_actions[s] = a, v
                #utilities[s] = (v,)
        else:
            best_actions[s] = a, v    
            #utilities[s] = (v,)
        
    f = open('q_eps_%s_rollouts_%s_depth_%s.policy' % (epsilon, rollouts_per_date, rollout_depth), "w")
    for state, action in best_actions.items():
        f.write('%s %s %s\n' % (state, action[0], action[1]))

    f.close()

    f = open('q_eps_%s_rollouts_%s_depth_%s.utilities' % (epsilon, rollouts_per_date, rollout_depth), "w")
    for state, utility in utilities.items():
        f.write('%s %s\n' % (state, utility))

    f.close()

## Q learning -> (state,action) + Q(s,a) -> logging all this
## kernel smoothin -> Q(s,a) -> sample a batch -> s_prime-> Q(s_sprime,action_k) -> PICK ACTION THAT MAXIMIZES Q