"""
Evaluate learned policy on held-out test data.
"""
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

from data_preprocessing import Data
from state import State
from utils import extract_tuples
from environment import StockTradingEnv

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1  # Exploration rate
rollouts_per_date = 1000  # Number of epochs for training
start_dates = ['1/4/12', '1/5/12', '1/6/12', '1/9/12', '1/9/12']
date_format = '%m/%d/%y'
n_tickers = 2
initial_balance = 10000.0
max_stocks = 5
rollout_depth = 20

filenames = ['archive/etfs/VOO.csv', 'archive/etfs/VTI.csv']
data = Data(filenames)

env = StockTradingEnv(initial_balance, max_stocks, n_tickers, gamma, alpha, epsilon)

def load_policy(policy_filename):
    policy = {}
    with open(policy_filename, 'r') as f:
        for line in f.readlines():
            s, a = tuple(extract_tuples(line))
            policy[s] = a
    return policy
learnt_policy = load_policy('q_eps_0.1_rollouts_3000_depth_50.policy')
random_policy = {} ## empty dict will enforce calling random action each time

def rollout_helper(state, depth, date, policy):
    if depth == 0:
        return 0
    else:
        if state not in policy:
            action = env.get_next_action(state) ## random action, as epsilon is 0
            policy[state] = action
        else:
            action = policy[state]
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
        reward = s_prime.b - state.b + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy)
        return reward

def compute_mean_discounted_reward(policy):
    mean_discounted_reward = 0
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
        discounted_reward = 0
        for _ in range(rollouts_per_date):
            discounted_reward += rollout_helper(init_state, rollout_depth, init_date, policy)
        mean_discounted_reward += discounted_reward / rollouts_per_date
    mean_discounted_reward /= len(start_dates)
    return mean_discounted_reward

print('Mean Discounted Reward, rollout_depth %s, start_dates %s' % (rollout_depth, start_dates))
print('Learnt policy: %.2f' % (compute_mean_discounted_reward(learnt_policy)))
print('Random policy: %.2f' % (compute_mean_discounted_reward(random_policy)))
