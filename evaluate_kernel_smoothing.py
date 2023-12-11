"""
Evaluate learned policy on held-out test data.
"""
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
import random

from data_preprocessing import Data
from state import State
from utils import extract_tuples
from environment import StockTradingEnv

import pickle as pkl

np.random.seed(42)
random.seed(42)
# torch.manual_seed(42)

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1  # Exploration rate
rollouts_per_date = 100  # Number of epochs for training
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

def kernel(sk,s):
    #Inverse L1 loss between sk and s
    net_worth_sk = sk.b + np.sum(sk.p*sk.h)
    net_worth_s = s.b + np.sum(s.p*s.h)
    return 1.0/max(abs(net_worth_s - net_worth_sk),1e-5)
'''
def kernel_smoothing(basis,state,action,n):
    q_calc = 0
    normalize = 0
    #Sample N different states that have the same action as action
    sampled_basis = random.sample(basis[action],n)
    for s_basis,q_sa in sampled_basis:
        state_basis = State(s_basis[0],np.array(s_basis[1:3]),np.array(s_basis[3:5]),max_stocks)
        similarity_score = kernel(state_basis,state)
        normalize += similarity_score
        q_calc += similarity_score*q_sa
    
    return q_calc/normalize
'''

def kernel_smoothing(basis,state,action,n):
    q_calc = 0
    normalize = 0
    #Sample N different states that have the same action as action
    sampled_basis = random.sample(basis[action],n)
    states = [State(x[0],np.array(x[1:3]),np.array(x[3:5]),5) for x,q in sampled_basis]
    q_sa = np.array([q for x,q in sampled_basis])
    net_worth_sk= np.array([s.b + np.sum(s.p*s.h) for s in states])
    net_worth_s= state.b + np.sum(state.p*state.h)
    similarity_scores = 1.0/(np.maximum(np.abs(net_worth_s - net_worth_sk),1e-5))
    estimated_q = np.sum(similarity_scores * q_sa)/(np.sum(similarity_scores))
    
    return estimated_q
'''    
def rollout_helper(state, depth, date, policy,basis,is_random_policy):
    if depth == 0:
        return 0
    elif is_random_policy:
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        print(f"Random")
        return reward
    else:
        if state not in policy:
            #action = env.get_next_action(state) ## random action, as epsilon is 0
            #policy[state] = action
            maxq = -np.inf
            best_action = None
            for a in random.sample(env.possible_actions,100):
                q_val = kernel_smoothing(basis,state,a,1000)
                if q_val > maxq:
                    maxq = q_val
                    best_action = a
            action = best_action
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        print(f"Depth:{depth}\n")
        return reward
'''
'''   
def rollout_helper(state, depth, date, policy,basis,is_random_policy):
    if depth == 0:
        return 0
    elif is_random_policy:
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        return reward
    else:
        if state not in policy:
            #action = env.get_next_action(state) ## random action, as epsilon is 0
            #policy[state] = action
            np.random.seed(42)
            random_actions = [a for a in random.sample(env.possible_actions,50)]
            #Sample N different states that have the same action as action
            dtype = [('inner_tuple', [('int1', int), ('int2', int), ('int3', int), ('float1', float), ('float2', float)]), 
                    ('value', float)]
            sampled_basis = np.array([random.sample(basis[a],1000) for a in random_actions],dtype=dtype)
            states = np.array([[State(x[0],np.array(x[1],x[2]),np.array(x[3],x[4]),5) for x,q in basis_action]for basis_action in sampled_basis])
            q_sa = np.array([[q for x,q in basis_action]for basis_action in sampled_basis])
            net_worth_sk= np.array([[s.b + np.sum(s.p*s.h) for s in a]for a in states])
            net_worth_s= np.array(state.b + np.sum(state.p*state.h))
            similarity_scores = 1.0/(np.maximum(np.abs(net_worth_s - net_worth_sk),1e-5)) 
            #print(f"States shape {states.shape}")
            #print(f"Similarity scores shape: {similarity_scores.shape}")
            #print(f"Q_sa shape: {q_sa.shape}")
            estimated_q = np.argmax(np.sum((similarity_scores * q_sa)/(np.sum(similarity_scores)),axis=1))
            action = random_actions[estimated_q]
    
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        #print(f"Depth:{depth}\n")
        return reward
'''
def rollout_helper(state, depth, date, policy,basis,is_random_policy):
    if depth == 0:
        return 0
    elif is_random_policy:
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        return reward
    else:
        if state not in policy:
            #action = env.get_next_action(state) ## random action, as epsilon is 0
            #policy[state] = action
            num_samples=100
            num_actions = 100
            np.random.seed(42)
            random.seed(42)
            random_actions = [a for a in random.sample(env.possible_actions,num_actions)]
            #Sample N different states that have the same action as action
            dtype = [('inner_tuple', [('int1', int), ('int2', int), ('int3', int), ('float1', float), ('float2', float)]), 
                    ('value', float)]
            # sampled_basis = np.array([np.random.choice(basis[a],1000) for a in random_actions],dtype=dtype)
            
            sampled_basis = np.empty((num_actions, num_samples, 6))

            for i, action in enumerate(random_actions):
                basis_array = basis[action]
                indices = np.random.choice(basis_array.shape[0], num_samples, replace=False)
                sampled_basis[i] = basis_array[indices]

            states = sampled_basis[:,:,:5]
            q_sa = sampled_basis[:,:,5]
            net_worth_sk = np.squeeze(states[:,:,0]+ np.sum(states[:,:,1:3]*states[:,:,3:5],axis=-1))
            net_worth_s= np.array(state.b + np.sum(state.p*state.h))
            similarity_scores = 1.0/(np.maximum(np.abs(net_worth_s - net_worth_sk),1e-5)) 
            estimated_q = np.argmax(np.sum((similarity_scores * q_sa)/(np.sum(similarity_scores)),axis=1))
            action = random_actions[estimated_q]

    
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
        reward = (s_prime.b - state.b)*s_prime.balance_bucket_size + np.sum(s_prime.p*s_prime.prices_bucket_size*s_prime.h) - np.sum(state.p*state.prices_bucket_size*state.h)
        reward += gamma*rollout_helper(s_prime, depth-1, new_date, policy, basis, is_random_policy)
        #print(f"Depth:{depth}\n")
        return reward

def compute_mean_discounted_reward(basis,learnt_policy,is_random_policy):
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
        for i in range(rollouts_per_date):
            print(f"Rollout: {i}")
            discounted_reward += rollout_helper(init_state, rollout_depth, init_date, learnt_policy, basis, is_random_policy)
        mean_discounted_reward += discounted_reward / rollouts_per_date
    mean_discounted_reward /= len(start_dates)
    return mean_discounted_reward

learnt_policy = load_policy('q_eps_0.1_rollouts_3000_depth_50.policy')
#Need to rewrite load_theta in a different file
with open('basis_dict.pkl','rb') as f:
    basis = pkl.load(f)
with open('basis_dict_numpy.pkl','rb') as f:
    basis_np = pkl.load(f)
#Mixture of SARSA and Kernel smoothing
random_policy = {} ## empty dict will enforce calling random action each time
print('Mean Discounted Reward, rollout_depth %s, start_dates %s' % (rollout_depth, start_dates))
print('Learnt policy: %.2f' % (compute_mean_discounted_reward(basis_np, learnt_policy, False)))
print('Random policy: %.2f' % (compute_mean_discounted_reward(basis_np, random_policy, True)))
