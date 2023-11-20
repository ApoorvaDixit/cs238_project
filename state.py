import numpy as np

# Should we use bucketing to create a discrete state space for Q learning
class State:
	def __init__(self, b : float, p : np.ndarray, h : np.ndarray, max_stocks : int):
		"""
		self.b: Balance at the state.
		self.p: Numpy array of prices of all stickers being tracked, dtype = float.
		self.h: Numpy array of amount held of all stickers being tracked, dtype = int. We will have a limit on the maximum number of units of a single ticker we can hold. 
		This determines the size of the action space.
  		"""
		self.b = b
		self.p = p
		self.h = h
		self.prices_bucket_size = 1.0
		self.balance_bucket_size = 5.0
		self.discretize()
		self.max_stocks = max_stocks
  
	def discretize(self):
		self.p = self.p/self.prices_bucket_size
		self.p = self.p.astype(int)
		self.b = int(self.b/self.prices_bucket_size)

	def is_valid_action(self, action):
		
		## check 1: enough stocks
		for i in range(len(self.h)):
			if self.h[i] + action[i] < 0: return False

		## check 2: new balance is non-negative
		new_balance = self.b*self.balance_bucket_size
		for i in range(len(self.h)):
			new_balance -= action[i] * self.p[i]*self.prices_bucket_size
		if new_balance < 0: return False

		return True

	def get_tup(self):
		ids = [self.b] + [p for p in self.p] + [h for h in self.h]
		return tuple(ids)

