import numpy as np

# Should we use bucketing to create a discrete state space for Q learning
class State:
	def __int__(self, b : float, p : np.ndarray, h : np.ndarray, max_stocks : int):
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
		self.balance_bucket_size = 50.0
		self.discretize()
		self.max_stocks = max_stocks
  
	def discretize(self):
		self.p = self.p/self.prices_bucket_size
		self.p = self.p.astype(int)
		self.b = int(self.b/self.prices_bucket_size)

	def get_valid_actions(self):
		"""
		Return a range of valid actions for each ticker. 
		How numbers map to actions:
		[1, max_tokens]: Buying 
		0: Holding all stocks
		[-1, -max_tokens]: Selling
		"""
		ranges = []
		for idx in range(len(self.h)):
			num_stocks = self.h[idx]
			max_action = self.max_stocks - num_stocks
			min_action = -1*num_stocks
			ranges.append((min_action, max_action))

		return ranges


     

