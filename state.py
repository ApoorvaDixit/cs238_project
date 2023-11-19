import numpy as np

# should we use bucketing to create a discrete state space for Q learning?
class State:
	def __int__(self, b : float, p : np.ndarray, h = np.ndarray):
		"""
		self.b: Balance at the state.
		self.p: Numpy array of prices of all stickers being tracked, dtype = float.
		self.h: Numpy array of amount held of all stickers being tracked, dtype = int. We will have a limit on the maximum number of units of a single ticker we can hold. 
		This determines the size of the action space.
  		"""
		self.b = b 
		self.p = p 
		self.h = h


     

