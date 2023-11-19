"""
Load CSVs of chosen stocks, divide them by date into a train, validation, and test set. The data should be in a form that supports trajectory sampling.
"""
import csv

# Train: 2015
# Validation: 2016
# Test: 2018-2020

class Data:
	def __init__(self, filenames):
		"""
		Map of maps, <ticker_name, <date, opening_price>>
		"""
		self.ticker_data = dict()
		for file in filenames:
			# Extract ticker name 
			ticker_name = file.split('.')[0]
			ticker_name = ticker_name.split('/')[-1]
			self.ticker_data[ticker_name] = dict()
			pricereader = csv.DictReader(open(file))
			for row in pricereader:
				self.ticker_data[ticker_name][row['Date']] = row['Open'] 