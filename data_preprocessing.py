"""
Load CSVs of chosen stocks, divide them by date into a train, validation, and test set. The data should be in a form that supports trajectory sampling.
"""
import csv

from datetime import datetime

# Train: 2015-2017
# Validation: 2018
# Test: [2019 - ]

class Data:
	def __init__(self, filenames):
		"""
		List of maps, [<date, opening_price>]
		"""
		self.ticker_data = []		
		for file in filenames:
			# Extract ticker name 
			ticker_name = file.split('.')[0]
			ticker_name = ticker_name.split('/')[-1]
			self.ticker_data.append(dict())
			pricereader = csv.DictReader(open(file))
			for row in pricereader:
				start_date = row['Date']
				date_format = '%Y-%m-%d'
				date = datetime.strptime(start_date, date_format)
				self.ticker_data[-1][date] = row['Open'] 