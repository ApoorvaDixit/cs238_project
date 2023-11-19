"""
Load CSVs of chosen stocks, divide them by date into a train, validation, and test set. The data should be in a form that supports trajectory sampling.
"""

filenames = ['archive/VOO.csv', 'archive/VTI.csv']

# Train: 2011 to 2016 (inclusive)
# Validation: 2017
# Test: 2018-2020