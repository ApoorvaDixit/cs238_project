import re
'''
def extract_tuples(input_string):
    tuple_pattern = re.compile(r'\(([^)]+)\)')
    matches = tuple_pattern.findall(input_string)
    tuples = [tuple(map(float, match.split(','))) for match in matches]
    return tuples
'''

def extract_tuples(input_string):
    # Define a regular expression pattern to match content within parentheses
    tuple_pattern = re.compile(r'\(([^)]+)\)')
    # Find all matches of the pattern in the input string
    matches = tuple_pattern.findall(input_string)
    # Convert each match (string within parentheses) into a tuple of floats
    tuples = [tuple(map(float, filter(None, match.split(',')))) for match in matches]
    # Return the list of tuples
    return tuples

def generate_combinations(dimensions, current=[]):
    if not dimensions:
        return [tuple(current)]
    else:
        currents = []
        for value in range(-dimensions[0], dimensions[0]+1):
            currents.extend(generate_combinations(dimensions[1:], current + [value]))
        return currents

