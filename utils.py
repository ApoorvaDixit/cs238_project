import re

def extract_tuples(input_string):
    tuple_pattern = re.compile(r'\(([^)]+)\)')
    matches = tuple_pattern.findall(input_string)
    tuples = [tuple(map(float, match.split(','))) for match in matches]
    return tuples

def generate_combinations(dimensions, current=[]):
    if not dimensions:
        return [tuple(current)]
    else:
        currents = []
        for value in range(-dimensions[0], dimensions[0]+1):
            currents.extend(generate_combinations(dimensions[1:], current + [value]))
        return currents

