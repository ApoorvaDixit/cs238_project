def generate_combinations(dimensions, current=[]):
    if not dimensions:
        return [tuple(current)]
    else:
        currents = []
        for value in range(-dimensions[0], dimensions[0]+1):
            currents.extend(generate_combinations(dimensions[1:], current + [value]))
        return currents

