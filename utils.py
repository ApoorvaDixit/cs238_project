def generate_combinations(dimensions, current=[]):
    if not dimensions:
        print(tuple(current))
    else:
        for value in range(-dimensions[0], dimensions[0]+1):
            generate_combinations(dimensions[1:], current + [value])

