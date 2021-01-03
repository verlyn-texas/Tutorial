import random

def make_pairs(set_size, noise):
    x_values = []
    y_values = []
    
    for pair_idx in range(set_size):
        x = random.random() * 10 - 5 # x values over range of -5 to + 5
        y = x * x + noise*random.random() - noise / 2
        x_values.append(x)
        y_values.append(y)
    
    return x_values, y_values