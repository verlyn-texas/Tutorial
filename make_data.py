import random
import matplotlib.pyplot as plt
import torch

# My code
import custom_dataset
import pair_generator

def main():

    set_size = 100
    noise = 2

    x_values = []
    y_values = []
    x_values, y_values = pair_generator.make_pairs(set_size, noise)

    training_set_x = torch.tensor(x_values).float().unsqueeze(1)
    training_set_y = torch.tensor(y_values).float().unsqueeze(1).unsqueeze(1)

    dataset_train = custom_dataset.shared_task_ds(training_set_x, training_set_y)
    
    with open('training_set','wb') as training_file:
        torch.save(dataset_train, training_file)

    plt.scatter(x_values,y_values)
    plt.show()

main()
