import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# My code
import custom_dataset
import pair_generator

class Net(nn.Module):
    def __init__(self, hidden_one):
        super(Net, self).__init__()
        input_size = 1
        self.hidden_one = hidden_one
        output_size = 1

        self.fc1 = nn.Linear(input_size, hidden_one)
        self.fc2 = nn.Linear(hidden_one, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # shape [batch,1,1]; shape [batch,1,1]
        optimizer.zero_grad()
        output = model(data) # shape [batch,1,1]
        criteria = nn.MSELoss()
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

def test(model, device):
    
    set_size = 100
    noise = 2

    x_values = []
    y_values = []
    x_values, y_values = pair_generator.make_pairs(set_size, noise)

    test_set_x = torch.tensor(x_values).float().unsqueeze(1)
    test_set_y = torch.tensor(y_values).float().unsqueeze(1).unsqueeze(1)

    dataset_test = custom_dataset.shared_task_ds(test_set_x, test_set_y)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, drop_last=True, shuffle=True)
    
    model.eval()

    actual_x = []
    actual_y = []
    predicted_y = []

    for batch_idx, (data, target) in enumerate(test_loader):
        x = data.item()
        y = target.item()
        y_pred = model(data).item()
        actual_x.append(x)
        actual_y.append(y)
        predicted_y.append(y_pred)

    return actual_x, actual_y, predicted_y


def main():
    # Hyperparameters
    learning_rate = 0.05
    epochs = 50
    batch_size = 16
    hidden_nodes = 40

    # Load in Training Data
    filename = 'training_set'
    infile = open(filename,'rb')
    dataset = torch.load(infile)

    # Create Learning Objects
    device = 'cpu'
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    model = Net(hidden_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learn
    losses = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, losses)

    # Plot Losses
    plt.plot(losses)
    plt.title('Losses')
    plt.show()

    # Perform Test

    actual_x, actual_y, predicted_y = test(model, device)

    plt.title = 'Test Data'
    plt.scatter(actual_x,actual_y)
    plt.scatter(actual_x,predicted_y)
    plt.legend(['Actual','Predicted'])
    plt.show()


main()

