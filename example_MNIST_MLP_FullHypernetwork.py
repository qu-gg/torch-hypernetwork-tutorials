"""
@file example_MNIST_MLP_FullHypernetwork.py

Example of using a Hypernetwork for a simple MNIST MLP example where the Hypernetwork has a global
embedding vector E that is predefined and optimized alongside the Hypernetwork to solve MNIST.

This is a "Full" Hypernetwork, meaning that the the output of the Hypernetwork is the full weight vector
of the main network. This is the simplest version, however it scales poorly due to the parameter growth.

The base code for this MNIST example was found at @url{https://kirenz.github.io/deep-learning/docs/mnist-pytorch.html}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import reduce
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR

# Get CPU or GPU device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# Random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Batch sizes for training and testing
batch_size = 64
test_batch_size = 14

# Training epochs (usually set to 10)
n_epochs = 10

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.7

# How many batches before logging training status
log_interval = 100

# Number of target classes in the MNIST data
num_classes = 10

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}

# CUDA settings
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

# The scaled mean and standard deviation of the MNIST dataset (precalculated)
data_mean = 0.1307
data_std = 0.3081

# Convert input images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((data_mean,), (data_std,))
])

# Get the MNIST data from torchvision
dataset1 = datasets.MNIST('Data/mnist/', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('Data/mnist/', train=False, transform=transform)

# Define the data loaders that will handle fetching of data
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


# Define the architecture of the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Array that holds dimensions over hidden layers
        # Given the simplistic network, here we just hardcode the layer node counts
        self.layers_dim = [784, 64, 10]
        self.total_mainnet_params = reduce(lambda x, y: x*y, self.layers_dim)

        # Embedding vector, predefined and optimized as input to the hypernetwork
        # Note that this needs to be a Parameter to properly link it as getting a gradient
        self.embedding = nn.Parameter(torch.randn([1, 5], requires_grad=True).float().cuda())

        # This will NOT work for gradient updating
        # self.embedding = torch.randn([1, 5], requires_grad=True).float().cuda()

        # Full hypernetwork, takes embedding input and outputs weights of the main network
        self.hypernet = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_mainnet_params)
        )

        # Main network architecture that gets populated each time with hypernetwork output
        self.main_net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax()
        )

    def sample_weights(self):
        # Get the main net weight outputs from the hypernetwork
        weights = self.hypernet(self.embedding)[0]

        # Split the output vector per layer
        next_idx = 0
        for i in range(len(self.layers_dim) - 1):
            # Get the current and next layer's neuron counts for the spice
            cur_idx = next_idx
            next_idx += self.layers_dim[i] * self.layers_dim[i + 1]

            # Get the weight splice for these layers and shape to weight tensor
            weights_splice = weights[cur_idx:next_idx].reshape([self.layers_dim[i + 1], self.layers_dim[i]])

            # Copy over the generated weights into the parameters of the dynamics network
            # Note that this delete is important to properly establish the computation graph link
            del self.main_net[i * 2].weight
            self.main_net[i * 2].weight = weights_splice

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.main_net(x)


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get batch samples
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Sample weights from hypernet
        model.sample_weights()

        # Get model predictions
        output = model(data)

        # Build loss and update
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Get batch samples
            data, target = data.to(device), target.to(device)

            # Sample weights from hypernet
            model.sample_weights()

            # Get model predictions
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Embedding: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        model.embedding.data.detach().cpu().numpy()))


# Send the model to the device (CPU or GPU)
model = Net().to(device)

# Outputting a table of parameter counts to visualize the impact of architecture changes
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:20}: {reduce(lambda x, y: x*y, param.data.shape)}")

# Define the optimizer to user for gradient descent
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

# Shrinks the learning rate by gamma every step_size
scheduler = ExponentialLR(optimizer, gamma=gamma)

# Train the model
for epoch in range(1, n_epochs + 1):
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)
    scheduler.step()
