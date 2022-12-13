"""
@file example_MNIST_ParallelMLP_FullHypernetwork.py

Example of using a Hypernetwork for a simple MNIST MLP example where the Hypernetwork is conditioned on each
input image and parallelized over a batch through the use of GroupConvolutions

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
test_batch_size = 64

# Training epochs (usually set to 10)
n_epochs = 200

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.99

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
                   'shuffle': True,
                   'drop_last': True}
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


class GroupSwish(nn.Module):
    """
    This class is simply reshaping the group convolution filters back into the batch in order
    to properly apply the Swish activation over the samples. On the pass back through, it is reshaped
    into the parallel version.
    """
    def __init__(self, groups):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5 for _ in range(groups)]))
        self.groups = groups

    def forward(self, x):
        n_ch_group = x.size(1) // self.groups
        t = x.shape[2:]
        x = x.reshape(-1, self.groups, n_ch_group, *t)
        beta = self.beta.view(1, self.groups, 1, *[1 for _ in t])
        return (x * torch.sigmoid_(x * F.softplus(beta))).div_(1.1).reshape(-1, self.groups * n_ch_group, *t)


# Define the architecture of the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Dimensions of the input, hidden sizes, and output
        x_dim = 784
        z_dim = 32

        # Array that holds dimensions over hidden layers
        # Given the simplistic network, here we just hardcode the layer node counts
        self.layers_dim = [x_dim, z_dim, num_classes]
        self.total_mainnet_params = reduce(lambda x, y: x*y, self.layers_dim)

        # Full hypernetwork example, tasks in the input sample x and outputs its networks weight
        self.hypernet = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_mainnet_params)
        )

        # This is the GroupConvolution network that handles multiple different MLPs in parallel while
        # maintaining that no overlap occurs in the weights. It uses the Convolution 'groups' tag to divide
        # the filters between layers. As we use a 1x1 kernel, 1 stride, and groups=batch_size with
        # filters=num_nodes*batch_size, we have each group relate to one individual MLP
        self.main_net = nn.Sequential(
            nn.Conv1d(x_dim * batch_size, z_dim * batch_size, 1, groups=batch_size, bias=True),
            GroupSwish(batch_size),
            nn.Conv1d(z_dim * batch_size, num_classes * batch_size, 1, groups=batch_size, bias=True),
        )

    def sample_weights(self, x):
        # Get the weight output of the hypernetwork
        weights = self.hypernet(x)

        # Split the output vector per layer
        next_idx = 0
        for i in range(len(self.layers_dim) - 1):
            # Get the current and next layer's neuron counts for the spice
            cur_idx = next_idx
            next_idx += self.layers_dim[i] * self.layers_dim[i + 1]

            # Get weight split and reshape to convolution filters
            # Note that here we reshape with the first dimension containing both batchsize and layers filters
            # given that we combine them in the convolution filters
            weight_splice = weights[:, cur_idx:next_idx].reshape(
                [batch_size * self.layers_dim[i + 1], self.layers_dim[i], 1]
            )

            # Copy over the generated weights into the parameters of the dynamics network
            del self.main_net[i * 2].weight
            self.main_net[i * 2].weight = weight_splice

    def forward(self, x):
        # Standard forward pass, just that in this case we reshape back to the batch setting after the CNN
        x = self.main_net(x)
        x = x.reshape([batch_size, -1])
        x = nn.Softmax()(x)
        return x


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get batch samples
        data, target = data.to(device), target.to(device)
        data = torch.flatten(data, 1)
        optimizer.zero_grad()

        # Sample weights from hypernet
        model.sample_weights(data)

        # Get model predictions
        # We reshape to [1, -1, 1] as we squish the batch into the filters dimension with a 1D convolution
        output = model(data.reshape([1, -1, 1]))

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
            data = torch.flatten(data, 1)

            # Sample weights from hypernet
            model.sample_weights(data)

            # Get model predictions
            output = model(data.reshape([1, -1, 1]))

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
