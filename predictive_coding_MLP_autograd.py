import torch
from torchviz import make_dot

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import os
os.environ["OMP_NUM_THREADS"] = "4"

class PredictiveCoding(nn.Module):
    def __init__(
        self,
        layer_dims=[784, 100, 10],
        activation_type="sigmoid",
        bias=False,
    ):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(self.layer_dims)
        self.activation_type = activation_type
        self.bias = bias

        # Create linear layers using a ModuleList
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(self.layer_dims) - 1):
            layer = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1], bias=bias)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization
            self.layers.append(layer)


        self._set_activation()
        self.softmax = nn.Softmax(dim=1)
        self.E = 0  
        self.mu = nn.ParameterList([nn.Parameter(torch.zeros(1, dim), requires_grad=(i not in [0, len(layer_dims) - 1])) for i, dim in enumerate(layer_dims)])
        self.error = [torch.zeros(1, dim) for dim in layer_dims]


    def _set_activation(self):
        # Set activation function
        if self.activation_type.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.activation_type.lower() == "tanh":
            self.activation = nn.Tanh()
        elif self.activation_type.lower() == "relu":
            self.activation = nn.ReLU()
        elif self.activation_type.lower() == "identity":
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(f"Activation type '{self.activation_type}' is not recognized.")

    def clamp_input(self, input):
        # Explicitly set the first layer's mu to the input tensor
        self.mu[0].data = input.clone().detach().to(device)

    def clamp_output(self, output):
        # Explicitly set the last layer's mu to the output (target) tensor
        self.mu[-1].data = output.clone().detach().to(device)

    def compute_error(self):
        for l in range(1, len(self.mu)):
            layer_output = self.layers[l-1](self.activation((self.mu[l-1])))
            # variance = torch.var(self.mu[l], dim=0, keepdim=True)  # Compute variance
            self.error[l] = (self.mu[l] - layer_output) #/ torch.sqrt(variance + 1e-8)  # Add small value to avoid division by zero

        batch_size = self.mu[0].shape[0]
        self.E = torch.sum(torch.stack([torch.sum(0.5 * error ** 2)/batch_size for error in self.error]))
        return self.E

    def label_pred(self, x):
        # Forward pass through the network
        for i, layer in enumerate(self.layers):
            x = layer(self.activation(x))
        return self.softmax(x)


def get_optimizer(parameters, optimizer_type, lr):
    if optimizer_type == "sgd":
        return optim.SGD(parameters, lr=lr)
    elif optimizer_type == "adam":
        return optim.Adam(parameters, lr=lr)
    elif optimizer_type == "adamw":
        return optim.AdamW(parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' is not supported.")
    


def train_model(model, train_loader, T=3, device="cpu", weight_optimizer_type = "sgd", mu_optimizer_type = "sgd", learning_rate=0.003):

    total_energy = 0
    weight_optimizer = get_optimizer(
        (param for layer in model.layers for param in layer.parameters()), 
        weight_optimizer_type, 
        lr=learning_rate
    )

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader)):

        input_pattern = data.view(data.size(0), -1).to(device)
        target_pattern =  torch.nn.functional.one_hot(target, num_classes=10).float().to(device)


        batch_size = data.size(0)
        model.mu = nn.ParameterList([nn.Parameter(torch.zeros(batch_size, dim).to(device), requires_grad=(i not in [0, len(model.layer_dims) - 1])) for i, dim in enumerate(model.layer_dims)])
        model.error = [torch.zeros(batch_size, dim).to(device) for dim in model.layer_dims]
        model.clamp_input(input_pattern)
        model.clamp_output(target_pattern)


        mu_optimizer = get_optimizer(model.mu.parameters(),mu_optimizer_type,lr = 0.2)

        for t in range(T):
            energy = model.compute_error()
            # print(f"Batch {t}, Energy: {energy:.4f}")
            mu_optimizer.zero_grad()
            energy.backward()
            mu_optimizer.step()
        
        energy = model.compute_error()

        weight_optimizer.zero_grad()
        energy.backward()
        weight_optimizer.step()

        energy = model.compute_error()

        total_energy += energy
        
    total_energy /= len(train_loader)
    del model.mu, model.error, mu_optimizer  
    tqdm.write(f"Total Energy: {total_energy:.4f}")



def evaluate_model(model, test_loader, device="cpu"):
    model.eval()  # Set the model to evaluation model
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients for evaluation
        for data, target in test_loader:
            input_pattern = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            batch_size = data.size(0)
            output = model.label_pred(input_pattern)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')



if __name__ == "__main__":

    # accuracy: 95.5% weight_optimizer: adamw, num_T: 60, learning_rate: 0.001 

    # Parameters
    batch_size = 128

    layer_dims=[784, 100, 10]
    num_epochs = 20
    num_T = 60 # iteratioins of relaxation loop

    mu_optimizer_type = "sgd" # Options: "sgd", "adam", "adamw"
    weight_optimizer_type = "adamw" # Options: "sgd", "adam", "adamw"
    learning_rate = 0.001 # learning rate for weight optimizer, learning rate for mu optimizer is set to 0.1

    activation_type = "relu" # Options: "sigmoid", "tanh", "relu", "identity"
    bias = True

    use_gpu = False # might not be faster on GPU due to small network size


    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PredictiveCoding(layer_dims=layer_dims, activation_type=activation_type, bias=bias) 

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_model(model, train_loader, T = num_T, device = device, weight_optimizer_type=weight_optimizer_type, mu_optimizer_type = mu_optimizer_type, learning_rate = learning_rate)
        evaluate_model(model, test_loader, device)
