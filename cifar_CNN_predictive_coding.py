import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import os
os.environ["OMP_NUM_THREADS"] = "4"




class PredictiveCodingCNN(nn.Module):
    def __init__(self, activation_type="relu"):
        super().__init__()

        self.num_layers = 5
        self.activation_type = activation_type

        # Define CNN layers
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Linear(8192, 512, bias=True),
            nn.Linear(512, 10, bias=True)
        ])
        
        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)



        self._set_activation()
        self.softmax = nn.Softmax(dim=1)
        self.E = 0  
        
        self.mu = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 3, 32, 32), requires_grad=False),  # Input
            nn.Parameter(torch.zeros(1, 64, 16, 16), requires_grad=True),   # After first CNN
            nn.Parameter(torch.zeros(1, 128, 8, 8), requires_grad=True),    # After second CNN
            nn.Parameter(torch.zeros(1, 512), requires_grad=True),          # After first Linear
            nn.Parameter(torch.zeros(1, 10), requires_grad=False)           # Output
        ])
        self.error = [torch.zeros_like(mu) for mu in self.mu]

    def _set_activation(self):
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
        self.mu[0].data = input.clone().detach().to(device)

    def clamp_output(self, output):
        self.mu[-1].data = output.clone().detach().to(device)

    def compute_error(self):
        self.error[1] = self.mu[1] - (self.layers[0](self.activation(self.mu[0])))
        self.error[2] = self.mu[2] - (self.layers[1](self.activation(self.mu[1])))
        flattened = self.mu[2].view(self.mu[2].size(0), -1)
        self.error[3] = self.mu[3] - (self.layers[2](self.activation(flattened)))
        self.error[4] = self.mu[4] - (self.layers[3](self.activation(self.mu[3])))
    
        batch_size = self.error[1].size(0)
        self.E = 200*torch.sum(torch.stack([torch.sum(error ** 2)/batch_size for error in self.error]))
        return self.E

    def label_pred(self, x):
        x = self.layers[0](self.activation(x))
        x = self.layers[1](self.activation(x))
        x = x.view(x.size(0), -1)
        x = self.layers[2](self.activation(x))
        x = self.layers[3](self.activation(x))
        return self.softmax(x)


def get_optimizer(parameters, optimizer_type, lr):
    if optimizer_type == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    elif optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' is not supported.")
    
def train_model(model, train_loader, T=60, device="cpu", weight_optimizer_type="sgd", mu_optimizer_type="sgd", learning_rate=0.001):
    total_energy = 0
    weight_optimizer = get_optimizer(
        (param for layer in model.layers for param in layer.parameters()),
        weight_optimizer_type, 
        lr=learning_rate
    )

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader)):
        input_pattern = data.to(device)
        target_pattern = torch.nn.functional.one_hot(target, num_classes=10).float().to(device)
        
        batch_size = data.size(0)
        model.mu = nn.ParameterList([
            nn.Parameter(torch.zeros(batch_size, 3, 32, 32).to(device), requires_grad=False),  # Input
            nn.Parameter(torch.zeros(batch_size, 64, 16, 16).to(device), requires_grad=True),   # After first CNN
            nn.Parameter(torch.zeros(batch_size, 128, 8, 8).to(device), requires_grad=True),    # After second CNN
            nn.Parameter(torch.zeros(batch_size, 512).to(device), requires_grad=True),          # After first Linear
            nn.Parameter(torch.zeros(batch_size, 10).to(device), requires_grad=False)           # Output
        ])
        model.error = [torch.zeros_like(mu) for mu in model.mu]
        model.clamp_input(input_pattern)
        model.clamp_output(target_pattern)

        mu_optimizer = get_optimizer(model.mu.parameters(), mu_optimizer_type, lr=0.1)

        for t in range(T):
            energy = model.compute_error()
            mu_optimizer.zero_grad()
            energy.backward()
            mu_optimizer.step()

        energy = model.compute_error()
        
        weight_optimizer.zero_grad()
        energy.backward()
        weight_optimizer.step()

        energy = model.compute_error()
        
        total_energy += energy
    
    del model.mu, model.error, mu_optimizer
    total_energy /= len(train_loader)  
    tqdm.write(f"Total Energy: {total_energy:.4f}")

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            input_pattern = data.to(device)
            target = target.to(device)
            output = model.label_pred(input_pattern)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # accuracy = 34% with num_T = 60, learning_rate = 0.000001, weight_optimizer_type = "adamw", mu_optimizer_type = "sgd"

    
    # Parameters
    batch_size = 128

    num_epochs = 20
    num_T = 60
    mu_optimizer_type = "sgd"
    weight_optimizer_type = "adamw"
    learning_rate = 0.000001

    activation_type = "relu"

    use_gpu = True

    # Data loaders
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PredictiveCodingCNN(activation_type=activation_type) 

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_model(model, train_loader, T=num_T, device=device, weight_optimizer_type=weight_optimizer_type, mu_optimizer_type=mu_optimizer_type, learning_rate=learning_rate)
        evaluate_model(model, test_loader, device)
