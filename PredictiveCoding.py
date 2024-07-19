import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm



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
            self.layer_norms.append(nn.LayerNorm(self.layer_dims[i]))  # Add LayerNorm
            layer = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1], bias=bias)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization
            self.layers.append(layer)


        self._set_activation()
        self.softmax = nn.Softmax(dim=1)
        self.E = 0  # Initialize total energy

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
        self.mu[0].data = input.clone().detach()

    def clamp_output(self, output):
        # Explicitly set the last layer's mu to the output (target) tensor
        self.mu[-1].data = output.clone().detach()

    def compute_error(self):
        for l in range(1, len(self.mu)):
            # layer_output = self.layers[l-1](self.activation(self.layer_norms[l-1](self.mu[l-1])))
            layer_output = self.layers[l-1](self.activation((self.mu[l-1])))
            # variance = torch.var(self.mu[l], dim=0, keepdim=True)  # Compute variance
            self.error[l] = (self.mu[l] - layer_output) #/ torch.sqrt(variance + 1e-8)  # Add small value to avoid division by zero

        return sum(torch.mean(error ** 2).item() for error in self.error)

    def label_pred(self, x):
        # Forward pass through the network
        for i, layer in enumerate(self.layers):
            # x = self.layer_norms[i](x)  # Apply LayerNorm
            x = self.activation(x)
            x = layer(x)
        return self.softmax(x)

    def update_energy(self):
        # Compute and update the total energy based on current errors
        self.E = sum(torch.mean(error ** 2).item() for error in self.error)


    def _compute_activation_derivative(self, x):
        # Compute derivative of the activation function
        if self.activation_type.lower() == "sigmoid":
            return x * (1 - x)
        elif self.activation_type.lower() == "tanh":
            return 1 - x.pow(2)
        elif self.activation_type.lower() == "relu":
            return (x > 0).float()
        elif self.activation_type.lower() == "identity":
            return torch.ones_like(x)

def train_model(model, train_loader, T=3):
    total_energy = 0
    weight_optimizer = optim.SGD((param for layer in model.layers for param in layer.parameters()), lr=learning_rate)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader)):

      input_pattern = data.view(data.size(0), -1)
      target_pattern =  torch.nn.functional.one_hot(target, num_classes=10).float()


      batch_size = data.size(0)
      model.mu = nn.ParameterList([nn.Parameter(torch.zeros(batch_size, dim), requires_grad=(i not in [0, len(model.layer_dims) - 1])) for i, dim in enumerate(model.layer_dims)])
      model.error = [torch.zeros(batch_size, dim) for dim in model.layer_dims]
      model.clamp_input(input_pattern)
      model.clamp_output(target_pattern)


      mu_optimizer = optim.SGD(model.mu.parameters(), lr=learning_rate*1000)



      for t in range(T):
          error = model.compute_error()



          mu_optimizer.zero_grad()
          for l in range(1, model.num_layers-1):
              activation_derivative = model._compute_activation_derivative(model.mu[l])
              grad_x = -model.error[l] + activation_derivative * torch.mm(model.error[l+1], model.layers[l].weight)
              model.mu[l].grad = - grad_x
          mu_optimizer.step()

      # break
      error = model.compute_error()

      weight_optimizer.zero_grad()
      for l in range(0, model.num_layers-1):
          grad_w = torch.mm(model.error[l+1].T, model.activation(model.mu[l]))
          model.layers[l].weight.grad = - grad_w
          if model.bias:
            grad_b = model.error[l+1].T.sum(dim=1)
            model.layers[l].bias.grad = - grad_b
      weight_optimizer.step()


      model.update_energy()
      # print("w:", model.E)
      total_energy += model.E
    tqdm.write(f"Total Energy: {total_energy:.4f}")


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation model
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients for evaluation
        for data, target in test_loader:
            input_pattern = data.view(data.size(0), -1)
            batch_size = data.size(0)
            output = model.label_pred(input_pattern)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# Parameters
batch_size = 128
learning_rate = 0.0001
num_epochs = 100
num_T = 50

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = PredictiveCoding(layer_dims=[784, 100, 10], activation_type="relu", bias=False)



for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_model(model, train_loader, T = num_T)
    evaluate_model(model, test_loader)
