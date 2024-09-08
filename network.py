import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer=None, dropout_prob=0, m=None):
        super().__init__()
        
        if hidden_layers == 2:
            # neurons = [m, m // 2]
            neurons = [neurons_per_layer, neurons_per_layer // 2]
        elif hidden_layers == 3:
            neurons = [m, m // 2, m // 4]
        elif hidden_layers == 4:
            neurons = [m, m // 2, m // 4, m // 4]
        else:
            neurons = [neurons_per_layer] * hidden_layers  # Default case, using neurons_per_layer if hidden_layers is not 2, 3, or 4

        layers = [nn.Linear(input_size, neurons[0]), nn.ReLU(), nn.Dropout(dropout_prob)]
        for i in range(1, hidden_layers):
            layers += [nn.Linear(neurons[i - 1], neurons[i]), nn.ReLU(), nn.Dropout(dropout_prob)]
        layers.append(nn.Linear(neurons[-1], output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
