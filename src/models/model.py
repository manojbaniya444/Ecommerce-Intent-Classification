import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, X):
        out = self.relu(self.layer_1(X))
        out = self.relu(self.layer_2(out))
        out = self.layer_3(out)
        
        return out