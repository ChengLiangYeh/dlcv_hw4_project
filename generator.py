import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=1600, out_channels=1600, hidden_channels=3200):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

#G
'''
nn.Linear(in_channels, out_channels),
nn.ReLU(),
nn.Linear(out_channels, out_channels),
nn.ReLU(),
nn.Linear(out_channels, out_channels),
nn.ReLU(),
'''

#modifyG
'''
nn.Linear(in_channels, hidden_channels),
nn.ReLU(),
nn.Linear(hidden_channels, hidden_channels),
nn.ReLU(),
nn.Linear(hidden_channels, out_channels),
nn.ReLU(),
'''

#G2
'''
nn.Linear(in_channels, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 1600),
            nn.ReLU(),
'''
