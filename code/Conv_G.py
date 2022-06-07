import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class conv_g(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, m_augment=100, in_channels=1600, hidden_channels=1600, out_channels=1600):  # m_augment記得修改!!
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.m_augment = m_augment

        self.Generator = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        noise = torch.randn(x.shape[0]*self.m_augment, x.shape[1])
        noise = noise.cuda()
        cat_data = torch.cat((x, noise), 0)
        proto = self.Generator(cat_data)

        return proto
