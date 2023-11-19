import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import torch
from torch_geometric.nn import GAE, GCNConv, VGAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_FEATURES = 20
HIDDEN_SIZE = 200
OUT_CHANNELS = 20
EPOCHS = 40


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCNEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, out_channels, cached=True)
        self.droupout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.conv2(x,edge_index).relu()
        x_temp2 = self.droupout(x_temp1)
        return self.conv2(x_temp2, edge_index)


gae_model = GAE(GCNEncoder(NUM_FEATURES, HIDDEN_SIZE, OUT_CHANNELS, 0.5))
gae_model = gae_model.to(device)
