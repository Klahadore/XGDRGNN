import torch
from torch_geometric.nn import GATConv, Linear, to_hetero, GCNConv, GAE
from data import train_dataset
import torch.nn.functional as F
from torch.optim import optimizer

# for model debug
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = train_dataset.to_homogeneous()
# train_dataset = Planetoid(root='/tmp/Cora', name='Cora')

print(train_dataset)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


hidden_channels, out_channels = train_dataset.num_features, 16

model = GAE(GCN(hidden_channels, out_channels)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_dataset.x, train_dataset.edge_index)
    loss = model.recon_loss(z, train_dataset.edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


for epoch in range(200):
    print(train())