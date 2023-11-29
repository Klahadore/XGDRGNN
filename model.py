import torch
from torch_geometric.nn import GATConv, Linear, to_hetero, GCNConv, GAE
from data import train_dataset, num_gene
import torch.nn.functional as F
from torch.optim import optimizer
from sklearn.metrics import roc_auc_score
from visualize import visualize_graph, visualize_emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# visualize_graph(train_dataset)
# visualize_graph(test_dataset)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, visualize=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # if visualize:
        #     visualize_emb(x)
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

# Assuming training is done
# model.eval()
# with torch.no_grad():
#     # You can pass in any dataset here (train_dataset or test_dataset)
#     model(train_dataset.x.to(device), train_dataset.edge_index.to(device), visualize=True)
# print model embeddings

with torch.no_grad():
    print(model.encode(train_dataset.x, train_dataset.edge_index))

visualize_emb(model.encode(train_dataset.x, train_dataset.edge_index).detach(), num_gene)

