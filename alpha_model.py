import torch

from visualize import visualize_emb
from data import new_train_dataset, train_dataset
from HGATConv import SimpleHGATConv
from torch_geometric.nn import Linear


"""
Single head attention only
no edge features
"""
class EdgeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.enc1 = SimpleHGATConv(hidden_channels, hidden_channels, 1, train_dataset.metadata(), 20, concat=True, residual=False)
        self.enc2 = SimpleHGATConv(hidden_channels, hidden_channels, 1, train_dataset.metadata(), 20, concat=True, residual=False)

    def forward(self, x, edge_index, node_type, edge_attr, edge_type):
        x = self.enc1(x, edge_index, node_type, edge_attr, edge_type)
        x = self.enc2(x, edge_index, node_type, edge_attr, edge_type)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_label_index):
        row, col = edge_label_index

        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = EdgeEncoder(hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x, edge_index, node_type, edge_attr, edge_type, edge_label_index):
        z = self.encoder(x, edge_index, node_type, edge_attr, edge_type)

        return self.decoder(z, edge_label_index), z


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    model = Model(20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    def train():
        model.train()
        optimizer.zero_grad()
        pred, z = model.forward(new_train_dataset.x, new_train_dataset.edge_index, new_train_dataset.node_type, new_train_dataset.edge_attr,
                             new_train_dataset.edge_type, new_train_dataset.edge_label_index)
        print(pred.shape, "pred")

        target = new_train_dataset.edge_label[:len(new_train_dataset.edge_label_index[0])]
        loss = torch.nn.BCEWithLogitsLoss()(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)


    for epoch in range(200):
        print(train())


torch.save(model.state_dict(), "cheese2.pt")


