import torch
from torch_geometric.nn import Linear, to_hetero, SAGEConv, summary

# from torch_geometric.data import HeteroData
# import torch.nn.functional as F
#
# from torch_geometric.utils import remove_self_loops, contains_self_loops
# from sklearn.metrics import roc_auc_score
# from visualize import visualize_graph, visualize_emb
# import torch_geometric.transforms as T

# from data import train_dataset, test_dataset, val_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
    Beta model using sage conv on 384 blank features. 
    Does not use edge attributes.
    Uses train_dataset which is a HeteroData object.
    Uses SageConv.
    
    Model Hyperperameters:
        hidden_channels = 20
        out_channels = 20
        lr = 0.01
        epochs = 200
        batch_size = 1
        loss = BCEWithLogitsLoss
        optimizer = Adam
        device = cuda if available else cpu
"""

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(20, hidden_channels)
        self.conv2 = SAGEConv(20, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['gene'][row], z_dict['disease'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class ModelOne(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GCN(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_dataset.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index), z_dict


if __name__ == "__main__":

    model = ModelOne(hidden_channels=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pred = None
    def train():
        model.train()
        optimizer.zero_grad()
        pred, z_dict = model.forward(train_dataset.x_dict, train_dataset.edge_index_dict, train_dataset['gene','gene_disease', 'disease'].edge_label_index)
        target = train_dataset['gene', 'gene_disease', 'disease'].edge_label
        loss = torch.nn.BCEWithLogitsLoss()(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)



    for epoch in range(200):
        print(train())


    torch.save(model.state_dict(), "cheese.pt")