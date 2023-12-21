import torch.nn
from torch_geometric.nn.conv import MessagePassing, HEATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear, HeteroLinear

from data import train_dataset, test_dataset, val_dataset, new_test_dataset, new_train_dataset, new_val_dataset


class SIUGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, metadata, edge_dim, concat=False, residual=False):
        super(SIUGATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        self.metadata = metadata
        self.edge_dim = edge_dim

        self.node_lin = HeteroLinear(in_channels, out_channels, len(metadata[0]))
        self.edge_lin = HeteroLinear(edge_dim, out_channels, len(metadata[1]))

        self.attention = torch.nn.ModuleList(
            [Linear(2 * in_channels + edge_dim, 1) for _ in range(num_heads)]
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.node_lin.reset_parameters()
        self.edge_lin.reset_parameters()
        for a in self.attention:
            a.reset_parameters()

    def forward(self, x, edge_index, node_type, edge_type_emb, edge_type):

        x = self.node_lin(x, node_type)
        print(edge_type_emb)
        edge_type_emb = self.edge_lin(edge_type_emb, edge_type)

        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb)

        # Concatenate or average the outputs from the different heads
        if self.concat:
            # Concatenate along the last dimension
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            # Average the outputs
            out = out.mean(dim=2)

        # Add residual connection if needed
        if self.residual:
            out = out + x
        print(out.shape, "out")
        return out

    def message(self, x_i, x_j, edge_type_emb, index):
        e_ij = torch.cat([x_i, x_j, edge_type_emb], dim=1)

        alpha = []
        for a in self.attention:
            alpha.append(a(e_ij))

        alpha = torch.cat(alpha, dim=-1)
        alpha = F.leaky_relu(alpha, .2)
        alpha = softmax(alpha, index)

        out = x_j * alpha
        return out


class testEdgeDecoder(torch.nn.Module):
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


class testModel(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, edge_emb_len):
        super().__init__()
        self.encoder = SIUGATConv(hidden_channels, hidden_channels, 1, metadata, 20, concat=True, residual=False)
        self.decoder = testEdgeDecoder(hidden_channels)

    def forward(self, x, edge_index, node_type, edge_attr, edge_type, edge_label_index):
        z = self.encoder(x, edge_index, node_type, edge_attr, edge_type)
        print(z.shape, "z")


        return self.decoder(z, edge_label_index), z


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    model = testModel(20, train_dataset.metadata(), 20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




    def train():
        model.train()
        optimizer.zero_grad()
        pred, z = model.forward(new_train_dataset.x, new_train_dataset.edge_index, new_train_dataset.node_type, new_train_dataset.edge_attr,
                             new_train_dataset.edge_type, new_train_dataset.edge_label_index)
        print(pred)
        target = new_train_dataset.edge_label
        loss = torch.nn.BCEWithLogitsLoss()(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)


    for epoch in range(200):
        print(train())

    torch.save(model.state_dict(), "cheese.pt")