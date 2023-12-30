import torch.nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear, HeteroLinear


class SimpleHGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, metadata, edge_dim, concat=False, residual=False):
        super(SimpleHGATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        self.metadata = metadata
        self.edge_dim = edge_dim

        self.node_lin = HeteroLinear(in_channels, out_channels, len(metadata[0]))
        self.edge_lin = HeteroLinear(edge_dim, edge_dim, len(metadata[1]))

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

        edge_type_emb = self.edge_lin(edge_type_emb, edge_type)

        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb)

        # Concatenate or average the outputs from the different heads
        if self.concat:
            # Concatenate along the last dimension
            # print(out.shape)
            return out
        else:
            # Average the outputs
            out = out.view(len(out), self.out_channels, self.num_heads)
            return out.mean(dim=-1)

        # Add residual connection if needed
        # if self.residual:
        #     out = out + x



    def message(self, x_i, x_j, edge_type_emb, index):
        e_ij = torch.cat([x_i, x_j, edge_type_emb], dim=1)
        alpha = []
        for a in self.attention:
            alpha.append(a(e_ij))

        alpha = torch.cat(alpha, dim=-1)
        alpha = F.leaky_relu(alpha, .2)
        alpha = softmax(alpha, index)

        out = x_j * alpha[:, 0].view(-1, 1)
        out = out.unsqueeze(2)

        if self.num_heads == 1:
            return out

        for i in range(1, self.num_heads):
            out = torch.cat([out, (x_j * alpha[:, i].view(-1, 1)).unsqueeze(2)], dim=2)

        return out.reshape(len(out), -1)
