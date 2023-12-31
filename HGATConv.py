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

        self.att = Linear(2 * in_channels + edge_dim, self.num_heads, bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        self.node_lin.reset_parameters()
        self.edge_lin.reset_parameters()

        self.att.reset_parameters()

    def forward(self, x, edge_index, node_type, edge_type_emb, edge_type):

        x = self.node_lin(x, node_type)

        edge_type_emb = self.edge_lin(edge_type_emb, edge_type)

        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb)

        print("i ran once")
        # Concatenate or average the outputs from the different heads
        if self.concat and self.residual:
            # Concatenate along the last dimension
            out = out + x.view(-1, 1, self.out_channels)
            return out.view(-1, self.out_channels * self.num_heads)
        elif self.concat and not self.residual:
            return out.view(-1, self.out_channels * self.num_heads)
        elif self.residual:
            # Average the outputs
            out = out.mean(dim=1)
            return out + x
        else:
            return out.mean(dim=1)

    def message(self, x_i, x_j, edge_type_emb, index):
        alpha = torch.cat([x_i, x_j, edge_type_emb], dim=-1)
        alpha = []

        alpha = self.att(alpha)
        alpha = F.leaky_relu(alpha, .2)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j.unsqueeze(-2) * alpha.unsqueeze(-1)

