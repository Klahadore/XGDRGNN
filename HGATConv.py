import torch.nn
from torch import Tensor
from torch_geometric import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear, HeteroDictLinear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
import torch.nn.functional as F
from data import train_dataset, test_dataset, val_dataset



class Simple_HGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, metadata, concat=False, residual=False):
        super(Simple_HGATConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        # converts metadata into a list of edge types that are strings instead of tuples
        # this is because the HeteroData object requires edge types to be strings
        self.edge_types = []
        for edge_type in metadata[1]:
            self.edge_types.append(",".join(edge_type))
        # lin edge must have its own list of types
        self.lin_edge = HeteroDictLinear(in_channels, in_channels, self.edge_types)
        self.lin_node = HeteroDictLinear(in_channels, in_channels, metadata[0])

        self.attention = torch.nn.ModuleList(
            [Linear(2 * in_channels, 1) for _ in range(num_heads)]
        )

    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        for a in self.attention:
            a.reset_parameters()

    def forward(self, x_node, edge_index_dict, edge_type_emb, return_attention_weights=False):

        # This is done because the HeteroData object requires edge types to be strings,
        # just as the edge types are in the metadata
        print(edge_index_dict)
        edge_type_emb_with_string_keys = {}
        for edge_type in edge_type_emb.keys():
            edge_type_emb_with_string_keys[",".join(edge_type)] = edge_type_emb[edge_type]
        print(edge_type_emb_with_string_keys.keys())
        # print(self.edge_types)
        x_node = self.lin_node(x_node)
        x_edge = self.lin_edge(edge_type_emb)


        out_dict = {}
        for node_type in x_node.keys():
            out_dict[node_type] = []

        # Propagate over every edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            out = self.propagate(edge_index, x=(x_node[src_type], x_node[dst_type]), edge_type_emb=x_edge[(src_type, dst_type)], size=None)

            out_dict[dst_type].append(F.relu(out))

        # Concatenate all the outputs

        return out_dict

    def message(self, x_i, x_j, edge_type_emb, index):
        e_ij = torch.cat([x_i, x_j, edge_type_emb], dim=1)
        alpha = []

        for i in range(self.num_heads):
            a = self.attention[i](e_ij)
            alpha.append(a)

        alpha = torch.cat(alpha, dim=-1)

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)

        out = x_i * alpha
        return out


class testEdgeDecoder(torch.nn.Module):
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


class testModel(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = Simple_HGATConv(hidden_channels, hidden_channels, 1, metadata)
        self.decoder = testEdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_type_emb):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_type_emb)
        return self.decoder(z_dict, edge_label_index), z_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    model = testModel(20, train_dataset.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pred = None


    def train():
        model.train()
        optimizer.zero_grad()
        pred, z_dict = model.forward(train_dataset.x_dict, train_dataset.edge_index_dict,
                                     train_dataset['gene', 'gene_disease', 'disease'].edge_label_index, train_dataset.edge_attr_dict)
        target = train_dataset['gene', 'gene_disease', 'disease'].edge_label
        loss = torch.nn.BCEWithLogitsLoss()(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss)


    for epoch in range(200):
        print(train())

    torch.save(model.state_dict(), "cheese.pt")





