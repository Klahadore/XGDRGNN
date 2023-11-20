
import torch_geometric.transforms as T
import torch
import pandas as pd
import networkx as nx
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt

from torch_geometric.data import Data, download_url, extract_gz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url = 'http://snap.stanford.edu/biodata/datasets/10012/files/DG-AssocMiner_miner-disease-gene.tsv.gz'
extract_gz(download_url(url, 'data'), 'data')

data_path = "data/DG-AssocMiner_miner-disease-gene.tsv"


# Maps each distinct node to a unique integer index.c
def load_node_mapping(datafile_path, index_col, offset=0):
    df = pd.read_csv(datafile_path, index_col=index_col, sep="\t")
    mapping = {index_id: i + offset for i, index_id in enumerate(df.index.unique())}
    return mapping


# Generates edge list in terms of node indices.
def load_edge_list(datafile_path, src_col, src_mapping, dst_col, dst_mapping):
    df = pd.read_csv(datafile_path, sep="\t")
    src_nodes = [src_mapping[index] for index in df[src_col]]
    dst_nodes = [dst_mapping[index] for index in df[dst_col]]
    edge_index = torch.tensor([src_nodes, dst_nodes])

    return edge_index


def initialize_data(datafile_path, num_features=20):
    dz_col, gene_col = "# Disease ID", "Gene ID"
    dz_mapping = load_node_mapping(datafile_path, dz_col, offset=0)
    gene_mapping = load_node_mapping(datafile_path, gene_col, offset=519)

    # edge indexes and reverse edge indexes
    edge_index = load_edge_list(datafile_path, dz_col, dz_mapping, gene_col, gene_mapping)
    rev_edge_index = load_edge_list(datafile_path, gene_col, gene_mapping, dz_col, dz_mapping)

    # Construct pytorch_geometric Data object
    data = Data()
    data.num_nodes = len(dz_mapping) + len(gene_mapping)
    data.edge_index = torch.cat((edge_index, rev_edge_index), dim=1)
    # initializes features as uniform ones
    data.x = torch.ones(data.num_nodes, num_features)

    return data, gene_mapping, dz_mapping

def visualize_graph(data, node_labels=None):
    """
    Visualize a PyTorch Geometric graph using NetworkX and Matplotlib.

    :param data: PyTorch Geometric Data object.
    :param node_labels: A dictionary mapping node indices to labels.
    """
    G = pyg_utils.to_networkx(data, to_undirected=True)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # positions for all nodes - can be customized

    # Draw the nodes and the edges (with optional labels)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="black")
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", alpha=0.9)

    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

    plt.title("Graph Visualization")
    plt.show()

# Assuming you have a way to create node labels from your data




# Read data and construct Data object.
data_object, gene_mapping, dz_mapping = initialize_data(data_path)
print(data_object)
print("Number of genes:", len(gene_mapping))
print("Number of diseases:", len(dz_mapping))

# Data Transformations
# Splits data into train, test, and eval
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True, split_labels=True,
                      add_negative_train_samples=True)
])

train_dataset, val_dataset, test_dataset = transform(data_object)

print("Train Data:\n", train_dataset)
print("Validation Data:\n", val_dataset)
print("Test Data:\n", test_dataset)
node_labels = {i: f'Label {i}' for i in range(train_dataset.num_nodes)}
visualize_graph(train_dataset, node_labels="")