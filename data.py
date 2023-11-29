import torch
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric import utils
import networkx as nx
import json
from torch_geometric.data import HeteroData
from collections import OrderedDict
from embeddings import *

TGBAURL = "https://zenodo.org/records/5911097/files/TBGA.zip?download=1"

extract_zip(download_url(TGBAURL, 'data'), 'data')

TRAINPATH = "data/TBGA/TBGA_train.txt"
TESTPATH = "data/TBGA/TBGA_test.txt"
VALPATH = "data/TBGA/TBGA_val.txt"


# helper function to assign unique ids to all elements because some elements might have multiple connections


# helper function to assign gene to an array of disease associations
def add_GDA_to_map(key, value, mapping):
    if key in mapping:
        if value in mapping[key]:
            pass
        else:
            mapping[key].append(value)
    else:
        mapping[key] = [value]


def build_index_mappings(path):
    """
    Returns mappings from NCBI Gene ID to index id, Disease to index
    Also returns NCBI Gene to Disease
    """
    gene_to_index = {}
    disease_to_index = {}

    with open(path) as lines:
        # First pass to index genes
        counter = 0
        for line in lines:
            data = json.loads(line)
            if data["h"]["id"] not in gene_to_index:
                gene_to_index[data["h"]["id"]] = counter
                counter += 1

        # Reset the read position of the file
        lines.seek(0)

        # Second pass to index diseases
        counter = 0
        for line in lines:
            data = json.loads(line)
            if data["t"]["id"] not in disease_to_index:
                disease_to_index[data["t"]["id"]] = counter
                counter += 1

    return gene_to_index, disease_to_index


def build_GDA_mapping(path):
    """
    builds NCBI gene to Disease mapping
    """
    gene_to_disease = {}
    with open(path) as lines:
        for line in lines:
            data = json.loads(line)
            add_GDA_to_map(data["h"]["id"], data["t"]["id"], gene_to_disease)
    # print(gene_to_disease[1])
    return gene_to_disease


def build_features(mapping, embedding_function, embedding_dim):
    """
    Returns a tensor of dimension: number of items in mapping X number of features.
    Utilizes embedding function to generate embeddings for each item in mapping.
    """
    x = torch.empty(0, embedding_dim, dtype=torch.float32)
    for key in mapping:
        # Generate embedding for the current key
        embedding = embedding_function(key).view(1, -1)

        # Concatenate the new embedding to the existing tensor x
        x = torch.cat((x, embedding), dim=0)
    return x


def build_edge_indices(gene_mapping, disease_mapping, GDA_mapping):
    """
    Builds edge indices tensor given gene mapping, disease mapping, and GDA mapping

    """
    edge_indices = torch.empty(2, 0, dtype=torch.int64)
    for key, value in GDA_mapping.items():
        for i in GDA_mapping[key]:
            column = torch.tensor([[gene_mapping[key]], [disease_mapping[i]]], dtype=torch.int64)
            edge_indices = torch.cat((edge_indices, column), dim=1)
    # print(edge_indices)
    return edge_indices


def make_dataset(path):
    gene_mapping, disease_mapping = build_index_mappings(path)

    num_disease = len(disease_mapping)
    num_gene = len(gene_mapping)

    # offset disease indices by number of genes
    for key in disease_mapping:
        disease_mapping[key] += num_gene

    GDA_mapping = build_GDA_mapping(path)

    x_gene = build_features(gene_mapping, gene_embedding, 20)
    x_disease = build_features(disease_mapping, disease_embedding, 20)

    # concatenate x_gene and x_disease tensors
    x = torch.cat((x_gene, x_disease), dim=0)

    GDA_edge_indices = build_edge_indices(gene_mapping, disease_mapping, GDA_mapping)

    # build pytorch geometric homogenous data object
    dataset = Data(x=x, edge_index=GDA_edge_indices, num_nodes=num_disease + num_gene)

    return dataset, num_gene


"""
    These are currently separate datasets, but we would like to combine them
    into a single dataset and then use masks to mask out different datasets for 
    training, testing, and evaluating
"""
train_dataset, num_gene = make_dataset(TRAINPATH)
# test_dataset, num_gene = make_dataset(TESTPATH)
# val_dataset = make_dataset(VALPATH)



# dataset = HeteroData()
#
# dataset["gene"].x = torch.cat([
#     train_dataset["gene"].x,
#     test_dataset["gene"].x,
#     val_dataset["gene"].x
# ], dim=0)
#
# dataset["disease"].x = torch.cat([
#     train_dataset["disease"].x,
#     test_dataset["disease"].x,
#     val_dataset["disease"].x
# ], dim=0)
#
# dataset["gene", "associated_with", "disease"] = torch.cat([
#     train_dataset["gene", "associated_with", "disease"].edge_index,
#     test_dataset["gene", "associated_with", "disease"].edge_index,
#     val_dataset["gene", "associated_with", "disease"].edge_index,
# ], dim=1)

# num_nodes_gene = dataset["gene"].num_nodes
# num_nodes_disease = dataset["disease"].num_nodes
# num_edges_gene_disease = dataset["gene"]["associated_with"]["disease"].num_edges

# train_mask = torch.zeros(num_nodes_disease + num_nodes_gene, dtype=torch.bool)
# test_mask = torch.zeros(num_nodes_disease + num_nodes_gene, dtype= torch.bool)
# val_mask = torch.zeros(num_nodes_disease + num_nodes_gene, dtype = torch.bool)

# print(dataset)
