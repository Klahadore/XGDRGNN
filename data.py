import torch
from torch_geometric.data import Data, download_url, extract_zip
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
def add_index_to_map(key, value, mapping):
    if key in mapping:
        pass
    else:
        mapping[key] = value


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

    counter = 0
    with open(path) as lines:
        for line in lines:
            data = json.loads(line)
            add_index_to_map(data["h"]["id"], counter, gene_to_index)
            add_index_to_map(data["t"]["id"], counter, disease_to_index)
            counter += 1
    print(counter)

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
    edge_indices()

map1, map2 = build_index_mappings(VALPATH)
print(build_features(map1, gene_embedding, 20))
