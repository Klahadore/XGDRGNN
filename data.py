import torch
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric import utils
import networkx as nx
import json
from torch_geometric.data import HeteroData
from collections import OrderedDict
import pickle
import os
import sys

from embeddings import *

URL = "https://drive.google.com/drive/folders/1sfBElzOJA8RSrnUXE1AsxY8tzASakXIl"

# extract_zip(download_url(URL, 'data'), 'data')

FOLDER_PATH = "data/Gene_Disease_Network"

pickled_data_path = "data/pickled_data.pickle"

"""
    Check if pickled data exists and is not empty
"""
if os.path.exists(pickled_data_path) and os.path.getsize(pickled_data_path) > 0:
    try:
        with open(pickled_data_path, 'rb') as file:
            dataset = pickle.load(file)

        if dataset:
            print("Loaded pickled data")
            sys.exit(0)
        else:
            print("Pickled data is empty")
    except Exception as e:
        print(f"Failed to load pickled data due to an error: {e}")
else:
    print("Pickled data file does not exist or is empty")


def build_file_mapping(filename):
    with open(FOLDER_PATH + "/" + filename) as file:
        json_data = json.load(file)
    return json_data


def add_to_map_index(mapping, key, value):
    if key in mapping:
        pass
    else:
        mapping[key] = value


def build_index_map_of_keys(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        add_to_map_index(index_map, key, len(index_map))


def build_index_map_of_values(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        for value in json_data[key]:
            add_to_map_index(index_map, value, len(index_map))


"""
Builds node to index maps for each type of node
"""
gene_to_index = {}
disease_to_index = {}
chemical_to_index = {}
phe_to_index = {}
mutation_to_index = {}
pathway_to_index = {}

build_index_map_of_keys("gene_disease.json", gene_to_index)
build_index_map_of_keys("gene_mutation.json", gene_to_index)
build_index_map_of_keys("gene_phe.json", gene_to_index)
build_index_map_of_keys("gene_pathway.json", gene_to_index)
build_index_map_of_keys("gene_chemical.json", gene_to_index)
build_index_map_of_keys("gene_gene.json", gene_to_index)
build_index_map_of_values("gene_gene.json", gene_to_index)
build_index_map_of_values("disease_gene.json", gene_to_index)

build_index_map_of_keys("disease_gene.json", disease_to_index)
build_index_map_of_keys("disease_mutation.json", disease_to_index)
build_index_map_of_keys("disease_phe.json", disease_to_index)
build_index_map_of_keys("disease_pathway.json", disease_to_index)
build_index_map_of_keys("disease_chemical.json", disease_to_index)
build_index_map_of_keys("disease_disease.json", disease_to_index)
build_index_map_of_values("disease_disease.json", disease_to_index)
build_index_map_of_values("gene_disease.json", disease_to_index)

build_index_map_of_values("disease_chemical.json", chemical_to_index)
build_index_map_of_values("gene_chemical.json", chemical_to_index)

build_index_map_of_values("disease_phe.json", phe_to_index)
build_index_map_of_values("gene_phe.json", phe_to_index)

build_index_map_of_values("disease_mutation.json", mutation_to_index)
build_index_map_of_values("gene_mutation.json", mutation_to_index)

build_index_map_of_values("disease_pathway.json", pathway_to_index)
build_index_map_of_values("gene_pathway.json", pathway_to_index)


def create_node_embedding_tensor(index_map, embedding_function):
    tensor_list = []
    for key in index_map:
        tensor_list.append(embedding_function(key))
    return torch.stack(tensor_list)




def create_edge_indices(file_mapping, index_map_1, index_map_2):
    # Calculate the total number of elements to preallocate
    total_elements = sum(len(values) for values in file_mapping.values())

    # Preallocate the tensor
    edge_indices = torch.empty(2, total_elements, dtype=torch.int64)

    # Fill the tensor
    current_index = 0
    for key in file_mapping:
        for value in file_mapping[key]:
            edge_indices[:, current_index] = torch.tensor([index_map_1[key], index_map_2[value]], dtype=torch.int64)
            current_index += 1

    return edge_indices




dataset = HeteroData()

dataset['gene'].x = create_node_embedding_tensor(gene_to_index, gene_embedding)
dataset['disease'].x = create_node_embedding_tensor(disease_to_index, disease_embedding)
dataset['chemical'].x = create_node_embedding_tensor(chemical_to_index, chemical_embedding)
dataset['phe'].x = create_node_embedding_tensor(phe_to_index, phe_embedding)
dataset['mutation'].x = create_node_embedding_tensor(mutation_to_index, mutation_embedding)
dataset['pathway'].x = create_node_embedding_tensor(pathway_to_index, pathway_embedding)

dataset['gene', 'associated_with', 'disease'].edge_index = create_edge_indices(build_file_mapping("gene_disease.json"),
                                                                               gene_to_index, disease_to_index)
dataset['gene', 'associated_with', 'chemical'].edge_index = create_edge_indices(build_file_mapping("gene_chemical.json"),
                                                                                gene_to_index, chemical_to_index)
dataset['gene', 'associated_with', 'phe'].edge_index = create_edge_indices(build_file_mapping("gene_phe.json"),
                                                                            gene_to_index, phe_to_index)
dataset['gene', 'associated_with', 'mutation'].edge_index = create_edge_indices(build_file_mapping("gene_mutation.json"),
                                                                                gene_to_index, mutation_to_index)
dataset['gene', 'associated_with', 'pathway'].edge_index = create_edge_indices(build_file_mapping("gene_pathway.json"),
                                                                                gene_to_index, pathway_to_index)
dataset['gene', 'associated_with', 'gene'].edge_index = create_edge_indices(build_file_mapping("gene_gene.json"),
                                                                                 gene_to_index, gene_to_index)
dataset['disease', 'associated_with', 'chemical'].edge_index = create_edge_indices(build_file_mapping("disease_chemical.json"),
                                                                                    disease_to_index, chemical_to_index)
dataset['disease', 'associated_with', 'phe'].edge_index = create_edge_indices(build_file_mapping("disease_phe.json"),
                                                                                disease_to_index, phe_to_index)
dataset['disease', 'associated_with', 'mutation'].edge_index = create_edge_indices(build_file_mapping("disease_mutation.json"),
                                                                                    disease_to_index, mutation_to_index)
dataset['disease', 'associated_with', 'pathway'].edge_index = create_edge_indices(build_file_mapping("disease_pathway.json"),
                                                                                    disease_to_index, pathway_to_index)
dataset['disease', 'associated_with', 'disease'].edge_index = create_edge_indices(build_file_mapping("disease_disease.json"),
                                                                                  disease_to_index, disease_to_index)


print(dataset)

with open(pickled_data_path, 'wb') as file:
    pickle.dump(dataset, file)

print("Pickled data saved")