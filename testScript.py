# from data import new_train_dataset
import pickle
import torch
import requests
import torch_geometric
# from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, LinkNeighborLoader
import json
from alpha_model import Model
# print(torch_geometric.typing.WITH_PYG_LIB)
# torch.manual_seed(69)
# print(new_train_dataset)
#
# with open("data/train_dataset_metadata.pickle", "rb") as file:
#     metadata = pickle.load(file)
#     print("loaded metadata")
#
# print(type(new_train_dataset))
#
# loader = LinkNeighborLoader(
#     new_train_dataset,
#     num_neighbors=[15] * 2,
#     batch_size=256,
#     shuffle=True,
#     edge_label=new_train_dataset.edge_label,
#     edge_label_index=new_train_dataset.edge_label_index
# )
# print(loader)
#
# batch_iterator = iter(loader)
# first_batch = next(batch_iterator)
#
# # Now you can debug or inspect the first batch
# print(first_batch)
# print(first_batch.edge_type)
# print(first_batch.node_type)
# print(len(loader))
# print(metadata[0].index("gene"))
# print(metadata[0].index("disease"))
#
# x_j = torch.ones(100, 20).unsqueeze(-2)
# alpha = torch.zeros(100, 8)
#
# final = x_j * alpha.unsqueeze(-1)
# print(final.shape)
#
# print(final.view(-1, 8 * 20).shape)
# print(final.mean(1).shape)
#
# with open('data/connection_embedding.pkl', 'rb') as file:
#     edge_dict = pickle.load(file)
# print(edge_dict)
# print(edge_dict.keys())
#
# new_edge_dict = {}
# for i in edge_dict.keys():
#     new_edge_dict[i.lower()] = edge_dict[i]
#
# with open('data/new_connection_embedding.pkl', 'wb') as file:
#     pickle.dump(new_edge_dict, file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
#
# print(new_edge_dict.keys())



with open('data/new_test_dataset.pickle', 'rb') as file:
    new_test_dataset = pickle.load(file)
    

