import torch

from model import ModelOne
from visualize import *
from data import val_dataset

model = ModelOne(20)
print(type(model))
state_dict = torch.load("cheese.pt")
# Set the model to evaluation mod
print(state_dict)

model.load_state_dict(state_dict)
print(type(model))

model.eval()
output, dict = model(val_dataset.x_dict, val_dataset.edge_index_dict, val_dataset["gene", "gene_disease", "disease"].edge_label_index)
# Assuming output is a dictionary containing embeddings for 'gene' and 'disease'
print(type(dict))
gene_embeddings = dict['gene']
disease_embeddings = dict['disease']

# Call the visualize_emb function with extracted embeddings
visualize_emb(gene_embeddings, disease_embeddings)


# https://github.com/pyg-team/pytorch_geometric/discussions/8422