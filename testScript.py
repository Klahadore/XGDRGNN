from data import new_train_dataset
import pickle
import torch
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, LinkNeighborLoader

torch.manual_seed(69)
print(new_train_dataset)

with open("data/train_dataset_metadata.pickle", "rb") as file:
    metadata = pickle.load(file)
    print("loaded metadata")

print(type(new_train_dataset))

loader = LinkNeighborLoader(
    new_train_dataset,
    num_neighbors=[15] * 2,
    batch_size=256,
    shuffle=True,
    edge_label=new_train_dataset.edge_label,
    edge_label_index=new_train_dataset.edge_label_index
)
print(loader)

batch_iterator = iter(loader)
first_batch = next(batch_iterator)

# Now you can debug or inspect the first batch
print(first_batch)
print(first_batch.edge_type)
print(first_batch.node_type)
print(len(loader))
print(metadata[0].index("gene"))
print(metadata[0].index("disease"))



