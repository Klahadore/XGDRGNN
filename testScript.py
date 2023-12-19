from data import train_dataset, new_train_dataset
import torch

# print(train_dataset.metadata()[1])
# print(train_dataset.edge_index_dict)
# new_dataset = train_dataset.to_homogeneous(edge_attrs=train_dataset.metadata()[1])
# print(new_dataset)
# print(new
# _dataset.edge_attr)
# new_dataset = train_dataset.to_homogeneous()
#
# print(train_dataset.generate_ids())
#
# print(new_dataset.edge_type)
#
#
#
#
#
# mapping = generate_edge_type_map(train_dataset.metadata())
#
# new_dataset.edge_attr = generate_new_edge_attr_tensor(mapping, new_dataset.edge_type, train_dataset)
# print(new_dataset)
# print(new_dataset.edge_attr)
#

print(new_train_dataset)
print(new_train_dataset.edge_attr)
print(new_train_dataset.edge_type)

