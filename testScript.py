from data import train_dataset, new_train_dataset, dataset
import torch
import numpy
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
print(train_dataset)
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

#
# row, col = new_train_dataset.edge_label_index
# print(row)
# print(col)
#
# print(new_train_dataset.x[row].shape)
# print(new_train_dataset.x[col].shape)
#print(torch.cat([new_train_dataset.x[row], new_train_dataset.x[col]], dim=-1).shape)
# print(new_train_dataset)
# print(new_train_dataset.edge_label_index.shape)
# # print(new_train_dataset.edge_label.shape)
# print(train_dataset['gene', 'gene_disease', 'disease'].edge_label.shape)
# print(train_dataset['gene', 'gene_disease', 'disease'].edge_label_index.shape)
# print(train_dataset)
print(new_train_dataset)
print(new_train_dataset.node_type)
print(dataset.to_homogeneous().node_type)
num_positive_samples = (new_train_dataset.edge_label == 1).sum().item()
num_negative_samples = (new_train_dataset.edge_label == 0).sum().item()
print(num_negative_samples)
print(num_positive_samples)

# 21_289_950
# 29_147_123