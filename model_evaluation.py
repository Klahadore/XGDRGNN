import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pickle
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.utils as utils

from alpha_model import Model, metadata

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/new_test_dataset.pickle", "rb") as file:
    test_data = pickle.load(file)
    print(test_data.edge_label[: test_data.edge_label_index.shape[1]].shape, "ter")
    # test_data.edge_index = utils.negative_sampling(test_data.edge_index)
    print(test_data.edge_index)
    print(test_data.edge_label[: test_data.edge_label_index.shape[1]].shape)

model = Model(384, training=False).cuda()
model.load_state_dict(torch.load("alphaModel_2_epochs.pt"))
model.eval()

data_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[25, 21],
    batch_size=256,
    shuffle=False,
    # edge_label=test_data.edge_label[: test_data.edge_label_index.shape[1]],
    edge_label_index=test_data.edge_label_index,
    neg_sampling="binary",
    num_workers=4,
)

all_probs = []
all_targets = []


with torch.no_grad():
    model.eval()

    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        batch_data.cuda()
        pred, _ = model(batch_data)

        probs = torch.sigmoid(pred).cpu().numpy()
        target = batch_data.edge_label[: len(pred)].cpu().numpy()
        print(target)
        print(roc_auc_score(target, probs))
