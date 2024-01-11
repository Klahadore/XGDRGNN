import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import pickle
from torch_geometric.loader import LinkNeighborLoader

from visualize import visualize_graph, visualize_emb

from alpha_model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("data/new_test_dataset.pickle", 'rb') as file:
    test_data = pickle.load(file)

model = Model(384, training=False)
model.load_state_dict(torch.load('cheeese_epoch2_3.pt'))
model.eval()

data_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[25,21],
    batch_size=128,
    shuffle=False,
    edge_label=test_data.edge_label[:test_data.edge_label_index.shape[1]],
    edge_label_index=test_data.edge_label_index,
    num_workers=4
)

all_probs = []
all_targets = []


with torch.no_grad():
    model.eval()

    for batch_data in data_loader:
        batch_data = batch_data.to(device)

        pred, _ = model(batch_data)

        probs = torch.sigmoid(pred).cpu().numpy()

        all_probs.extend(probs[:,1])
        all_targets.extend(batch_data.edge_label[:len(pred)].cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    fpr, tpr, thresholds = roc_auc_score(all_probs, all_targets)






