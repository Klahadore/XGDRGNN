import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

from visualize import visualize_graph, visualize_emb

from alpha_model import Model







new_val_dataset = None
with open("data/new_train_dataset.pickle", 'rb') as file:
    new_val_dataset = pickle.load(file)
    print("loaded val dataset")







@torch.no_grad()
def model_ROC_AUC(model_pred, model_label):
    model_pred = model_pred.cpu().numpy()
    model_label = model_label.cpu().numpy()
    return roc_auc_score(model_label, model_pred)



model = Model(384).cuda()

net = torch.load("cheese_epoch2_2.pt")
model.load_state_dict(net)

model.eval()

data_loader = LinkNeighborLoader(
    new_val_dataset,
    num_neighbors=[8,6],
    batch_size=384,
    edge_label=new_val_dataset.edge_label,
    edge_label_index=new_train_dataset.edge_label_index

)


with torch.no_grad():
    for batch in data_loader:
        batch = batch.cuda()
        pred, _ = model(batch)


print(pred)
print(type(model))



