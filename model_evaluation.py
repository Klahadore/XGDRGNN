import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from visualize import visualize_graph, visualize_emb


@torch.no_grad()
def model_ROC_AUC(model_pred, model_label):
    model_pred = model_pred.cpu().numpy()
    model_label = model_label.cpu().numpy()
    return roc_auc_score(model_label, model_pred)



