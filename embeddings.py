import torch
from torch_geometric.data import Data, download_url, extract_gz

DISEASE_DATASET_URL = "https://snap.stanford.edu/biodata/datasets/10021/files/D-DoMiner_miner-diseaseDOID.tsv.gz"

#1
def gene_embedding(id):
    return torch.ones(20)

def disease_embedding(id):
    return torch.ones(20, dtype=torch.float32)
