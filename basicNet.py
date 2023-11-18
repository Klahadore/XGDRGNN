import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import pandas as pd

from torch_geometric.data import Data, download_url, extract_gz

url = 'http://snap.stanford.edu/biodata/datasets/10012/files/DG-AssocMiner_miner-disease-gene.tsv.gz'
extract_gz(download_url(url,'data'), 'data')

data_path = "data/DG-AssocMiner_miner-disease-gene.tsv"

df = pd.read_csv(data_path, sep="\t")
print(df.head(), '\n')

