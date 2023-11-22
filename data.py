from torch_geometric.data import Data, download_url, extract_zip
import networkx as nx
import json
from torch_geometric.data import HeteroData

from embeddings import *

TGBAURL = "https://zenodo.org/records/5911097/files/TBGA.zip?download=1"

extract_zip(download_url(TGBAURL, 'data'), 'data')

trainPath = "data/TBGA/TBGA_train.txt"
testPat = "data/TBGA/TBGA_test.txt"
valPath = "data/TBGA/TBGA_val.txt"

# returns mapping of gene id to a tuple containing disease id and relationship
def build_GDA_mapping(path):
    mapping = {}
    with open(path) as lines:
        for line in lines:
            data = json.loads(line)
            mapping[data["h"]["id"]] = (data["t"]["id"], data["relation"])

    return mapping

def build_graph(mapping):
    data = HeteroData()
    data[""]



print(build_GDA_mapping(valPath))
