import torch
import requests
import json
from sentence_transformers import SentenceTransformer


def gene_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def gene_embedding(id):
    return torch.ones(20)

def convert_string(string):
    newlist = list(string.split(" "))
    return newlist

#input disease id, return embedding based off definiton from API of NCBI
def disease_embedding(id):
#     try:
#         url = f'https://id.nlm.nih.gov/mesh/{id}.json'
#
#         response = requests.get(url)
#
#         if response.status_code == 200:
#             y = json.loads(response.content)
#             cat = y["scopeNote"]["@value"]
#
#         else:
#             print("Failed")
#             return None
#     except requests.RequestException as e:
#         print(f"Request failed: {e}")
#         return None
#
#     print(convert_string(cat))
#     model = SentenceTransformer('BAAI/bge-small-en-v1.5')
#     embeddings = model.encode(cat, show_progress_bar=False)
#     print(embeddings)
    return torch.ones(20, dtype=torch.float32)
#
# print(disease_embedding("D011782"))

def chemical_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def phe_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def mutation_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def pathway_embedding(id):
    return torch.ones(20, dtype=torch.float32)

