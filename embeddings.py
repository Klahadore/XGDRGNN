import torch
import requests
import json
import pickle
from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import os
# import matplotlib.pyplot as plt
import time
import re
import wikipediaapi

# Generates embeddings for all genes in the dataset and reduces the dimensionality of the embeddings to 384
# Mapping is of GO ID to embedding
# When using must convert GO to MESH ID
# def generate_gene_embeddings():
#     model = torch.load("hig2vec_human_1000dim.pt", map_location="cpu")
#     objects, embeddings = model['objects'], model['embeddings']
#
#     pca = PCA(n_components=384)
#     pca.fit(embeddings)
#     embeddings = pca.transform(embeddings)
#
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings)
#
#     # Plot
#     if os.exists('data/gene_embeddings.pickle'):
#         return
#
#     with open('data/gene_embeddings.pickle', 'wb') as file:
#         pickle.dump(dict(zip(objects, embeddings)), file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
#
#
#


def gene_embedding(id):
    return torch.ones(384 , dtype=torch.float32)


def gene_embedding(id):
    return torch.ones(384, dtype=torch.float32)

    # print(embeddings)

# def wiki_summary(name):
#     url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
#
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             y = response.json()
#             return y['extract']
#         else:
#             print("Failed. Status code did not equal 200." + response.status_code)
#             print("wikipedia summary failed")
#
#             print(name)
#
#             return input("please input a definition here: ")
#     except requests.RequestException as e:
#         print(f"Request failed: {e}")
#         print("URL might not exist. Add your own definition?: ")
#         print(name)
#         return input("Add definiton: ")



#
# Example usage



counter = 0


def disease_and_symptom_embedding(id):
    if "MESH" in id:
        id = id[5:]
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    global counter
    print(counter)
    counter += 1
    with open('data/symptom_defs_wiki.pkl', 'rb') as fp:
        symptom_dictionary = pickle.load(fp)
    if id in symptom_dictionary:
        print("found symptom")
        return torch.tensor(model.encode(symptom_dictionary[id], show_progress_bar=False))

    with open('data/disease_definition_wiki.pkl', 'rb') as fp:
        disease_dictionary = pickle.load(fp)
    if id in disease_dictionary:
        print("found disease")
        return torch.tensor(model.encode(disease_dictionary[id], show_progress_bar=False))

    return torch.tensor(model.encode(input(f"didnt find embedding, inpput definition, {id}")))









def chemical_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def phe_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def mutation_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def pathway_embedding(id):
    return torch.ones(384, dtype=torch.float32)

