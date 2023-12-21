import torch
import requests
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def gene_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def gene_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def convert_string(string):
    newlist = list(string.split(" "))
    return newlist

#input disease id, return embedding based off definiton from API of NCBI
def disease_and_symptom_embedding(id, path_to_file):
    pilot = ""
    if id[0] == "D":
        try:
            url = f'https://id.nlm.nih.gov/mesh/{id}.json'
            response = requests.get(url)
    
            if response.status_code == 200:
                y = json.loads(response.content)
                search = y["preferredConcept"]

                try:
                    url = search + ".json"
                    response = requests.get(url)
    
                    if response.status_code == 200:
                        y = json.loads(response.content)
                        cat = y["scopeNote"]["@value"]
                        pilot = y["label"]["@value"]
                    else:
                        print("Failed")
                        return None
                except requests.RequestException as g:
                    print(f"Request failed: {g}")
                    return None
    
            else:
                print("Failed")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
    else:
        try:
            url = f'https://id.nlm.nih.gov/mesh/{id}.json'
        
            response = requests.get(url)
        
            if response.status_code == 200:
                y = json.loads(response.content)
                cat = y["scopeNote"]["@value"]
        
            else:
                print("Failed")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    print(convert_string(cat))
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = model.encode(cat, show_progress_bar=False)
    print(embeddings)

    data_dictionary = {}

    with open(path_to_file, 'r') as file:
        lines = file.readlines()[1:]

        for line in lines:
            columns = line.strip().split('\t')
            key = columns[1]
            value = columns[0]
            if key in data_dictionary:
                if value not in data_dictionary[key]:
                    data_dictionary[key].append(value)
            else:
                data_dictionary[key] = [value]
    return embeddings, data_dictionary[pilot]

#downloads_folder = 'C:/Downloads'
#file_pathway = os.path.join(downloads_folder, '41467_2014_BFncomms5212_MOESM1045_ESM.txt')
print(disease_and_symptom_embedding("D012221", 'sypmtom_data.txt'))

def chemical_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def phe_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def mutation_embedding(id):
    return torch.ones(20, dtype=torch.float32)

def pathway_embedding(id):
    return torch.ones(20, dtype=torch.float32)