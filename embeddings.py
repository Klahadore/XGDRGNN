import torch
import requests
import json
import pickle
from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import os
# import matplotlib.pyplot as plt


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

# if cant find embeddings of disease, ask user for input and input an embedding of the disease. Dont stop running.
# check if scopenote is blank, ask user for definition.      Done
# input disease id or symptom id, return embedding based off definiton from API of NCBI

    # print(embeddings)




def disease_and_symptom_embedding(id):
    extract = ""
    with open('symptom_data.pkl', 'rb') as fp:
        symptom_dictionary = pickle.load(fp)
    if id in symptom_dictionary:
        extract = symptom_dictionary[id]
    else:
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
                        dog = y["label"]["@value"].lower()
                        extract = None
                        extract = y["scopeNote"]["@value"]
                        if extract == None:
                            extract = input(
                                "Scopenote does not exist. Please input your own definition here if you want to: ")
                    else:
                        print("Failed. Status code did not equal 200.")
                        return None
                except requests.RequestException as g:
                    print(f"Request failed: {g}")
                    print("URL might not exist. Add your own definition?: ")
                    extract = input("Add definiton: ")

                other_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{dog}'
                response = requests.get(other_url)
                if response.status_code == 200:
                    y = response.json()
                    extract = y.get("extract")
                    print("It worked")
                else:
                    print("Failed")
                    print(dog)
                    extract = input("Please input a definition here: ")
            else:
                print("Failed. Status code did not equal 200.")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            print("URL might not exist. Add your own definition?: ")
            print(id)
            extract = input("Add definiton: ")

    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    # embeddings = model.encode(cat, show_progress_bar=False)
    embeddings_number_two = model.encode(extract, show_progress_bar=False)
    # print(embeddings)
    return (embeddings_number_two)


# Return empty list and print to console 'did not find symptoms and return disease id '

# in lsit of symptoms function, if you cant find symptoms for disease, do same thing: ask for prompt

def input_disease_output_symptoms(id, path_to_file):
    data_dictionary = {}
    with open(path_to_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            columns = line.strip().split('\t')
            key = columns[5]
            value = columns[6]
            if key in data_dictionary:
                if value not in data_dictionary[key]:
                    data_dictionary[key].append(value)
            else:
                data_dictionary[key] = [value]
    with open('disease_and_its_symptom_dictionary.pkl', 'wb') as fp:
        pickle.dump(data_dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
        print('Dictionary saved successfully')
        print(data_dictionary)
    # return data_dictionary






# print(symptom_description('Symptom-Occurence-Output.txt'))
# print(disease_and_symptom_embedding("D001835"))
# print(input_disease_output_symptoms("D012221", 'symptom_data.txt'))



def chemical_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def phe_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def mutation_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def pathway_embedding(id):
    return torch.ones(384, dtype=torch.float32)

