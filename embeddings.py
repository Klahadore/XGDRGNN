import torch
import requests
import json
import pickle
import unicodedata
from sentence_transformers import SentenceTransformer


def gene_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def disease_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def convert_string(string):
    newlist = list(string.split(" "))
    return newlist


def symptom_description(path_to_file):
    with open('data/symptom_data.pkl', 'rb') as fp:
        symptom_dictionary = pickle.load(fp)
    # symptom_dictionary = {}
    with open(path_to_file, 'r') as file:
        lines = file.readlines()[223:]  # was 1:
        for line in lines:  # for line in lines:
            columns = line.strip().split('\t')
            key = columns[3]  # was 6
            if key not in symptom_dictionary:
                # try:  Everything below this needs to be indented if try/except is to be used again FYI
                url = f'https://id.nlm.nih.gov/mesh/{key}.json'
                response = requests.get(url)
                if response.status_code == 200:
                    y = json.loads(response.content)
                    search = y["preferredConcept"]
                    try:
                        url = search + ".json"
                        response = requests.get(url)
                        y = json.loads(response.content)
                        cat = y["scopeNote"]["@value"]
                    except requests.RequestException as g:
                        print(response.status_code)
                        print(f"Request failed: {g}")
                        return None
                else:
                    print("Failed")
                    return None
                # except requests.RequestException as e:
                #     print(f"Request failed: {e}")
                #     return None
                symptom_dictionary[key] = [cat]
    with open('data/symptom_data.pkl', 'wb') as fp:
        pickle.dump(symptom_dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
        print('Dictionary saved successfully')
        print(symptom_dictionary)


# input disease id, return embedding based off definiton from API of NCBI
def disease_and_symptom_embedding(id):
    # if id[0] == "D":
    #     try:
    #         url = f'https://id.nlm.nih.gov/mesh/{id}.json'
    #         response = requests.get(url)
    #
    #         if response.status_code == 200:
    #             y = json.loads(response.content)
    #             search = y["preferredConcept"]
    #
    #             try:
    #                 url = search + ".json"
    #                 response = requests.get(url)
    #
    #                 if response.status_code == 200:
    #                     y = json.loads(response.content)
    #                     cat = y["scopeNote"]["@value"]
    #                     pilot = y["label"]["@value"]
    #                 else:
    #                     print("Failed")
    #                     return None
    #             except requests.RequestException as g:
    #                 print(f"Request failed: {g}")
    #                 return None
    #
    #         else:
    #             print("Failed")
    #             return None
    #     except requests.RequestException as e:
    #         print(f"Request failed: {e}")
    #         return None
    # else:
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
    # model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    # embeddings = model.encode(cat, show_progress_bar=False)
    # return embeddings
    return torch.ones(384, dtype=torch.float32)

    #
    # with open('symptom_data.pkl', 'rb') as fp:
    #     symptom_dictionary = pickle.load(fp)
    #     print('symptom_dictionary: ')
    #     print(symptom_dictionary)
    #
    # data_dictionary = {}
    #
    # with open(path_to_file, 'r') as file:
    #     lines = file.readlines()[1:]
    #     for line in lines:
    #         columns = line.strip().split('\t')
    #         key = columns[2]
    #         #value = columns[1]
    #         value = symptom_dictionary[columns[6]]
    #         if key in data_dictionary:
    #             if value not in data_dictionary[key]:
    #                 data_dictionary[key].append(value)
    #         else:
    #             data_dictionary[key] = [value]
    # return embeddings, data_dictionary[pilot]
    #


# print(symptom_description('Symptom-Occurence-Output.txt'))
# print(disease_and_symptom_embedding("D001835"))


def chemical_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def phe_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def mutation_embedding(id):
    return torch.ones(384, dtype=torch.float32)


def pathway_embedding(id):
    return torch.ones(384, dtype=torch.float32)
