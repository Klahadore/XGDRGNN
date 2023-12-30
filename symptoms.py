import json
import pickle
from embeddings import disease_and_symptom_embedding

"""
    Makes a json map of diseases and their associated symptoms. 
    Outputs a file: disease_symptom.json in data/Gene_Disease_Network
"""
if not os.path.exists('data/symptom_data.pkl'):



def build_json():
    with open('data/disease_symptom.json', 'w') as file:
        json.dump(symptom_dictionary, file)
