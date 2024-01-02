import json
import pickle
import requests

"""
    Makes a json map of diseases and their associated symptoms. 
    Symptoms are also diseases, but they are represented seperately as they are a different
    interaction than disease-disease. 
    
    Inputs: list of diseases 
    Outputs a file: disease_symptom.json in data/Gene_Disease_Network
    
"""
def tester(id):
    with open('disease_and_its_symptom_dictionary.pkl', 'rb') as fp:
        data_dictionary = pickle.load(fp)
        #print('data_dictionary: ')
        #print(data_dictionary)
    value = data_dictionary.get(id, None)
    if value == None:
        print(f"Did not find symptoms for disease {id}. Please manually input symptoms." )
        key = id
        value = input("Please input a symptom for this disease: ")
        data_dictionary[key] = None
        while value != "done":
            data_dictionary[key].append(value)
            value = input("Keep inputting symptoms or type 'done' exactly as shown to stop adding symptoms: ")
        with open('disease_and_its_symptom_dictionary.pkl', 'wb') as fp:
            pickle.dump(data_dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
            print('Dictionary saved successfully')
    return data_dictionary[id]
def name_resolver(name):
    value = requests.get(f"https://id.nlm.nih.gov/mesh/{name}.json").json()
    return value["label"]["@value"]
def list_of_symptoms(id):
    with open('disease_and_its_symptom_dictionary.pkl', 'rb') as fp:
        data_dictionary = pickle.load(fp)
        # print('symptom_dictionary: ')
        # print(data_dictionary)
    # value = data_dictionary.get(id, None)
    # if value == None:
    if id not in data_dictionary.keys():
        # print(f"Did not find symptoms for disease {id}. Please manually input symptoms.")
        # print("Type 'done' exactly as shown to stop adding symptoms")
        try:
            print(name_resolver(id))
        except:
            print("No name found")
        #
        # los = []
        # while True:
        #     value = input()
        #     if value == "":
        #         print("please enter in a disease or enter 'done' to stop adding symptoms")
        #         continue
        #     elif value == 'done':
        #         break
        #     else:
        #         los.append(value.strip())
        # # with open('disease_and_its_symptom_dictionary.pkl', 'wb') as fp:
        # #     pickle.dump(data_dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)
        # #     print('Dictionary saved successfully')
        return []
    else:
        print("symptoms found for " + id)
        return data_dictionary[id]

def build_file_mapping(filename):
    with open("data/Gene_Disease_Network/" + filename) as file:
        json_data = json.load(file)
    return json_data

def add_to_map_index(mapping, key):
    if key in mapping:
        pass
    else:
        los = list_of_symptoms(key[5:])
        mapping[key] = los
        print(len(mapping.keys()), len(los))

def build_index_map_of_keys(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        add_to_map_index(index_map, key)


def build_index_map_of_values(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        for value in json_data[key]:
            add_to_map_index(index_map, value)

def name_resolver(name):
    value = requests.get(f"https://id.nlm.nih.gov/mesh/{name}.json").json()
    return value["label"]["@value"]


def symptom_description(path_to_file):
    with open('data/symptom_data.pkl', 'rb') as fp:
        symptom_dictionary = pickle.load(fp)
        print(symptom_dictionary)
    # symptom_dictionary = {}
    with open(path_to_file, 'r') as file:
        lines = file.readlines()[223:]  # was 1:
        for line in lines:  # for line in lines:
            columns = line.strip().split('\t')
            key = columns[3]  # was 6
            if key not in symptom_dictionary:
                # try: Everything below this needs to be indented if try/except is to be used again FYI
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

if __name__ == "__main__":
    disease_symptom = {}

    build_index_map_of_keys("disease_disease.json", disease_symptom)
    build_index_map_of_values("disease_disease.json", disease_symptom)
    build_index_map_of_keys("disease_chemical.json", disease_symptom)
    build_index_map_of_keys("disease_gene.json", disease_symptom)
    build_index_map_of_keys("disease_mutation.json", disease_symptom)
    build_index_map_of_keys("disease_phe.json", disease_symptom)
    build_index_map_of_keys("disease_pathway.json", disease_symptom)
    build_index_map_of_values('gene_disease.json', disease_symptom)

    with open('data/Gene_Disease_Network/disease_symptom.json', 'w') as fp:
        json.dump(disease_symptom, fp)
        print('Dictionary saved successfully')
        print(len(disease_symptom.keys()))






