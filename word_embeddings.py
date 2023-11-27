from torch_geometric.data import Dataloader, HeteroData, Dataset, Data
from transformers import AutoTokenizer, AutoModel
import torch
 

    def process_my_data(self, dataset_file_pointer):
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        model.eval()

        with open(dataset_file_pointer, 'r') as file:
            lines_read = file.readlines()
            words_read = [line.strip() for line in lines_read]
            
            new_embeddings = []
            for word in words_read:
                encoded_input = tokenizer(word, return_tensors = 'pt') # encoded_input = tokenizer(dataset_file_pointer, padding = True, truncation = True, return_tensors = 'pt')
                with torch.no_grad():   #model output
                    model_output = model(**encoded_input)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                new_embeddings.append(embeddings)


        #Pass in whole descriptions, return a vector of whatever comes out of model

        #immport dataset, turn sentence into array of words, turn array into vector

        #do all of this in the ebeddings.py area