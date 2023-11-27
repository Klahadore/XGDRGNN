import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

def gene_embedding(id):
    return torch.ones(20)

def disease_embedding(id):
    SNAP_dataset = "D-DoMiner_miner-diseaseDOID.tsv.gz"
    extract_gz(download_url(SNAP_dataset, 'data'), 'data')

    PATH_TO_SNAP = "data/D-DoMiner_miner-diseaseDOID.tsv"

    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    sentences_tot = []
    with open(PATH_TO_SNAP, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) >= 3:
                id_value = columns[0]
                sentence = columns[2]
                sentences_tot.append(sentence)
    embeddings = model.encode(sentences.tot, show_progress_bar=True)

    specific_embedding = embeddings[id]
        
        
    return torch.ones(20, dtype=torch.float32)