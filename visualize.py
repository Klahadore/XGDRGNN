import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import LinkNeighborLoader
import pickle
from alpha_model import Model

@torch.no_grad()
def visualize_graph(data):
    G = nx.Graph()

    # Assuming 'data' is a PyTorch Geometric Data object
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Adding nodes
    for node in range(num_nodes):
        G.add_node(node)

    # Adding edges
    if edge_index is not None:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

    # Drawing the graph
    nx.draw(G)
    plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def visualize_emb(gene_embeddings, disease_embeddings):
    # Concatenate gene and disease embeddings
    all_embeddings = torch.cat((gene_embeddings, disease_embeddings), dim=0)

    # Project embeddings to 2D space using t-SNE
    tsne = TSNE(n_components=2, random_state=69)
    embeddings_2d = tsne.fit_transform(all_embeddings.cpu().numpy())

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot gene embeddings
    num_genes = gene_embeddings.shape[0]
    plt.scatter(embeddings_2d[:num_genes, 0], embeddings_2d[:num_genes, 1], label='gene')

    # Plot disease embeddings
    plt.scatter(embeddings_2d[num_genes:, 0], embeddings_2d[num_genes:, 1], label='disease')

    plt.legend()
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('Gene and Disease Embeddings Visualization')
    plt.show()

if __name__ == "__main__":
    model = Model(384, training=False)
    model.load_state_dict(torch.load('alphaModel_2_epochs.pt', map_location=torch.device('cpu')))

    test_data = pickle.load(open("data/new_test_dataset.pickle", 'rb'))
    data_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[25, 21],
        batch_size=128,
        shuffle=False,
        edge_label=test_data.edge_label[:test_data.edge_label_index.shape[1]],
        edge_label_index=test_data.edge_label_index,
        num_workers=4
    )
    model.eval()
    for batch in data_loader:
        pred, z = model(batch)
        print(z)
