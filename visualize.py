import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE


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
    plt.scatter(
        embeddings_2d[:num_genes, 0], embeddings_2d[:num_genes, 1], label="gene"
    )

    # Plot disease embeddings
    plt.scatter(
        embeddings_2d[num_genes:, 0], embeddings_2d[num_genes:, 1], label="disease"
    )

    plt.legend()
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("Gene and Disease Embeddings Visualization")
    plt.show()
