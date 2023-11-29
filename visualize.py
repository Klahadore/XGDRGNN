import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE


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


def visualize_emb(embeddings, num_genes):
    # Project embeddings to 2D space using t-SNE
    tsne = TSNE(n_components=2, random_state=69)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

    # Determine node types based on the index
    node_types = ['gene' if i < num_genes else 'disease' for i in range(embeddings.shape[0])]

    # Plotting
    plt.figure(figsize=(10, 8))
    for node_type in set(node_types):
        indices = [i for i, t in enumerate(node_types) if t == node_type]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=node_type)

    plt.legend()
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('Node Embeddings Visualization')
    plt.show()
