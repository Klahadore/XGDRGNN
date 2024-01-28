import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import LinkNeighborLoader
import pickle
from alpha_model import Model, metadata


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


@torch.no_grad()
def visualize_emb_1(embeddings, colors):
    tsne = TSNE(n_components=2, random_state=69)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
    print("embeddings len", len(embeddings))
    plt.figure(figsize=(10, 8))

    num_embeddings = embeddings.shape[0]
    plt.scatter(
        embeddings_2d[:num_embeddings, 0],
        embeddings_2d[:num_embeddings, 1],
        color="orange",
    )
    print("scattered")
    plt.legend()

    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title("Disease Embeddings")
    plt.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    model = Model(384, training=False)
    model.load_state_dict(torch.load("alphaModel_2_epochs.pt"))
    print("t")
    model.cuda()
    print("loaded model")
    test_data = pickle.load(open("data/new_test_dataset.pickle", "rb"))

    data_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=[30, 21],
        batch_size=512,
        shuffle=False,
        edge_label=test_data.edge_label[: test_data.edge_label_index.shape[1]],
        edge_label_index=test_data.edge_label_index,
        num_workers=4,
    )

    model.eval()
    map = {}
    print(metadata)
    for i in data_loader:

        i.cuda()
        _, z = model(i)
        z.cpu()
        for line in range(len(z)):
            key = (i.n_id[line], i.node_type[line])
            if key not in map.keys():
                map[key] = z[line].cpu()
            else:
                pass
            # map[key] = torch.mean(map[key], z[line]).cpu()
        print(z.shape)

        break

    genes = []
    # makes tensor of genes and diseases
    for i in map.keys():
        index, node_type = i
        if node_type == 2:
            genes.append(map[i])

    genes = torch.stack(genes)
    visualize_emb_1(genes, "blue")
# print(map)
