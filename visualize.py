import matplotlib.pyplot as plt
import networkx as nx


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

