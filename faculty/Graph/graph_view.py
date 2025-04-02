import networkx as nx
import pickle
import matplotlib.pyplot as plt

# Load the graph from the gpickle file
g_path = "graph.gpickle"
with open(g_path, 'rb') as f:
    G = pickle.load(f)

# Draw the graph using a spring layout
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color="gray")

# Draw edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show the plot
plt.title("Graph Visualization with Edge Weights")
plt.axis('off')
plt.show()