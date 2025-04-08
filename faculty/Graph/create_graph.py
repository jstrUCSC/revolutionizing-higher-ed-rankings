import csv
import networkx as nx
import pickle

# Initialize a directed graph
G = nx.DiGraph()

# Path to the CSV file
csv_file_path = '../Scoring/faculty_full_names.csv'

# Read the CSV file and populate the graph
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for row in reader:
        reference_author = row[0].strip()
        original_author = row[1].strip()
        weight = float(row[2].strip())

        # Add nodes if they don't already exist
        if not G.has_node(reference_author):
            G.add_node(reference_author)
        if not G.has_node(original_author):
            G.add_node(original_author)

        # Add or update the edge with the weight
        if G.has_edge(original_author, reference_author):
            G[original_author][reference_author]['weight'] += weight
        else:
            G.add_edge(original_author, reference_author, weight=weight)

with open('graph.gpickle', 'wb') as f:
    pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

print("Graph creation complete. Nodes and edges have been added.")