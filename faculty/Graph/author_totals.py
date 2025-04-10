import networkx as nx
import pickle
import csv

def calculate_author_scores(graph_path, output_file="author_scores.csv"):
    # Load the graph from the gpickle file
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        author_scores = {}
    
    # For each node (author) in the graph
    for author in G.nodes():
        total_score = 0

        # Get all edges connected to this author
        for neighbor in G.predecessors(author):
            # Add the weight of this edge to the total score
            # Default to 0.0 if weight is not specified
            edge_data = G.get_edge_data(neighbor, author)
            weight = edge_data.get('weight', 0.0)
            total_score += weight
        
        author_scores[author] = total_score
    
    # Sort authors by score (highest to lowest)
    sorted_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Write the results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Author', 'Research Score'])
        for author, score in sorted_authors:
            writer.writerow([author, score])
    
    print(f"Author research scores have been written to {output_file}")
    return author_scores

if __name__ == "__main__":
    g_path = "graph.gpickle"
    calculate_author_scores(g_path)