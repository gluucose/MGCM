import pandas as pd
import numpy as np

# Load adjacency matrix A, skipping the header row
A = pd.read_csv("../datasets_csv/lusc/Adj_Matrix/0_Adj.csv", header=0)
# Convert all elements to float type
A = A.astype(float)

# A is a KxK adjacency matrix: K represents the Top-K differentially expressed genes (DEGs)
print(f"A.shape: {A.shape}")

def generate_edge_matrix(A, adj_thresh):
	"""
    Generate edge matrix E from adjacency matrix A based on threshold

    Args:
        A (pd.DataFrame): Adjacency matrix
        adj_thresh (float): Threshold for edge creation

    Returns:
        E (np.ndarray): Edge matrix (binary)
        num_edges (int): Number of edges in the graph
    """
	# Get matrix dimensions
	n = A.shape[0]

	# Initialize edge matrix with zeros
	E = np.zeros_like(A, dtype=int)

	# Iterate through upper triangular part of matrix (symmetric)
	for i in range(n):
		for j in range(i + 1, n):
			if A.iloc[i, j] > adj_thresh:
				E[i, j] = 1
				E[j, i] = 1  # Ensure symmetry

	# Calculate number of edges (each edge is counted twice)
	num_edges = np.sum(E) // 2
	return E, num_edges


# Threshold for edge generation (e.g, 0.05)
adj_thresh_example = 0.05

# Generate edge matrix and calculate edge count
E, num_edges = generate_edge_matrix(A, adj_thresh_example)

# Save edge matrix to CSV file
output_path = "../datasets_csv/lusc/Edge_Matrix/0_Edge.csv"
np.savetxt(output_path, E, delimiter=",", fmt="%d")

# Output results
print(f"Number of edges in matrix E: {num_edges}")
print(f"Edge matrix saved to: {output_path}")
print(f"E.shape: {E.shape}")