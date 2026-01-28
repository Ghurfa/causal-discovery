from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import pandas as pd

reader = BIFReader("cancer.bif")
model = reader.get_model()

sampler = BayesianModelSampling(model)
data = sampler.forward_sample(size=100000)

for col in data.columns:
    unique_values = data[col].unique()
    if len(unique_values) == 2:
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        data[col] = data[col].map(mapping)

data.to_csv("output.csv", index=False)

# Extract ground truth adjacency matrix from BIF structure
nodes = list(model.nodes())
num_nodes = len(nodes)
node_to_idx = {node: i for i, node in enumerate(nodes)}

# Create adjacency matrix (edge from i to j means i -> j)
true_matrix = np.zeros((num_nodes, num_nodes))
for parent, child in model.edges():
    i = node_to_idx[parent]
    j = node_to_idx[child]
    true_matrix[i, j] = 1

# Save true matrix to CSV for comparison
true_df = pd.DataFrame(true_matrix, index=nodes, columns=nodes)
true_df.to_csv("true_structure.csv")

print("Ground truth adjacency matrix from BIF:")
print(true_df)
print("\nSaved to true_structure.csv for comparison with NOTEARS results")
