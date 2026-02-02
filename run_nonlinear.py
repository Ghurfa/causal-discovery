import pandas as pd
import numpy as np
import torch
from nonlinear_test import NotearsMLP, notears_nonlinear

torch.set_default_dtype(torch.double)

# Load your data
df = pd.read_csv('../data/output.csv')
X = df.values.astype(float)
d = X.shape[1]

# Convert to torch double tensor
X_torch = torch.from_numpy(X).double()

# Instantiate the MLP model
model = NotearsMLP(dims=[d, 10, 1], bias=True)

# Run nonlinear NOTEARS
W_est = notears_nonlinear(
    model,
    X,
    lambda1=1e-5,      # L1 sparsity penalty
    lambda2=.1,      # DAG + L2 penalty
    max_iter=2000,
    w_threshold=1e-6
)

# Print adjacency
print("Estimated weighted adjacency:")
print(W_est)

# Thresholded edges
edges = np.argwhere(W_est != 0)
print("\nDiscovered edges:")
if len(edges) > 0:
    for i, j in edges:
        print(f"  {df.columns[i]} → {df.columns[j]}: {W_est[i, j]:.4f}")
else:
    print("  No significant edges found")
