import pandas as pd
import numpy as np
from linear_test import notears_linear
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/output.csv') 
X = df.values.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

W_est = notears_linear(X, lambda1=0.05, loss_type='logistic', max_iter=1000)
print("Estimated adjacency matrix:")
print(W_est)

print("\nDiscovered edges:")
edges = np.argwhere(W_est != 0)
if len(edges) > 0:
    for i, j in edges:
        print(f"  {df.columns[i]} → {df.columns[j]}: {W_est[i, j]:.4f}")
else:
    print("  No significant edges found")