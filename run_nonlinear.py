import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from nonlinear_test import NotearsMLP, notears_nonlinear

torch.set_default_dtype(torch.double)

# Load your data
df = pd.read_csv('../data/output.csv')
X = df.values.astype(float)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
d = X_scaled.shape[1]
# Convert to torch double tensor
X_torch = torch.from_numpy(X).double()

# Instantiate the MLP model
model = NotearsMLP(dims=[d, 64, 32, 1], bias=True)

res = notears_nonlinear(
    model,
    X_scaled,
    lambda1=0.01,
    lambda2=1e-4,
    max_iter=1000,
    w_threshold=.1,
    record_loss=True
)

if isinstance(res, (tuple, list)):
    W_est, outer_loss, inner_loss_history = res
else:
    W_est = res
    outer_loss, inner_loss_history = None, None

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

# Plot outer loss history
script_dir = os.path.dirname(__file__)
if outer_loss is not None:
    plt.figure()
    plt.plot(range(1, len(outer_loss) + 1), outer_loss, marker='o')
    plt.xlabel('Outer Epoch')
    plt.ylabel('Loss')
    plt.title('NOTEARs Nonlinear: Outer Loss per Epoch')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(script_dir, 'loss_outer_nonlinear.png')
    plt.savefig(out_path)
    print(f"Saved outer loss plot to: {out_path}")

# Plot inner optimizer loss (gradient-descent iterations)
if inner_loss_history is not None:
    flat_losses = []
    boundaries = []
    cum = 0
    for epoch_hist in inner_loss_history:
        flat_losses.extend(epoch_hist)
        cum += len(epoch_hist)
        boundaries.append(cum)
    if len(flat_losses) > 0:
        plt.figure()
        plt.plot(range(1, len(flat_losses) + 1), flat_losses, marker='.', linewidth=1)
        for b in boundaries[:-1]:
            plt.axvline(b + 0.5, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Inner Iteration (cumulative)')
        plt.ylabel('Loss')
        plt.title('NOTEARs Nonlinear: Inner Optimizer Loss per Iteration')
        plt.grid(True)
        plt.tight_layout()
        in_path = os.path.join(script_dir, 'loss_inner_nonlinear.png')
        plt.savefig(in_path)
        print(f"Saved inner loss plot to: {in_path}")

dag_pred = (W_est != 0).astype(int)

import sys
sys.path.append('../DAGPA')

from eval import f1, f1_skeleton, shd, shd_skeleton, ci_mcc
print("\nEvaluation vs Ground Truth (Sachs):")

dag_sachs_truth = np.array([
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
[1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0],
[0.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0],
[0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],
[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
], dtype=int)

print("SHD (directed):       ", shd(dag_pred, dag_sachs_truth))
print("CI-MCC:              ", ci_mcc(dag_pred, true_dag=dag_sachs_truth))