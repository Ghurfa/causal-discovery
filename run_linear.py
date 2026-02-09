import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from linear_test import notears_linear

# Load data
df = pd.read_csv('../data/output.csv')
X = df.values.astype(float)  

X = X.astype(float)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
#X = X - np.mean(X, axis=0, keepdims=True)


res = notears_linear(
    X_scaled,
    lambda1=0.001,
    loss_type='logistic',
    max_iter=1000,
    w_threshold=0.05,
    record_loss=True
)

if isinstance(res, (tuple, list)):
    W_est, outer_loss, inner_loss_history = res
else:
    W_est = res
    outer_loss, inner_loss_history = None, None


# Print weighted adjacency
print("Estimated adjacency matrix:")
print(np.round(W_est, 4))

# Print discovered edges
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
    plt.title('NOTEARs Linear: Outer Loss per Epoch')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(script_dir, 'loss_outer.png')
    plt.savefig(out_path)
    print(f"Saved outer loss plot to: {out_path}")

# Plot inner optimizer loss (gradient-descent iterations)
if inner_loss_history is not None:
    flat_losses = []
    boundaries = []
    cum = 0
    for epoch_hist in inner_loss_history:
        flat_losses.extend(epoch_hist)
    if len(flat_losses) > 0:
        plt.figure()
        plt.plot(range(1, len(flat_losses) + 1), flat_losses, marker='.', linewidth=1)
        plt.xlabel('Inner Iteration (cumulative)')
        plt.ylabel('Loss')
        plt.title('NOTEARs Linear: Inner Optimizer Loss per Iteration')
        plt.grid(True)
        plt.tight_layout()
        in_path = os.path.join(script_dir, 'loss_inner.png')
        plt.savefig(in_path)
        print(f"Saved inner loss plot to: {in_path}")
