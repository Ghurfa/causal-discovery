import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
    lambda1=0.01,
    loss_type='l2',
    max_iter=1000,
    w_threshold=0.1, 
    record_loss=True
)
if isinstance(res, (tuple, list)):
    W_est, outer_loss, inner_loss_history = res
else:
    W_est = res
    outer_loss, inner_loss_history = None, None

np.fill_diagonal(W_est, 0)

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

# Create and visualize graph using networkx
G = nx.DiGraph()
node_names = df.columns.tolist()
G.add_nodes_from(node_names)

if len(edges) > 0:
    for i, j in edges:
        weight = W_est[i, j]
        source = df.columns[i]
        target = df.columns[j]
        G.add_edge(source, target, weight=weight)

print(f"\nGraph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Visualize the graph
script_dir = os.path.dirname(__file__)
if G.number_of_edges() > 0:
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=40, arrowstyle='->', width=2.5,
                          min_source_margin=25, min_target_margin=25)
    
    # Draw edge labels with weights
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
    
    plt.title('Causal Discovery Graph (NOTEARs Linear)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    graph_path = os.path.join(script_dir, 'causal_graph_linear.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"Saved causal graph to: {graph_path}")
    plt.close()
else:
    print("No edges to visualize")


# Plot outer loss history
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