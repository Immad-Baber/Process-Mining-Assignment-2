"""
SE4009 - Process Mining and Simulation | Assignment 2
Member 3 Tasks:
  Part 3.1: Agglomerative Clustering (Dendrogram, Cluster Map, Interpretation)
  Part C  : Lift Calculation (Handwritten)
Dataset : Mall Customer Segmentation Dataset
Builds on: mall_customers_clean.csv (Member 1)
           mall_customers_clustered.csv (Member 2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df_clean      = pd.read_csv(os.path.join(BASE_DIR, 'mall_customers_clean.csv'))
df_clustered  = pd.read_csv(os.path.join(BASE_DIR, 'mall_customers_clustered.csv'))

scaled_cols = ['Age_scaled', 'Income_scaled', 'Score_scaled']
orig_cols   = ['Age', 'Annual_Income_kUSD', 'Spending_Score']
X_scaled    = df_clean[scaled_cols].values

PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
N_CLUSTERS = 5

print("=" * 60)
print("Member 3 — Part 3.1: Agglomerative Clustering")
print("=" * 60)
print(f"Dataset shape : {df_clean.shape}")

# ─────────────────────────────────────────────────────────────
# PART 3.1 — Agglomerative Clustering
# ─────────────────────────────────────────────────────────────

# Ward linkage minimises within-cluster variance at each merge step.
# It produces compact, roughly equal-sized clusters and is well-suited
# for Euclidean feature spaces after standardisation — making it the
# most appropriate choice for this dataset.

Z = linkage(X_scaled, method='ward', metric='euclidean')

print("\nLinkage matrix computed (Ward's method, Euclidean distance)")
print(f"Merge steps : {Z.shape[0]}  (n-1 merges for {len(X_scaled)} observations)")

# ── Dendrogram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

dendrogram(
    Z,
    ax=ax,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True,
    color_threshold=0,
    above_threshold_color='#457B9D'
)

cut_height = Z[-N_CLUSTERS + 1, 2]
ax.axhline(y=cut_height, color='crimson', linestyle='--',
           linewidth=1.8, label=f'Cut level = {cut_height:.2f}  →  k = {N_CLUSTERS}')

ax.set_title('Agglomerative Clustering Dendrogram (Ward Linkage)\nMall Customer Segmentation Dataset',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Customer (leaf node / merged cluster size)', fontsize=11)
ax.set_ylabel('Ward Linkage Distance', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_dendrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Dendrogram saved: plot_dendrogram.png")

# ── Cluster assignments ──────────────────────────────────────
agg_model  = AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                     linkage='ward',
                                     metric='euclidean')
agg_labels = agg_model.fit_predict(X_scaled) + 1  # 1-indexed

df_clean['Agglomerative_Cluster'] = agg_labels

# ── PCA for visualisation ────────────────────────────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
      f"Total={sum(explained):.1%}")

# ── Cluster map (PCA scatter) ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for cid in range(1, N_CLUSTERS + 1):
    mask = agg_labels == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PALETTE[cid - 1], label=f'Cluster {cid}',
               alpha=0.75, edgecolors='white', linewidths=0.4, s=65)

ax.set_xlabel('PCA Component 1', fontsize=12)
ax.set_ylabel('PCA Component 2', fontsize=12)
ax.set_title('Agglomerative Clustering (Ward, k=5) — PCA View',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_agglomerative_pca.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Agglomerative PCA cluster map saved: plot_agglomerative_pca.png")

# ── Domain-specific cluster map (Income vs Spending Score) ───
fig, ax = plt.subplots(figsize=(9, 6))
for cid in range(1, N_CLUSTERS + 1):
    mask = agg_labels == cid
    ax.scatter(df_clean['Annual_Income_kUSD'].values[mask],
               df_clean['Spending_Score'].values[mask],
               c=PALETTE[cid - 1], label=f'Cluster {cid}',
               alpha=0.75, edgecolors='white', linewidths=0.4, s=65)

ax.set_xlabel('Annual Income (k$)', fontsize=12)
ax.set_ylabel('Spending Score (1-100)', fontsize=12)
ax.set_title('Agglomerative Clusters: Income vs Spending Score (k=5)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_agglomerative_domain.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Agglomerative domain cluster map saved: plot_agglomerative_domain.png")

# ── Cluster means ────────────────────────────────────────────
df_analysis = df_clean[orig_cols].copy()
df_analysis['Cluster'] = agg_labels
cluster_means = df_analysis.groupby('Cluster')[orig_cols].mean().round(2)

print("\nAgglomerative Cluster-wise Mean Values:")
print(cluster_means.to_string())

cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
print("\nAgglomerative Cluster Sizes:")
print(cluster_sizes.to_string())

# ── CSV output ───────────────────────────────────────────────
df_out = df_clean.copy()
df_out['Agglomerative_Cluster'] = agg_labels
df_out.to_csv(os.path.join(BASE_DIR, 'mall_customers_agglomerative.csv'), index=False)
print("\nSaved: mall_customers_agglomerative.csv")

# ─────────────────────────────────────────────────────────────
# COMPARISON: Agglomerative vs K-Means
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Agglomerative vs K-Means Comparison")
print("=" * 60)

kmeans_labels = df_clustered['Cluster'].values
ari = adjusted_rand_score(kmeans_labels, agg_labels)
print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
print("(ARI = 1.0 → perfect agreement, 0.0 → random)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
titles = ['K-Means (k=5)', 'Agglomerative Ward (k=5)']
label_sets = [kmeans_labels, agg_labels]

for ax, labels, title in zip(axes, label_sets, titles):
    for cid in range(1, N_CLUSTERS + 1):
        mask = np.array(labels) == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=PALETTE[cid - 1], label=f'Cluster {cid}',
                   alpha=0.75, edgecolors='white', linewidths=0.4, s=60)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('PCA Component 1', fontsize=11)
    ax.set_ylabel('PCA Component 2', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')

fig.suptitle(f'Cluster Assignment Comparison  (ARI = {ari:.3f})',
             fontsize=14, fontweight='bold', y=1.02)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_comparison_kmeans_vs_agglomerative.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Comparison plot saved: plot_comparison_kmeans_vs_agglomerative.png")

print("\n" + "=" * 60)
print("All Member 3 outputs generated successfully.")
print("=" * 60)