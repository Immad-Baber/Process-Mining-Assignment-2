"""
=============================================================
SE4009 - Process Mining and Simulation | Assignment 2
Member 2 Tasks:
 Part 2.4: Cluster Visualisation (PCA scatter plots + domain plot)
 Part 2.5: CSV Output (clustered dataset)
 Part 2.6: Per-Cluster Visualisation & Interpretation
Dataset : Mall Customer Segmentation Dataset
Builds on Member 1's preprocessed file: mall_customers_clean.csv
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Resolve important paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'mall_customers_clean.csv')

# ─────────────────────────────────────────────────────────────
# Load Member 1's cleaned/scaled dataset
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Scaled feature matrix (produced by Member 1's StandardScaler)
scaled_cols = ['Age_scaled', 'Income_scaled', 'Score_scaled']
X_scaled = df[scaled_cols].values

# Original (unscaled) feature columns
orig_cols = ['Age', 'Annual_Income_kUSD', 'Spending_Score']

print("=" * 60)
print("Member 2 — Sections 2.4 / 2.5 / 2.6")
print("=" * 60)
print(f"Dataset shape : {df.shape}")
print(f"Scaled features : {scaled_cols}")

# ─────────────────────────────────────────────────────────────
# === 2.4 === CLUSTER VISUALISATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2.4 — Cluster Visualisation")
print("=" * 60)

# Reduce to 2D with PCA for all plots
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
      f"Total={sum(explained):.1%}")

# Colour palette — distinct colours per cluster (up to 5)
PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Transform centroids to PCA space
    centroids_pca = pca.transform(km.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=PALETTE[cluster_id],
            label=f'Cluster {cluster_id + 1}',
            alpha=0.75, edgecolors='white', linewidths=0.4, s=60
        )
    # Plot centroids
    ax.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        marker='X', s=200, c='black', zorder=5,
        label='Centroids', edgecolors='white', linewidths=0.8
    )
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title(f'K-Means Clustering: k = {k}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    fname = os.path.join(BASE_DIR, f'plot_kmeans_k{k}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ── Domain-specific visualisation for k=5 ──────────────────
km5 = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels5 = km5.fit_predict(X_scaled)

# Reconstruct original-scale centroids from the scaled centroids
# Member 1's scaler was fit on [Age, Annual_Income_kUSD, Spending_Score]
# We re-fit the same scaler to get exact inverse transform
scaler = StandardScaler()
scaler.fit(df[orig_cols].values)
centroids_orig = scaler.inverse_transform(km5.cluster_centers_)

fig, ax = plt.subplots(figsize=(9, 6))
for cluster_id in range(5):
    mask = labels5 == cluster_id
    ax.scatter(
        df['Annual_Income_kUSD'].values[mask],
        df['Spending_Score'].values[mask],
        c=PALETTE[cluster_id],
        label=f'Cluster {cluster_id + 1}',
        alpha=0.75, edgecolors='white', linewidths=0.4, s=65
    )
# Centroids in original scale (income = col 1, score = col 2)
ax.scatter(
    centroids_orig[:, 1], centroids_orig[:, 2],
    marker='X', s=220, c='black', zorder=5,
    label='Centroids', edgecolors='white', linewidths=0.8
)
ax.set_xlabel('Annual Income (k$)', fontsize=12)
ax.set_ylabel('Spending Score (1-100)', fontsize=12)
ax.set_title('Customer Segments: Income vs Spending Score (k=5)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_domain_specific.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_domain_specific.png")

# ─────────────────────────────────────────────────────────────
# === 2.5 === CSV OUTPUT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2.5 — CSV Output (mall_customers_clustered.csv)")
print("=" * 60)

df_out = df.copy()
# 1-indexed cluster labels
df_out['Cluster'] = labels5 + 1
df_out.to_csv(os.path.join(BASE_DIR, 'mall_customers_clustered.csv'), index=False)
print("Saved: mall_customers_clustered.csv")

print("\nFirst 15 rows (CustomerID, Age, Annual_Income_kUSD, Spending_Score, Cluster):")
display_cols = ['CustomerID', 'Age', 'Annual_Income_kUSD', 'Spending_Score', 'Cluster']
print(df_out[display_cols].head(15).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# === 2.6 === PER-CLUSTER VISUALISATION & INTERPRETATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2.6 — Per-Cluster Visualisation")
print("=" * 60)

# Prepare a clean analysis dataframe with original features + cluster
df_analysis = df[orig_cols].copy()
df_analysis['Cluster'] = labels5 + 1   # 1-indexed

feature_labels = {
    'Age': 'Age (years)',
    'Annual_Income_kUSD': 'Annual Income (k$)',
    'Spending_Score': 'Spending Score (1-100)'
}

# ── Box plots (1 row, 3 subplots) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Feature Distribution per Cluster', fontsize=15, fontweight='bold', y=1.01)

for ax, feat in zip(axes, orig_cols):
    groups = [df_analysis[df_analysis['Cluster'] == c][feat].values
              for c in range(1, 6)]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, colour in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel(feature_labels[feat], fontsize=11)
    ax.set_title(feature_labels[feat], fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f'C{i}' for i in range(1, 6)])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f8f9fa')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_boxplots_per_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_boxplots_per_cluster.png")

# ── Cluster means bar chart ──────────────────────────────────
cluster_means = df_analysis.groupby('Cluster')[orig_cols].mean().round(2)
print("\nCluster-wise Mean Values:")
print(cluster_means.to_string())

x = np.arange(5)
bar_w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))

# Normalise each feature to 0-100 scale for legible comparison on one chart
norm_means = cluster_means.copy()
for col in orig_cols:
    col_min = df_analysis[col].min()
    col_max = df_analysis[col].max()
    norm_means[col] = (cluster_means[col] - col_min) / (col_max - col_min) * 100

bar_colours = ['#457B9D', '#2A9D8F', '#E9C46A']
for i, (feat, colour) in enumerate(zip(orig_cols, bar_colours)):
    ax.bar(x + i * bar_w, norm_means[feat], bar_w,
           label=feature_labels[feat], color=colour, alpha=0.85,
           edgecolor='white', linewidth=0.6)

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Normalised Mean Value (0-100 scale)', fontsize=11)
ax.set_title('Cluster Mean Feature Values (Normalised)', fontsize=13, fontweight='bold')
ax.set_xticks(x + bar_w)
ax.set_xticklabels([f'Cluster {i}' for i in range(1, 6)])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'plot_cluster_means.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot_cluster_means.png")

# ── Print cluster sizes for reference ───────────────────────
cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
print("\nCluster sizes:")
print(cluster_sizes.to_string())

print("\n" + "=" * 60)
print("All Member 2 outputs generated successfully.")
print("=" * 60)
