
"""
=============================================================
SE4009 - Process Mining and Simulation | Assignment 2
Member 1 Tasks:
  Part 1  : Dataset Selection
  Part 2.1: Distance Matrix (Euclidean, with Heatmap)
  Part 2.2: Centroid Initialisation
  Part 2.3: Optimal Number of Clusters (Elbow Method)
Dataset : Mall Customer Segmentation Dataset
URL     : https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PART 1 — Load REAL Dataset from Kaggle CSV
# ─────────────────────────────────────────────────────────────
csv_path = os.path.join(BASE_DIR, 'Mall_Customers.csv')
df = pd.read_csv(csv_path)

# Rename columns for easier use (no spaces)
df.rename(columns={
    'Annual Income (k$)'       : 'Annual_Income_kUSD',
    'Spending Score (1-100)'   : 'Spending_Score'
}, inplace=True)

print("=" * 60)
print("PART 1 — Dataset Overview")
print("=" * 60)
print(f"Columns : {list(df.columns)}")
print(f"Shape   : {df.shape}")
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────
# Pre-processing — scale features for fair distance computation
# ─────────────────────────────────────────────────────────────
features = ['Age', 'Annual_Income_kUSD', 'Spending_Score']

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
X_df     = pd.DataFrame(X_scaled, columns=features, index=df['CustomerID'])

# Save cleaned dataset for other members
df_clean = df.copy()
df_clean['Age_scaled']    = X_scaled[:, 0]
df_clean['Income_scaled'] = X_scaled[:, 1]
df_clean['Score_scaled']  = X_scaled[:, 2]
clean_path = os.path.join(BASE_DIR, 'mall_customers_clean.csv')
df_clean.to_csv(clean_path, index=False)
print("\nCleaned dataset saved: mall_customers_clean.csv")

# ─────────────────────────────────────────────────────────────
# PART 2.1 — Distance Matrix
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2.1 — Euclidean Distance Matrix")
print("=" * 60)

# Use a 30-point representative sample for readability
sample_size = 30
np.random.seed(0)
sample_idx  = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample    = X_scaled[sample_idx]
ids_sample  = df['CustomerID'].iloc[sample_idx].values

dist_matrix = pairwise_distances(X_sample, metric='euclidean')
dist_df     = pd.DataFrame(dist_matrix,index=ids_sample,columns=ids_sample).round(4)

print("\nDistance Matrix (first 8 × 8 corner):")
print(dist_df.iloc[:8, :8].to_string())

dist_csv_path = os.path.join(BASE_DIR, 'distance_matrix_30sample.csv')
dist_df.to_csv(dist_csv_path)
print("\nFull 30×30 distance matrix saved: distance_matrix_30sample.csv")

# — Heatmap —
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(dist_df, cmap='YlOrRd', linewidths=0.3, linecolor='white',
            xticklabels=2, yticklabels=2,
            cbar_kws={'label': 'Euclidean Distance', 'shrink': 0.8},
            ax=ax)
ax.set_title('Pairwise Euclidean Distance Matrix\n(30-Customer Representative Sample — Real Kaggle Data)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Customer ID', fontsize=11)
ax.set_ylabel('Customer ID', fontsize=11)
ax.tick_params(axis='both', labelsize=8)
plt.tight_layout()
heatmap_path = os.path.join(BASE_DIR, 'plot_distance_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print("Distance heatmap saved: plot_distance_heatmap.png")

# ─────────────────────────────────────────────────────────────
# PART 2.2 — Centroid Initialisation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2.2 — Centroid Initialisation (K-Means++)")
print("=" * 60)

# Show final centroids (k=5) in original feature space
demo_k = 5
km_demo = KMeans(n_clusters=demo_k, init='k-means++',
                 n_init=10, random_state=42)
km_demo.fit(X_scaled)
init_centroids_orig = scaler.inverse_transform(km_demo.cluster_centers_)

print(f"Converged centroids (k={demo_k}) in original feature space:\n")
cents_df = pd.DataFrame(init_centroids_orig,
                        columns=features,
                        index=[f'C{i+1}' for i in range(demo_k)]).round(2)
print(cents_df.to_string())

# ─────────────────────────────────────────────────────────────
# PART 2.3 — Optimal Number of Clusters (Elbow Method)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2.3 — Elbow Method (k = 2 … 10)")
print("=" * 60)

k_range = range(2, 11)
wcss    = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++',
                n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

results_df = pd.DataFrame({'k': list(k_range), 'WCSS': wcss})
drops = [None] + [round((wcss[i-1] - wcss[i]) / wcss[i-1] * 100, 1)
                  for i in range(1, len(wcss))]
results_df['WCSS_Drop_%'] = drops

print("\nWCSS values per k (with % drop):\n")
print(results_df.to_string(index=False))

# — Elbow Plot —
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), wcss, 'o-', color='steelblue',
        linewidth=2.5, markersize=8, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='steelblue')

for k, w in zip(k_range, wcss):
    ax.annotate(f'{w:.1f}', xy=(k, w),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, color='#333333')

chosen_k    = 5
chosen_wcss = wcss[chosen_k - 2]
ax.axvline(x=chosen_k, color='crimson', linestyle='--',
           linewidth=1.8, alpha=0.8, label=f'Optimal k = {chosen_k}')
ax.scatter([chosen_k], [chosen_wcss], s=200, zorder=5,
           color='crimson', edgecolors='darkred', linewidths=1.5)

ax.set_title('Elbow Method — Within-Cluster Sum of Squares (WCSS)\nMall Customer Segmentation Dataset (Real Kaggle Data)',
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('WCSS (Inertia)', fontsize=12)
ax.set_xticks(list(k_range))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.35, linestyle='--')
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')
plt.tight_layout()
elbow_path = os.path.join(BASE_DIR, 'plot_elbow.png')
plt.savefig(elbow_path, dpi=150, bbox_inches='tight')
plt.close()
print("\nElbow plot saved: plot_elbow.png")
print(f"\nOptimal k = 5  (elbow visible after k=5, drops flatten to ~10%)")

