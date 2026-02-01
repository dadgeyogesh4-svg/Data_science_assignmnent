# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:30:01 2023

"""

# Import necessary libraries
import pandas as pd

wine_data = pd.read_csv('wine.csv')  

print(wine_data.head())
list(wine_data)
X = wine_data.drop('Alcohol', axis=1)
y = wine_data['Proline']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

from sklearn.cluster import AgglomerativeClustering, KMeans
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
agg_labels = agg_clustering.fit_predict(X_pca)

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Scree Plot (Elbow Curve) for K-means Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

optimal_k = 3

from scipy.cluster.hierarchy import dendrogram, linkage
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels_optimal = kmeans_optimal.fit_predict(X_pca)

from sklearn.metrics import silhouette_score
silhouette_agg = silhouette_score(X_pca, agg_labels)
silhouette_kmeans_optimal = silhouette_score(X_pca, kmeans_labels_optimal)

# Print silhouette scores
print(f'Silhouette Score (Hierarchical Clustering): {silhouette_agg:.2f}')
print(f'Silhouette Score (K-means Clustering): {silhouette_kmeans_optimal:.2f}')

comparison_df = pd.DataFrame({'Original Class': y, 'Agg Clustering': agg_labels, 'K-means Clustering': kmeans_labels_optimal})
print(comparison_df)
