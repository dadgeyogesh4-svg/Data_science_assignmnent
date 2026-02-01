# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:55 2023


"""

import pandas as pd
df=pd.read_csv("bank-full.csv")
df
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('bank-full.csv')
data
data.head
len(data)
list(data)
if 'y' in data.columns:

    X = data.drop('y', axis=1)
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
else:
    print("Column 'y' not found in the dataset.")



#==============================================================================
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the crime dataset
crime_data = pd.read_csv('crime_data.csv')

# Display the first few rows of the dataset
print(crime_data.head())

# Extract relevant features
X = crime_data[['Murder', 'Assault', 'UrbanPop', 'Rape']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate clustering using silhouette score
silhouette_agg = silhouette_score(X_scaled, agg_labels)
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)

# Print silhouette scores
print(f'Silhouette Score (Hierarchical Clustering): {silhouette_agg:.2f}')
print(f'Silhouette Score (K-means Clustering): {silhouette_kmeans:.2f}')
print(f'Silhouette Score (DBSCAN): {silhouette_dbscan:.2f}')

# Visualize Hierarchical Clustering Dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
