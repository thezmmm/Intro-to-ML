import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("./data/train.csv")
mean_cols = [col for col in df.columns if col.endswith(".mean")]
X = df[mean_cols]
y = df["class4"].astype('category').cat.codes

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# compute linkage matrix
linkage_methods = ['ward', 'complete']

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)

    # plot dendrogram
    plt.figure(figsize=(12, 6))
    plt.title(f"Dendrogram ({method} linkage)")
    dendrogram(Z, no_labels=True, count_sort='ascending')
    plt.show()

    # divide into 4 categories
    cluster_model = AgglomerativeClustering(n_clusters=4, linkage=method)
    clusters = cluster_model.fit_predict(X_scaled)

    # confusion matrix
    C = confusion_matrix(y, clusters)
    print(f"\nConfusion matrix ({method} linkage):\n", C)
