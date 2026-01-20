import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from itertools import permutations

# read data
df = pd.read_csv("./data/train.csv")

# get .mean features
mean_cols = [col for col in df.columns if col.endswith(".mean")]
X = df[mean_cols]
y = df["class4"].astype('category').cat.codes

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4-class clusters
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# confusion matrix
C = confusion_matrix(y, clusters)

# find best Permutations
best_perm = None
best_sum = -1

for perm in permutations(range(4)):
    M = C[:, perm]
    s = np.trace(M)
    if s > best_sum:
        best_sum = s
        best_perm = perm

C_aligned = C[:, best_perm]

print("Best column order:", best_perm)
print("Confusion matrix aligned:\n", C_aligned)
print("Diagonal sum (maximized):", np.trace(C_aligned))
