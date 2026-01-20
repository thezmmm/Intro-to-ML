import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("./data/train.csv")

# filter features end with mean
mean_cols = [col for col in df.columns if col.endswith(".mean")]
X = df[mean_cols]

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# compute k-mean loss
inertia = []
K_range = range(1, 20)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# plot
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("K-means loss (inertia)")
plt.title("Elbow Method for K-means")
plt.show()
