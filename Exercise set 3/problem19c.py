import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("./data/train.csv")
mean_cols = [col for col in df.columns if col.endswith(".mean")]
X = df[mean_cols]

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# random initialisations
losses = []

for i in range(1000):
    kmeans = KMeans(
        n_clusters=4,
        # init="random",
        init="k-means++",
        n_init=1,
        random_state=None
    )
    kmeans.fit(X_scaled)
    losses.append(kmeans.inertia_)   # inertia = k-means loss

losses = np.array(losses)

# plot
plt.hist(losses, bins=30)
plt.xlabel("K-means loss")
plt.ylabel("Frequency")
plt.title("Loss distribution across 1000 random initialisations")
plt.show()

print("Min loss:", np.min(losses))
print("Max loss:", np.max(losses))

# compute “reasonably good”
best_loss = np.min(losses)
threshold = best_loss * 1.01    # 1% 内
good_count = np.sum(losses <= threshold)

print("Best loss:", best_loss)
print("Threshold for 'reasonably good':", threshold)
print("Number of good solutions:", good_count)
