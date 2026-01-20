import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("./data/train.csv")
X = df.drop(columns=["id", "date", "class4", "partlybad"])  # numeric features
y = df["class4"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA on scaled data
pca = PCA()
pca.fit(X_scaled)

# Proportion of variance explained (PVE)
pve = pca.explained_variance_ratio_
cumulative_pve = pve.cumsum()

# Plot PVE and cumulative PVE
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pve)+1), pve, alpha=0.5, align='center', label='Individual PVE')
plt.step(range(1, len(cumulative_pve)+1), cumulative_pve, where='mid', label='Cumulative PVE', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('PVE and Cumulative PVE (Normalized Data)')
plt.legend()
plt.grid(True)
plt.show()

# PCA on unscaled data
pca_unscaled = PCA()
pca_unscaled.fit(X)  # no scaling

pve_unscaled = pca_unscaled.explained_variance_ratio_
cumulative_pve_unscaled = pve_unscaled.cumsum()

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pve)+1), cumulative_pve, marker='o', label='Scaled Data (Standardized)')
plt.plot(range(1, len(pve_unscaled)+1), cumulative_pve_unscaled, marker='s', label='Unscaled Data')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.title('Effect of Normalization on PCA')
plt.legend()
plt.grid(True)
plt.show()

