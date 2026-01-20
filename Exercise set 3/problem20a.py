import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the data
df = pd.read_csv("./data/train.csv")

# 2. Separate features and target
X = df.drop(columns=["id", "date", "class4", "partlybad"])  # numerical features only
y = df["class4"]

# 3. Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Create a DataFrame with PCA results
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["class4"] = y

# 6. Plot the PCA projection
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="class4",       # color by class
    style="class4",     # shape by class
    palette="Set2",
    s=100               # marker size
)
plt.title("PCA Projection of Dataset into 2D")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class4")
plt.grid(True)
plt.show()
