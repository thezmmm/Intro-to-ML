import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("./data/train.csv")
X = df.drop(columns=["id", "date", "class4", "partlybad"])
y = df["class4"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 50% training, 50% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.5, random_state=42, stratify=y
)

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on validation set
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation accuracy without PCA: {acc:.4f}")

# Try different numbers of PCA components
component_range = list(range(1, X_train.shape[1] + 1))
pca_acc = []

for n_comp in component_range:
    pca = PCA(n_components=n_comp)

    # Fit PCA on combined training + validation (semi-supervised)
    X_combined = np.vstack([X_train, X_val])
    pca.fit(X_combined)

    # Transform training and validation sets
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_pca, y_train)

    # Evaluate
    y_pred = clf.predict(X_val_pca)
    pca_acc.append(accuracy_score(y_val, y_pred))

# Find optimal number of components
optimal_idx = np.argmax(pca_acc)
optimal_components = component_range[optimal_idx]
optimal_accuracy = pca_acc[optimal_idx]

print(f"Optimal PCA components: {optimal_components}")
print(f"Validation accuracy with PCA: {optimal_accuracy:.4f}")

# Plot accuracy vs number of components
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(component_range, pca_acc, marker='o')
plt.axhline(acc, color='red', linestyle='--', label='No PCA Accuracy')
plt.xlabel('Number of PCA Components')
plt.ylabel('Validation Accuracy')
plt.title('Classifier Performance vs PCA Dimensionality')
plt.legend()
plt.grid(True)
plt.show()
