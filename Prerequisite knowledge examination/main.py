import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# problem1
data = pd.read_csv('x.csv')

variances = data.var(numeric_only=True)

top2 = variances.nlargest(2).index.tolist()
var1, var2 = top2[0], top2[1]
print(f"Top 2 variables by variance: {var1}, {var2}")

plt.figure()
plt.scatter(data[var1], data[var2])
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(f"Scatterplot of {var1} vs {var2}")
plt.grid(True)
plt.savefig("problem1.png")
plt.show()

# problem2
A = np.array([[1.0, 2.0],
              [2.0, 1.618]])
w, v = np.linalg.eig(A)

order = np.argsort(-w)
w = w[order]
v = v[:, order]

for i in range(2):
    v[:,i] = v[:,i] / np.linalg.norm(v[:,i])

print("Eigenvalues:", w)
print("Eigenvectors (columns):\n", v)
print("X^T X =\n", np.round(v.T @ v, 12))
recon = sum(w[i] * np.outer(v[:,i], v[:,i]) for i in range(2))
print("Reconstruction:\n", recon)

# problem3
