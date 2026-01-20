### Problem1

```python
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
```

![problem1](.\problem1.png)

### Problem2

```python
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
```

```shell
Eigenvalues: [ 3.33272948 -0.71472948]
Eigenvectors (columns):
 [[-0.65088847 -0.75917336]
 [-0.75917336  0.65088847]]
X^T X =
 [[1. 0.]
 [0. 1.]]
Reconstruction:
 [[1.    2.   ]
 [2.    1.618]]
```

### Problem3

**Target**: For any random variable `X, Y` and any real scalar `a, b`:
$$
E[aX+bY] = aE[X]+bE[Y]
$$

$$
E[aX+bY] = \sum_{\omega \in \Omega} P(\omega)(aX(\omega)+bY(\omega)) \\
=a\sum_{\omega \in \Omega} P(\omega)X(\omega)+b\sum_{\omega \in \Omega} P(w)Y(\omega)\\
=aE[X]+bE[Y]
$$
**Target**: if `μ=E[X]`
$$
Var[X] = E[(X-\mu)^2] = E[X^2] - E[X]^2
$$

$$
E[(X-\mu)^2] = E[X^2-2\cdot \mu \cdot X+\mu^2]\\
=E[X^2]-2\cdot\mu E[X] + \mu^2\\
=E[X^2]-2\cdot\mu \cdot\mu + \mu^2\\
=E[X^2]-\mu^2\\
=E[X^2]-E[X]^2
$$

### Problem4

A
$$
P(X | Y ) = \frac{P(X ∧Y)}{P(Y)}
$$

$$
P(Y | X ) = \frac{P(X ∧Y)}{P(X)}
$$

$$
\frac{P(X | Y )}{P(Y | X )} = \frac{P(X)}{P(Y)}
$$

B

Define Boolean random variables:

- X = “person has pollen allergy” (true/false).
- Y = “test result is positive” (true/false).

Given
$$
P(X)=0.20 \;\;\; P(¬X)=0.80\\
P(Y∣¬X)=0.23\\
P(Y∣X)=1−0.15=0.85
$$

$$
P(Y)=P(Y∣X)P(X)+P(Y∣¬X)P(¬X)=0.354
$$

$$
P(X∣Y)=\frac{P(X)P(Y∣X)}{P(Y)} ≈ 0.48
$$



### Problem5

A

differentiate b
$$
f'(b) = \frac{1}{2}\sum_{i=1}^n2x_i(bx_i-y_i)\\
=\sum_{i=1}^nx_i(bx_i-y_i)\\
=b\sum_{i=1}^nx_i^2-\sum_{i=1}^nx_iy_i
$$
set `f'(b) = 0`
$$
b = \frac{\sum_{i=1}^nx_iy_i}{\sum_{i=1}^nx_i^2}
$$

$$
f''(b) =\sum_{i=1}^nx_i^2 >= 0
$$

so `f'(b)` strict increase

so `f(b)` is min when `f'(b) = 0`
$$
b = \frac{\sum_{i=1}^nx_iy_i}{\sum_{i=1}^nx_i^2}
$$
B
$$
f''(b) =\sum_{i=1}^nx_i^2 > 0
$$
