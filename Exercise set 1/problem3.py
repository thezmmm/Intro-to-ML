import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# params
np.random.seed(0)
sigma = 0.4  # noise
sigma2 = sigma ** 2
n_sim = 1000
n_train = 10  # train set size
degrees = range(0, 7)


def f(x):
    return -2 - x + 0.5 * x ** 2


f0 = f(0.0)  # f(0)

results = []

for deg in degrees:
    fhat0_list = []
    y0_list = []

    for _ in range(n_sim):
        # generate train set
        x_train = np.random.uniform(-3, 3, n_train).reshape(-1, 1)
        eps_train = np.random.normal(0, sigma, n_train)
        y_train = f(x_train.ravel()) + eps_train

        # fit model
        if deg == 0:
            fhat0 = np.mean(y_train)
        else:
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_train = poly.fit_transform(x_train)
            model = LinearRegression()
            model.fit(X_train, y_train)
            fhat0 = model.predict(poly.transform([[0.0]]))[0]

        # generate a test sample (x=0, y0)
        y0 = f0 + np.random.normal(0, sigma)
        fhat0_list.append(fhat0)
        y0_list.append(y0)

    fhat0_arr = np.array(fhat0_list)
    y0_arr = np.array(y0_list)

    # compute
    irreducible = np.mean((y0_arr - f0) ** 2)
    bias = np.mean(fhat0_arr) - f0
    bias_sq = bias ** 2
    variance = np.mean((fhat0_arr - np.mean(fhat0_arr)) ** 2)
    total = irreducible + bias_sq + variance
    mse = np.mean((y0_arr - fhat0_arr) ** 2)

    results.append({
        "Degree": deg,
        "Irreducible": irreducible,
        "BiasSq": bias_sq,
        "Variance": variance,
        "Total (sum)": total,
        "MSE": mse
    })

df = pd.DataFrame(results)
print(df.round(6))

plt.figure(figsize=(8, 5))
plt.plot(df["Degree"], df["MSE"], 'o-', label="MSE (E[(y0 - fhat0)^2])")
plt.plot(df["Degree"], df["Irreducible"], 'o-', label="Irreducible (Noise)")
plt.plot(df["Degree"], df["BiasSq"], 'o-', label="Bias²")
plt.plot(df["Degree"], df["Variance"], 'o-', label="Variance")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error value at x=0")
plt.title("Bias–Variance Decomposition (n=10, 1000 simulations)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
