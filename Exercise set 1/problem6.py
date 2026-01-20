import numpy as np
import statsmodels.api as sm
import pandas as pd

# Load d2.csv
d2 = pd.read_csv('./data/d2.csv')

X = sm.add_constant(d2['x'])
y = d2['y']

# Number of bootstrap samples
B = 1000
w0_boot = []
w1_boot = []

n = len(d2)

for _ in range(B):
    # sample indices with replacement
    sample_indices = np.random.choice(n, n, replace=True)
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    # fit OLS
    model = sm.OLS(y_sample, X_sample).fit()

    w0_boot.append(model.params[0])
    w1_boot.append(model.params[1])

# Compute bootstrap standard errors
se_w0_boot = np.std(w0_boot, ddof=1)
se_w1_boot = np.std(w1_boot, ddof=1)

print(f"Bootstrap SE for w0: {se_w0_boot:.4f}")
print(f"Bootstrap SE for w1: {se_w1_boot:.4f}")
