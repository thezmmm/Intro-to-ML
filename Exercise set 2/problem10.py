import pandas as pd
import numpy as np

# Load training data
path = './data_E2/'
train = pd.read_csv(path+"penguins_train.csv")

# Species → Adelie vs notAdelie
train["label"] = (train["species"] == "Adelie").astype(int)

# The 4 Gaussian NB features (modify if your columns differ)
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Compute attribute means and std for each class
means = train.groupby("label")[features].mean()
stds = train.groupby("label")[features].std(ddof=1)  # unbiased std

print("=== Means per class ===")
print(means)
print("\n=== Standard deviations per class ===")
print(stds)

# Class probabilities with Laplace smoothing
N0 = sum(train["label"] == 0)
N1 = sum(train["label"] == 1)
N = len(train)
K = 2  # 2 classes

P0 = (N0 + 1) / (N + K)
P1 = (N1 + 1) / (N + K)

print("\n=== Class probabilities with Laplace smoothing ===")
print(f"P(y=0 | train) = {P0:.4f}   (notAdelie)")
print(f"P(y=1 | train) = {P1:.4f}   (Adelie)")


# Task c
# Load test data
test = pd.read_csv(path + "penguins_test.csv")
test["label"] = (test["species"] == "Adelie").astype(int)
X_test = test[features].values
y_test = test["label"].values


# Define Gaussian PDF function
def gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Function to compute P(y=Adelie | x)
def predict_proba(x):
    # convert pandas Series to numpy if needed
    if isinstance(x, pd.Series):
        x = x.values
    # likelihoods
    likelihood_0 = np.prod(gaussian_pdf(x, means.loc[0].values, stds.loc[0].values))
    likelihood_1 = np.prod(gaussian_pdf(x, means.loc[1].values, stds.loc[1].values))

    posterior_1 = P1 * likelihood_1
    posterior_0 = P0 * likelihood_0

    return posterior_1 / (posterior_0 + posterior_1)


# Compute probabilities and predictions for test set
probs = np.array([predict_proba(x) for x in X_test])
y_pred = (probs >= 0.5).astype(int)

# Classification accuracy
accuracy = np.mean(y_pred == y_test)
print(f"\nTest set classification accuracy: {accuracy:.4f}")

# Probabilities for the first three penguins
print("\nProbabilities P(y=Adelie | x) for first three test penguins:")
for i in range(3):
    print(f"Penguin {i + 1}: {probs[i]:.4f}")
