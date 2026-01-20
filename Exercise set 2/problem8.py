import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
path = './data_E2/'
train_df = pd.read_csv(path+"penguins_train.csv")
test_df = pd.read_csv(path+"penguins_test.csv")

# target: "species", other columns as input features
X_train = train_df.drop(columns=["species"])
y_train = (train_df["species"] == "Adelie").astype(int)  # Adelie=1, notAdelie=0

X_test = test_df.drop(columns=["species"])
y_test = (test_df["species"] == "Adelie").astype(int)

# fit logistic regression without regularisation(penalty)
model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
# fit logistic regression with Lasso regularisation
# model = LogisticRegression(penalty='l1',solver='liblinear',C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# report intercept and coefficients
print("Intercept (β₀):", model.intercept_)
print("Coefficients (β):")
for feature, coef in zip(X_train.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")

# report accuracy
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTraining accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")

# compute ti = βᵀxᵢ and P(y=Adelie|xᵢ)
t_train = model.decision_function(X_train)  # βᵀxᵢ
p_train = model.predict_proba(X_train)[:, 1]  # P(y=1|xᵢ)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(t_train[y_train == 1], p_train[y_train == 1], color="blue", label="Adelie")
plt.scatter(t_train[y_train == 0], p_train[y_train == 0], color="orange", label="Not Adelie")
plt.xlabel("Linear response tᵢ = βᵀxᵢ")
plt.ylabel("Predicted probability P(yᵢ = Adelie | xᵢ)")
plt.title("Logistic Regression Fit on Training Data")
plt.legend()
plt.grid(True)
plt.show()
