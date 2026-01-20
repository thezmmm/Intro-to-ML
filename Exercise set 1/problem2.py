import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

RANDOM_STATE = 0

TRAIN_FILE = "./data/train_syn.csv"
VALID_FILE = "./data/valid_syn.csv"
TEST_FILE  = "./data/test_syn.csv"

train = pd.read_csv(TRAIN_FILE)
valid = pd.read_csv(VALID_FILE)
test  = pd.read_csv(TEST_FILE)

# Determine input and target column names automatically if only two columns
def infer_xy(df):
    if df.shape[1] == 2:
        cols = df.columns.tolist()
        return df[cols[0]].values.reshape(-1,1), df[cols[1]].values.ravel()
    else:
        # if more columns (e.g., id,date,...), user should replace this function
        raise ValueError("Expected CSV with exactly 2 columns (x,y) or modify this script to select appropriate columns")

X_train, y_train = infer_xy(train)
X_valid, y_valid = infer_xy(valid)
X_test,  y_test  = infer_xy(test)

# combined training+validation for CV and TestTRVA
X_trva = np.vstack([X_train, X_valid])
y_trva = np.concatenate([y_train, y_valid])

def mse_for_degree(deg, X_tr, y_tr, X_eval, y_eval):
    """Fit polynomial OLS on (X_tr,y_tr) and evaluate MSE on (X_eval,y_eval). Handles deg=0."""
    if deg == 0:
        # deg 0: constant model predicting the mean of y_tr
        y_pred = np.full_like(y_eval, fill_value=np.mean(y_tr), dtype=float)
        return mean_squared_error(y_eval, y_pred)
    # build polynomial features (no bias term; LinearRegression will learn intercept)
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr)
    X_eval_poly = poly.transform(X_eval)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_tr_poly, y_tr)
    y_pred = model.predict(X_eval_poly)
    return mean_squared_error(y_eval, y_pred)

# function to compute 10-fold CV MSE (on combined train+valid)
def cv_mse_on_trva(deg, X_trva, y_trva, n_splits=10):
    if deg == 0:
        # CV of constant model: each fold's MSE = mean((y_val - mean(y_train_fold))^2)
        # Implement by manual KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        mses = []
        for train_idx, val_idx in kf.split(X_trva):
            y_tr_fold = y_trva[train_idx]
            y_val_fold = y_trva[val_idx]
            pred = np.full_like(y_val_fold, fill_value=np.mean(y_tr_fold), dtype=float)
            mses.append(mean_squared_error(y_val_fold, pred))
        return np.mean(mses)
    # use scikit's cross_val_score with neg_mean_squared_error
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_trva_poly = poly.fit_transform(X_trva)
    model = LinearRegression(fit_intercept=True)
    # cross_val_score gives arrays of scores; for MSE use scoring='neg_mean_squared_error'
    scores = cross_val_score(model, X_trva_poly, y_trva,
                             scoring='neg_mean_squared_error',
                             cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE))
    return -np.mean(scores)  # return positive MSE

# Loop degrees 0..8 and compute columns
rows = []
for deg in range(0, 9):
    train_mse = mse_for_degree(deg, X_train, y_train, X_train, y_train)
    val_mse   = mse_for_degree(deg, X_train, y_train, X_valid, y_valid)
    test_mse  = mse_for_degree(deg, X_train, y_train, X_test, y_test)
    # TestTRVA: train on combined train+valid, test on test set
    test_trva_mse = mse_for_degree(deg, X_trva, y_trva, X_test, y_test)
    # CV: 10-fold on combined train+valid
    cv_mse = cv_mse_on_trva(deg, X_trva, y_trva, n_splits=10)
    rows.append({
        "Degree": deg,
        "Train": train_mse,
        "Validation": val_mse,
        "Test": test_mse,
        "TestTRVA": test_trva_mse,
        "CV": cv_mse
    })

df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print(df.to_string(index=False))

# save to CSV
# df.to_csv("degree_mse_table.csv", index=False)
# print("\nSaved table to degree_mse_table.csv")

import matplotlib.pyplot as plt

# Continuous x grid for smooth curve plotting
x_plot = np.linspace(-3, 3, 256).reshape(-1, 1)

plt.figure(figsize=(10, 6))

# Scatter training data for reference
plt.scatter(X_train, y_train, color="black", s=25, label="Training data", alpha=0.6)

# Plot each polynomial fit
colors = plt.cm.viridis(np.linspace(0, 1, 9))  # one color per degree

for deg, color in zip(range(0, 9), colors):
    if deg == 0:
        # Constant model = mean of y_train
        y_plot = np.full_like(x_plot, fill_value=np.mean(y_train), dtype=float)
    else:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_poly, y_train)
        X_plot_poly = poly.transform(x_plot)
        y_plot = model.predict(X_plot_poly)
    plt.plot(x_plot, y_plot, color=color, lw=1.5, label=f"deg {deg}")

plt.xlabel("x")
plt.ylabel("ŷ")
plt.title("Polynomial regression fits for degrees 0–8")
plt.legend(ncol=3, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor  # 5th model example
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0

# load data
train = pd.read_csv("./data/train_real.csv")
test  = pd.read_csv("./data/test_real.csv")

# Separate features and target
y_train = train["Next_Tmax"].values
X_train = train.drop(columns=["Next_Tmax"]).values
y_test  = test["Next_Tmax"].values
X_test  = test.drop(columns=["Next_Tmax"]).values

# Standardise features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Define regressors
regressors = {
    "Dummy": DummyRegressor(strategy="mean"),
    "OLS": LinearRegression(),
    "RF": RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5)  # 5th model (can be changed to Ridge, XGB, etc.)
}

# Function to compute RMSE and CV RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cross_val_rmse(model, X, y, n_splits=10):
    # cross_val_score returns negative MSE when scoring="neg_mean_squared_error"
    scores = cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    return np.sqrt(-scores.mean())

# train and evaluate
rows = []
for name, model in regressors.items():
    # choose scaled or unscaled depending on model
    Xtr = X_train_scaled if name in ["SVR", "KNN"] else X_train
    Xte = X_test_scaled if name in ["SVR", "KNN"] else X_test

    model.fit(Xtr, y_train)
    train_rmse = rmse(y_train, model.predict(Xtr))
    test_rmse  = rmse(y_test,  model.predict(Xte))

    Xcv = X_train_scaled if name in ["SVR", "KNN"] else X_train
    cv_rmse = cross_val_rmse(model, Xcv, y_train)

    rows.append({"Regressor": name, "Train": train_rmse, "Test": test_rmse, "CV": cv_rmse})

results = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(results.to_string(index=False))

# save
# results.to_csv("regression_results_real.csv", index=False)
# print("\nSaved results to regression_results_real.csv")

