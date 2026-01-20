import pandas as pd
import statsmodels.api as sm

d1 = pd.read_csv('./data/d1.csv')
d2 = pd.read_csv('./data/d2.csv')
d3 = pd.read_csv('./data/d3.csv')
d4 = pd.read_csv('./data/d4.csv')

def fit_ols(df):
    X = sm.add_constant(df['x'])  # adds intercept term w0
    y = df['y']
    model = sm.OLS(y, X).fit()
    return model

models = [fit_ols(d) for d in [d1, d2, d3, d4]]

# for i, model in enumerate(models, 1):
#     print(f"Dataset d{i}:")
#     print(model.summary())
#     print("\n")

for i, model in enumerate(models, 1):
    print(f"Dataset d{i}:")
    print(f"Intercept (w0): {model.params[0]:.4f}, SE={model.bse[0]:.4f}, p={model.pvalues[0]:.4f}")
    print(f"Slope (w1): {model.params[1]:.4f}, SE={model.bse[1]:.4f}, p={model.pvalues[1]:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_regression(df, model, dataset_name):
    plt.figure(figsize=(6, 4))

    # Scatter plot of the data
    plt.scatter(df['x'], df['y'], color='blue', label='Data points')

    # Regression line
    X_range = pd.DataFrame({'x': sorted(df['x'])})
    X_range = sm.add_constant(X_range)  # add intercept
    y_pred = model.predict(X_range)
    plt.plot(sorted(df['x']), y_pred, color='red', label='Fitted line')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{dataset_name}: Scatterplot & Regression Line')
    plt.legend()
    plt.show()

datasets = [d1, d2, d3, d4]
for i, (df, model) in enumerate(zip(datasets, models), 1):
    plot_regression(df, model, f'd{i}')


def diagnostic_plot(model, dataset_name):
    residuals = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title(f'{dataset_name}: Residuals vs Fitted')
    plt.show()


# Apply to each dataset
for i, model in enumerate(models, 1):
    diagnostic_plot(model, f'd{i}')




