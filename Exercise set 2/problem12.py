import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import glob

# logistic function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# Optimal Bayes probability
def optimal_bayes_prob(X):
    x1, x2 = X[:,0], X[:,1]
    t = 0.1 - 2*x1 + x2 + 0.2*x1*x2
    return sigmoid(t)

# Dummy classifier probability
def dummy_prob(y_train):
    p = y_train.mean()
    return lambda X: np.full(shape=(X.shape[0],), fill_value=p)

# Compute perplexity from probabilities
def perplexity(y_true, y_prob):
    # Avoid log(0)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1-eps)
    return np.exp(log_loss(y_true, y_prob))


def evaluate_classifiers(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    X_train = train[['x1', 'x2']].values
    y_train = train['y'].values

    X_test = test[['x1', 'x2']].values
    y_test = test['y'].values

    results = {}

    # --- NB ---
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    p_nb = nb.predict_proba(X_test)[:, 1]
    results['NB'] = (accuracy_score(y_test, p_nb >= 0.5), perplexity(y_test, p_nb))

    # --- Logistic Regression without interaction ---
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    results['LR'] = (accuracy_score(y_test, p_lr >= 0.5), perplexity(y_test, p_lr))

    # --- Logistic Regression with interaction term ---
    X_train_i = np.hstack([X_train, (X_train[:, 0] * X_train[:, 1]).reshape(-1, 1)])
    X_test_i = np.hstack([X_test, (X_test[:, 0] * X_test[:, 1]).reshape(-1, 1)])

    lri = LogisticRegression()
    lri.fit(X_train_i, y_train)
    p_lri = lri.predict_proba(X_test_i)[:, 1]
    results['LRi'] = (accuracy_score(y_test, p_lri >= 0.5), perplexity(y_test, p_lri))

    # --- Optimal Bayes ---
    p_opt = optimal_bayes_prob(X_test)
    results['OptimalBayes'] = (accuracy_score(y_test, p_opt >= 0.5), perplexity(y_test, p_opt))

    # --- Dummy ---
    p_dummy_func = dummy_prob(y_train)
    p_dummy = p_dummy_func(X_test)
    results['Dummy'] = (accuracy_score(y_test, p_dummy >= 0.5), perplexity(y_test, p_dummy))

    return results

train_files = sorted(glob.glob('./data_E2/toy_train_*.csv'))
test_file = './data_E2/toy_test.csv'

summary = []

for f in train_files:
    n = int(f.split('_')[-1].split('.csv')[0])
    res = evaluate_classifiers(f, test_file)
    summary.append({
        'n': n,
        'NB_acc': res['NB'][0], 'NB_ppx': res['NB'][1],
        'LR_acc': res['LR'][0], 'LR_ppx': res['LR'][1],
        'LRi_acc': res['LRi'][0], 'LRi_ppx': res['LRi'][1],
        'OptimalBayes_acc': res['OptimalBayes'][0], 'OptimalBayes_ppx': res['OptimalBayes'][1],
        'Dummy_acc': res['Dummy'][0], 'Dummy_ppx': res['Dummy'][1]
    })

df_summary = pd.DataFrame(summary).sort_values('n')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)
print(df_summary)
