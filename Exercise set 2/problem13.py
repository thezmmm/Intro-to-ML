import pandas as pd
import numpy as np

# ------------------------
# Gini impurity
# ------------------------
def gini(y):
    if len(y) == 0:
        return 0
    p = np.mean(y == 1)
    return 1 - p**2 - (1 - p)**2

# ------------------------
# Find best split for a dataset
# ------------------------
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = 1.0
    n_samples, n_features = X.shape

    for f in range(n_features):
        values = X[:, f]
        # candidate thresholds = midpoints between sorted unique values
        sorted_vals = np.unique(values)
        if len(sorted_vals) == 1:
            continue
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

        for t in thresholds:
            left_idx = values < t
            right_idx = ~left_idx

            g_left = gini(y[left_idx])
            g_right = gini(y[right_idx])

            w_left = np.sum(left_idx) / n_samples
            w_right = 1 - w_left

            g_split = w_left * g_left + w_right * g_right

            if g_split < best_gini:
                best_gini = g_split
                best_feature = f
                best_threshold = t

    return best_feature, best_threshold, best_gini

# ------------------------
# Build the tree recursively
# ------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gini=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class for leaf
        self.gini = gini

def build_tree(X, y):
    # if pure, or no split possible
    if len(np.unique(y)) == 1:
        return Node(value=y[0], gini=0)

    feature, threshold, g_split = best_split(X, y)

    # no valid split found (all values identical)
    if feature is None:
        # majority vote leaf
        value = 1 if np.mean(y) >= 0.5 else 0
        return Node(value=value, gini=gini(y))

    # split
    left_idx = X[:, feature] < threshold
    right_idx = ~left_idx

    left = build_tree(X[left_idx], y[left_idx])
    right = build_tree(X[right_idx], y[right_idx])

    return Node(feature=feature, threshold=threshold, left=left, right=right, gini=g_split)

# ------------------------
# Pretty print the tree
# ------------------------
def print_tree(node, depth=0):
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: class={node.value}, gini={node.gini:.3f}")
        return

    print(f"{indent}Node: x{node.feature+1} < {node.threshold:.4f}  (gini={node.gini:.3f})")
    print(f"{indent} Left:")
    print_tree(node.left, depth + 1)
    print(f"{indent} Right:")
    print_tree(node.right, depth + 1)

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    df = pd.read_csv("toy_train_8.csv")  # <-- change file here
    X = df[["x1", "x2"]].values
    y = df["y"].values

    tree = build_tree(X, y)
    print_tree(tree)
