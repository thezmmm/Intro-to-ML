import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

path = './data/'
# problem1
# loda data
df = pd.read_csv(path+"train.csv")

# data info
print(df.head())
print(df.info())
print(df.describe())

# drop the columns "id" and "partlybad"
df = df.drop(columns=["id", "partlybad"])

print(df.columns)

# caculate mean and std of t84.mean
selected_columns = df[["T84.mean", "UV_A.mean", "CS.mean"]]
print(selected_columns.describe())

t84_array = df["T84.mean"].values
mean_t84 = np.mean(t84_array)
std_t84 = np.std(t84_array)
print("the mean of t84.mean",mean_t84)
print("the standard deviation of t84.mean",std_t84)

# task d
class4_counts = df["class4"].value_counts()
co242_data = df["CO242.mean"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Bar plot of class4
axes[0].bar(class4_counts.index, class4_counts.values, color='skyblue')
axes[0].set_title("Bar Plot of class4")
axes[0].set_xlabel("class4 categories")
axes[0].set_ylabel("Frequency")

# Histogram of CO242.mean
axes[1].hist(co242_data, bins=20, color='salmon', edgecolor='black')
axes[1].set_title("Histogram of CO242.mean")
axes[1].set_xlabel("CO242.mean")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# task e
vars_to_plot = ["UV_A.mean", "T84.mean", "H2O84.mean"]
data_subset = df[vars_to_plot]

sns.pairplot(data_subset)
plt.suptitle("Scatterplot Matrix of UV_A.mean, T84.mean, H2O84.mean", y=1.02)
plt.show()