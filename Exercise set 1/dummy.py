import pandas as pd

path = './data/'

train_df = pd.read_csv(path+"train.csv")

most_common_class = train_df["class4"].mode()[0]

event_probability = (train_df["class4"] != "nonevent").mean()

print(f"Most common class: {most_common_class}")
print(f"Event probability: {event_probability:.4f}")

test_df = pd.read_csv(path+"test.csv")

test_df["class4"] = most_common_class
test_df["p"] = event_probability

submission = test_df[["id", "class4", "p"]]

submission.to_csv("dummy_submission.csv", index=False)
print("Dummy submission saved as dummy_submission.csv")
