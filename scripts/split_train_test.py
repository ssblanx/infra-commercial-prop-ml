import pandas as pd

df = pd.read_csv("../data/raw/sberbank_train_raw.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

cut = int(len(df) * 0.8)
train = df.iloc[:cut]
test = df.iloc[cut:]

train.to_csv("../data/train_pr3.csv", index=False)
test.to_csv("../data/test_holdout_pr3.csv", index=False)

print("train:", train.shape, train["timestamp"].min(), train["timestamp"].max())
print("test:", test.shape, test["timestamp"].min(), test["timestamp"].max())
