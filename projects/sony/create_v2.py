import pandas as pd

train = pd.read_csv("data/train_v1.csv")
test = pd.read_csv("data/test_v1.csv")

train["dt"] = pd.to_datetime(train["yyyymmdd"], format="%Y%m%d")
test["dt"] = pd.to_datetime(test["yyyymmdd"], format="%Y%m%d")

train.to_csv("data/train_v2.csv", index=False)
test.to_csv("data/test_v2.csv", index=False)
