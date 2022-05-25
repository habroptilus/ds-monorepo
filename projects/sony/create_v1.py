import pandas as pd


def hoge(row):
    return f"{row['year']}{row['month']:02}{row['day']:02}"


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["yyyymmdd"] = train.apply(hoge, axis=1)
test["yyyymmdd"] = test.apply(hoge, axis=1)

train.to_csv("data/train_v1.csv", index=False)
test.to_csv("data/test_v1.csv", index=False)
