import pandas as pd

train = pd.read_csv("data/train_v2.csv")
test = pd.read_csv("data/test_v2.csv")


def make_city(row):
    return f"{row['prefecture']}{row['city']}"


def make_district(row):
    return f"{row['city']}{row['district']}"


train["city"] = train.apply(make_city, axis=1)
train["district"] = train.apply(make_district, axis=1)

test["city"] = test.apply(make_city, axis=1)
test["district"] = test.apply(make_district, axis=1)

train.to_csv("data/train_v4.csv", index=False)
test.to_csv("data/test_v4.csv", index=False)
