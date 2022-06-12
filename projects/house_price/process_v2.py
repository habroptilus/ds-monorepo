

import pandas as pd

train = pd.read_csv("data/train_v1.csv")
test = pd.read_csv("data/test_v1.csv")

train["city_l1"] = train["city"].apply(lambda x: x.split("市")[0] if "市" in x else None)
test["city_l1"] = test["city"].apply(lambda x: x.split("市")[0] if "市" in x else None)


def wareki2seireki(x):
    if type(x) == float:
        return None
    if x == "戦前":
        return 1945  # 昭和のminが1947だった
    else:
        d = {"昭和": 1926, "平成": 1989, "令和": 2019}
        return d[x[:2]] + int(x[2:-1])


train["built_year_seireki"] = train["built_year"].apply(lambda x: wareki2seireki(x))
test["built_year_seireki"] = test["built_year"].apply(lambda x: wareki2seireki(x))

train["layout_l1"] = train["layout"].apply(lambda x: x.replace("＋Ｓ", "").replace("＋Ｋ", "") if type(x) != float else x)
test["layout_l1"] = test["layout"].apply(lambda x: x.replace("＋Ｓ", "").replace("＋Ｋ", "") if type(x) != float else x)

train["ordered_year"] = train["ordered"].apply(lambda x: int(x[:4]))
test["ordered_year"] = test["ordered"].apply(lambda x: int(x[:4]))

train["ordered_quarter"] = train["ordered"].apply(lambda x: int(x[6]))
test["ordered_quarter"] = test["ordered"].apply(lambda x: int(x[6]))

train["age"] = train["ordered_year"] - train["built_year_seireki"]
test["age"] = test["ordered_year"] - test["built_year_seireki"]


def structure_multihot(x, key):
    if type(x) == float:
        return 0
    if key in x.split("、"):
        return 1
    return 0


keys = ["ＲＣ", "ＳＲＣ", "鉄骨造", "木造", "軽量鉄骨造", "ブロック造"]
for key in keys:
    train[key] = train["structure"].apply(structure_multihot, key=key)
    test[key] = test["structure"].apply(structure_multihot, key=key)

train["layout_+s"] = train["layout"].apply(lambda x: 1 if type(x) != float and "＋Ｓ" in x else 0)
train["layout_+k"] = train["layout"].apply(lambda x: 1 if type(x) != float and "＋Ｋ" in x else 0)
test["layout_+s"] = test["layout"].apply(lambda x: 1 if type(x) != float and "＋Ｓ" in x else 0)
test["layout_+k"] = test["layout"].apply(lambda x: 1 if type(x) != float and "＋Ｋ" in x else 0)

train = train.drop("city_code", axis=1)
test = test.drop("city_code", axis=1)


train.to_csv("data/train_v2.csv", index=False)
test.to_csv("data/test_v2.csv", index=False)
