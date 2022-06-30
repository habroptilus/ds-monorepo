"""built_yearの修正、重複削除、area,nearest_min,ageのlogをとる."""
import numpy as np
import pandas as pd

train = pd.read_csv("data/train_v4.csv")
test = pd.read_csv("data/test_v4.csv")


def wareki2seireki(x):
    if type(x) == float:
        return None
    if x == "戦前":
        return 1945  # 昭和のminが1947だった
    else:
        d = {"昭和": 1925, "平成": 1988, "令和": 2018}
        return d[x[:2]] + int(x[2:-1])


train["built_year_seireki"] = train["built_year"].apply(lambda x: wareki2seireki(x))
test["built_year_seireki"] = test["built_year"].apply(lambda x: wareki2seireki(x))

train["age"] = train["ordered_year"] - train["built_year_seireki"]
test["age"] = test["ordered_year"] - test["built_year_seireki"]

train.loc[train["age"] < 0, "age"] = None
test.loc[test["age"] < 0, "age"] = None

# price_logの底をそろえてlog10
train["area_log"] = train["area"].apply(lambda x: np.log10(x))
test["area_log"] = test["area"].apply(lambda x: np.log10(x))

train["nearest_min_log"] = train["nearest_min"].apply(lambda x: np.log1p(x))
test["nearest_min_log"] = test["nearest_min"].apply(lambda x: np.log1p(x))

train["age_log"] = train["age"].apply(lambda x: np.log1p(x))
test["age_log"] = test["age"].apply(lambda x: np.log1p(x))

# 重複削除
train = train[~train.drop("id", axis=1).duplicated()].reset_index(drop=True)


train.to_csv("data/train_v5.csv", index=False)
test.to_csv("data/test_v5.csv", index=False)
