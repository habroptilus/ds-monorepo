"""v3で追加していた特徴量をv5にいれてもう一度試す."""
import pandas as pd

train = pd.read_csv("data/train_v5.csv")
test = pd.read_csv("data/test_v5.csv")


def layout2rooms(text):
    if type(text) is float:
        return None
    if text in ["オープンフロア", "スタジオ", "メゾネット"]:
        return None
    x = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).replace("+", "")
    str2room = {"R": 0, "K": 0, "D": 1, "L": 0.5, "S": 1}
    return int(x[0]) + sum([str2room[c] for c in x[1:]])


train["rooms"] = train["layout"].apply(layout2rooms)
test["rooms"] = test["layout"].apply(layout2rooms)

train["building_area"] = train["area"] * train["coverage_ratio"] / 100
train["total_floor_area"] = train["area"] * train["floor_ratio"] / 100
train["floors"] = train["total_floor_area"] / train["building_area"]
train["room_size"] = train["total_floor_area"] / train["rooms"]

test["building_area"] = test["area"] * test["coverage_ratio"] / 100
test["total_floor_area"] = test["area"] * test["floor_ratio"] / 100
test["floors"] = test["total_floor_area"] / test["building_area"]
test["room_size"] = test["total_floor_area"] / test["rooms"]


train.to_csv("data/train_v7.csv", index=False)
test.to_csv("data/test_v7.csv", index=False)
