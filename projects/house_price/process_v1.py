import glob

import pandas as pd

train_path_list = glob.glob("data/train/*.csv")

train_df_list = []
for p in train_path_list:
    train_df_list.append(pd.read_csv(p))

train = pd.concat(train_df_list)
test = pd.read_csv("data/test.csv")

# 全てnoneか一種類しかない無意味なカラムを削除
drop_cols = ["土地の形状", "間口", "延床面積（㎡）", "前面道路：方位", "前面道路：種類", "前面道路：幅員（ｍ）", "種類", "地域"]
train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)

# 英語にカラムを変換
rename_dict = {
    "ID": "id",
    "市区町村コード": "city_code",
    "都道府県名": "prefecture",
    "市区町村名": "city",
    "地区名": "district",
    "最寄駅：名称": "nearest_sta",
    "最寄駅：距離（分）": "nearest_min",
    "間取り": "layout",
    "面積（㎡）": "area",
    "建築年": "built_year",
    "建物の構造": "structure",
    "用途": "usage",
    "今後の利用目的": "purpose",
    "都市計画": "plan",
    "建ぺい率（％）": "coverage_ratio",
    "容積率（％）": "floor_ratio",
    "取引時点": "ordered",
    "改装": "reform",
    "取引の事情等": "note",
    "取引価格（総額）_log": "price_log",
}
train = train.rename(columns=rename_dict)
test = test.rename(columns=rename_dict)

# idをstrに
train["id"] = train["id"].apply(lambda x: f"id_{x}")
train["city_code"] = train["city_code"].apply(lambda x: f"id_{x}")
test["id"] = test["id"].apply(lambda x: f"id_{x}")
test["city_code"] = test["city_code"].apply(lambda x: f"id_{x}")

# areaを数値に. 2000以上は全部2000に置き換える(もともと2000のものはなかった)


def truncate_area(x):
    if x == "2000㎡以上":
        return 2000
    else:
        return float(x)


train["area"] = train["area"].apply(lambda x: truncate_area(x)).astype(float)
test["area"] = test["area"].apply(lambda x: truncate_area(x)).astype(float)


# nearest_minを数値に変換する
# 30分以上のものは範囲になっているので、その中で最小のものに置換する.

min_map = {"30分?60分": "30", "30分?60分": "30", "1H30?2H": "90", "1H?1H30": "60", "2H?": "120"}
train["nearest_min"] = train["nearest_min"].apply(lambda x: min_map.get(x, x)).astype(float)
test["nearest_min"] = test["nearest_min"].apply(lambda x: min_map.get(x, x)).astype(float)


train.to_csv("data/train_v1.csv", index=False)
test.to_csv("data/test_v1.csv", index=False)
