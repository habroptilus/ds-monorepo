"""inputが重複していてprice_logだけ違うレコードを削除する."""
import pandas as pd

train = pd.read_csv("data/train_v5.csv")
test = pd.read_csv("data/test_v5.csv")

# 重複削除
train = train[~train.drop(["id", "price_log"], axis=1).duplicated(keep=False)].reset_index(drop=True)

print(len(train))

train.to_csv("data/train_v6.csv", index=False)
test.to_csv("data/test_v6.csv", index=False)
