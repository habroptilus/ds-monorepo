"""各統計量にrng(max-min)を追加する."""
import pandas as pd

train = pd.read_csv("data/train_v2.csv")
test = pd.read_csv("data/test_v2.csv")


stats_prefix_list = ["co", "no2", "so2", "o3", "temperature", "pressure", "humidity", "ws", "dew"]

for stats_prefix in stats_prefix_list:
    train[f"{stats_prefix}_rng"] = train[f"{stats_prefix}_max"] - train[f"{stats_prefix}_min"]
    test[f"{stats_prefix}_rng"] = test[f"{stats_prefix}_max"] - test[f"{stats_prefix}_min"]


train.to_csv("data/train_v3.csv", index=False)
test.to_csv("data/test_v3.csv", index=False)
