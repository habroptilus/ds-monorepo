"""v3からseとstdを抜く."""
import pandas as pd

train = pd.read_csv("data/train_v1.csv")
test = pd.read_csv("data/test_v1.csv")


stats_prefix_list = ["co", "no2", "so2", "o3", "temperature", "pressure", "humidity", "ws", "dew"]

for stats_prefix in stats_prefix_list:
    train[f"{stats_prefix}_rng"] = train[f"{stats_prefix}_max"] - train[f"{stats_prefix}_min"]
    test[f"{stats_prefix}_rng"] = test[f"{stats_prefix}_max"] - test[f"{stats_prefix}_min"]


train.to_csv("data/train_v4.csv", index=False)
test.to_csv("data/test_v4.csv", index=False)
