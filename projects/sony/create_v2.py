"""各統計量にstdとse(標準誤差)を追加する."""
import numpy as np
import pandas as pd

train = pd.read_csv("data/train_v1.csv")
test = pd.read_csv("data/test_v1.csv")


stats_prefix_list = ["co", "no2", "so2", "o3", "temperature", "pressure", "humidity", "ws", "dew"]

for stats_prefix in stats_prefix_list:
    train[f"{stats_prefix}_std"] = np.sqrt(train[f"{stats_prefix}_var"])
    train[f"{stats_prefix}_se"] = train[f"{stats_prefix}_std"] / np.sqrt(train[f"{stats_prefix}_cnt"])

    test[f"{stats_prefix}_std"] = np.sqrt(test[f"{stats_prefix}_var"])
    test[f"{stats_prefix}_se"] = test[f"{stats_prefix}_std"] / np.sqrt(test[f"{stats_prefix}_cnt"])


train.to_csv("data/train_v2.csv", index=False)
test.to_csv("data/test_v2.csv", index=False)
