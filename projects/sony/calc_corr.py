import re

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from scipy.spatial.distance import pdist, squareform


def make_corr_array(df, cols):
    output = pd.DataFrame(1 - squareform(pdist(df[cols].T, "correlation")), columns=cols, index=cols)
    return output


train = pd.read_csv("data/train_v4.csv")
test = pd.read_csv("data/test_v4.csv")


corr_df = make_corr_array(train, [c for c in train.columns if re.search(r"mid|min|max|cnt|lat|lon|se|std|rng", c)])
# fig, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corr_df, square=True, vmax=1, vmin=-1, center=0)


corr_top10 = (
    corr_df.loc[:, ["pm25_mid"]]
    .assign(abs_value=lambda d: np.abs(d["pm25_mid"]))
    .sort_values("abs_value", ascending=False)
    .iloc[1:11, :]
    .index.tolist()
)

for col in corr_top10:
    print(f"- {col}")
