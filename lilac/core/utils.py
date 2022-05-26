import matplotlib.pyplot as plt
import numpy as np


def df_copy(func):
    """dfをコピーしてコピー元に及ばないようにするデコレーター."""

    def copy(_, df):
        df = df.copy()
        result = func(_, df)
        return result

    return copy


def plot_feature_importance(df, path=None, max_n=20):
    """特徴量重要度をplotする.

    importanceカラムがあればどうさする.
    """
    plt.figure()
    # 特徴量数(説明変数の個数)
    n = len(df)
    df_plot = df.sort_values("importance")
    df_plot = df_plot.iloc[max(n - max_n, 0) :]  # 上位max_n個だけ表示
    n_features = len(df_plot)
    f_importance_plot = df_plot["importance"].values  # 特徴量重要度の取得
    plt.barh(range(n_features), f_importance_plot, align="center")
    cols_plot = df_plot.index  # 特徴量の取得
    plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定
    plt.xlabel("Importance")  # x軸のタイトル
    plt.ylabel("Feature")
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")
