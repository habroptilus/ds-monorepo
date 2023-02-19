import datetime
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


def df_copy(func):
    """dfをコピーしてコピー元に及ばないようにするデコレーター."""

    def copy(_, df):
        df = df.copy()
        result = func(_, df)
        return result

    return copy


def stop_watch(func):
    """時間計測デコレータ."""

    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        td = datetime.timedelta(seconds=elapsed_time)
        hour, minute, second = get_h_m_s(td)
        elapsed_time = f"{second}s"
        if hour > 0:
            elapsed_time = f"{hour}h{minute}m" + elapsed_time
        elif minute > 0:
            elapsed_time = f"{minute}m" + elapsed_time
        print(f"Elapsed: {elapsed_time}")
        return result

    return wrapper


def get_h_m_s(td):
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


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
    plt.tight_layout()
    plt.xlabel("Importance")  # x軸のタイトル
    plt.ylabel("Feature")
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")


def plot_numeric_feature_hist_for_cls(train_df, test_df, feature_name, label_name):
    """ある数値特徴量の分布をtrain/test,ラベル種別にplotする."""
    plt.suptitle(feature_name)
    plt.subplot(121, title="train vs test")

    plt.hist(train_df[feature_name], density=True, alpha=0.6, label="train")
    plt.hist(test_df[feature_name], density=True, alpha=0.4, label="test")
    plt.legend()

    plt.subplot(122, title=f"{label_name} in train")
    labels = sorted(train_df[label_name].unique())
    for label in labels:
        plt.hist(train_df[train_df[label_name] == label][feature_name], density=True, alpha=0.5, label=str(label))

    plt.legend()
    plt.show()
