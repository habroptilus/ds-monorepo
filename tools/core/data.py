from dataclasses import dataclass


@dataclass
class Predictions:
    pred: list  # 最終的な予測. 分類の場合は予測クラス.
    raw_pred: list  # 分類の場合はクラスの所属確率, 回帰の場合はpredと同じ
