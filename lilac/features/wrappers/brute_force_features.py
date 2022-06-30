import itertools

import pandas as pd

from lilac.features.generator_base import _FeaturesBase


class BruteForceFeatures(_FeaturesBase):
    """複数のパラメータについて複数候補を渡したときに、全組み合わせで特徴量生成クラスを内側で呼ぶ.

    保存は内側のみ(一つずつ)行われる.)
    """

    def __init__(self, params, FeaturesClass, **kwargs):
        """
        params = {
            "input_col": ["age","area"]
            "group_key": ["city", "layout"]
            "agg_func": ["max", "mean"]
        }
        """
        self.FeaturesClass = FeaturesClass
        self.features_generators = []
        # python3.7以上なので順序が保存されている
        col_names = params.keys()
        for tmp in itertools.product(*params.values()):
            tmp = {col_name: param for col_name, param in zip(col_names, tmp)}
            self.features_generators.append(FeaturesClass(**tmp, **kwargs))
        self.md5 = self._calc_md5({f"gen_{i}": gen.md5 for i, gen in enumerate(self.features_generators)})

    def run(self, train, test):
        train_res_list = []
        test_res_list = []
        for gen in self.features_generators:
            train_res, test_res = gen.run(train, test)
            train_res_list.append(train_res)
            test_res_list.append(test_res)
        train_res = pd.concat(train_res_list, axis=1)
        test_res = pd.concat(test_res_list, axis=1)
        return train_res, test_res

    def fit(self, df):
        for gen in self.features_generators:
            # 参照渡しなはず...
            gen.fit(df)
        return self

    def transform(self, df):
        features = []
        for gen in self.features_generators:
            feature = gen.transform(df)

            features.append(feature)
        return pd.concat(features, axis=1)
