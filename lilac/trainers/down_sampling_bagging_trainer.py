import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from lilac.models.model_base import MultiClassifierBase
from lilac.trainers.trainer_base import TrainerBase


class DownSamplingBaggingTrainer(TrainerBase):
    """downsampling+baggingを行う"""

    def __init__(self, target_col, base_class, bagging_num=5, seed=None, allow_less_than_base=True):
        self.target_col = target_col
        self.bagging_num = bagging_num
        self.sampler = ImbalancedClassDownSampler(target_col, base_class, seed, allow_less_than_base)
        self.base_class = base_class
        self.seed = seed

    def run(self, train, valid, model_factory, model_params):
        models = []
        for i in range(self.bagging_num):
            print(f"Bagging {i+1}")

            sampled_train = self.sampler.run(train, self.seed + i)

            # 複数モデルを作るためmodel_factoryをtrainer側で叩く必要がある.
            model = model_factory.run(**model_params)
            model.fit(sampled_train, valid)
            models.append(model)

        return self.aggregate_models(models)

    def aggregate_models(self, models):
        """baggingで作成した複数モデルをまとめる.現状は多クラス分類にのみ対応している."""
        # もしかしたら統一的に書けるかもだけどとりあえず
        # TODO 2クラス分類にも対応する.
        if issubclass(models[0].__class__, MultiClassifierBase):
            return MultiClassifiersAggregator(self.target_col, models)
        else:
            raise Exception(f"Not supported model class: {models[0].__class__}")

    def return_flag(self):
        return f"dsbt_{self.bagging_num}_{self.base_class}_{self.seed}"


class MultiClassifiersAggregator(MultiClassifierBase):
    """多クラス分類器をまとめたモデル.
    子モデルのpredict_probaで出力した確率を平均する.
    """

    def __init__(self, target_col, models):
        self.models = models
        super().__init__(target_col)

    def _predict_proba(self, df):
        """出力はクラス数分の次元でクラスごとの予測確率を想定."""
        preds = []
        for model in self.models:
            preds.append(model.predict_proba(df))
        preds = np.array(preds)  # (bagging_num, レコード数, num_class)
        return np.mean(preds, axis=0)

    def get_importance(self):
        # TODO  子モデルがlgbmじゃないと動かないのでどうにかする.
        dfs = [model.get_importance() for model in self.models]
        hoge = pd.concat(dfs, axis=1)
        imp_cols = [f"importance_bagging{i}" for i in range(len(dfs))]
        hoge.columns = imp_cols
        hoge["importance"] = hoge.mean(axis=1)
        return hoge[["importance"]]


class ImbalancedClassDownSampler:
    """不均衡データに対してダウンサンプリングをする.
    base_classに指定したクラスと同じデータ数に他のクラスをダウンサンプリングする.

    allow_less_than_base
    データ数がbase_classに満たないクラスがある場合を許容するか。
    Trueの場合、足りないまま用いる
    Falseの場合はエラーにする.
    """

    def __init__(self, target_col, base_class, allow_less_than_base=False):
        self.target_col = target_col
        self.base_class = base_class
        self.allow_less_than_base = allow_less_than_base

    def run(self, data, seed):
        X, y = self.split_df2xy(data)
        sample_ratio = self.get_sample_ratio(y)
        sampler = RandomUnderSampler(sampling_strategy=sample_ratio, random_state=seed)
        sampled_x, sampled_y = sampler.fit_resample(X, y)
        sampled_x[self.target_col] = sampled_y
        return sampled_x

    def get_sample_ratio(self, y):
        counter_classes = y.value_counts()
        base_class_num = counter_classes[self.base_class]
        less_than_classes = counter_classes[counter_classes < base_class_num]
        if len(less_than_classes) > 0:
            if self.allow_less_than_base:
                return {a: base_class_num for a in counter_classes.index if a not in less_than_classes.index}
            else:
                raise Exception("Not allowed class less than base.")
        return {a: base_class_num for a in counter_classes.index if a not in less_than_classes.index}

    def split_df2xy(self, df):
        """xとyに分ける."""
        y = df[self.target_col]
        X = df.drop(self.target_col, axis=1)
        return X, y
