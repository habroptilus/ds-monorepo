from tools.core.blocks import BlocksRunner
from abc import ABCMeta, abstractmethod
import pandas as pd


class SeedJobBase(metaclass=ABCMeta):
    """SeedJobの基底クラス.学習データ生成とモデリング、評価を行い結果をdictで返す."""

    @abstractmethod
    def run(self, train, test):
        pass


class BasicSeedJob:
    """基本的なSeedJob.

    以下を実行する.
    :特徴量生成
    :学習/テストデータ生成
    :モデル学習
    :評価
    """

    def __init__(self, train_path, test_path, features_dir, register_from, features_settings, target_col, unused_cols, seed,
                 fold_num, group_key_col, depth, n_estimators, trainer_str, model_str, folds_gen_str, evaluator_str):
        # TODO: 各settingsのパラメータを全部持ってくる

        trainer_factory_settings = {
            "model_str": trainer_str,
            "params": {
                "target_col": target_col,
                "seed": seed
            }
        }

        folds_gen_factory_settings = {
            "model_str": folds_gen_str,  # kfold, stratified, group, stratified_group
            "params": {
                "fold_num": fold_num,
                "seed": seed,
                "target_col": target_col,
                "key_col": group_key_col
            }
        }

        model_factory_settings = {
            "model_str": model_str,
            "params": {
                "depth": depth,
                "n_estimators": n_estimators,
                "seed": seed
            }
        }

        self.blocks_runner = BlocksRunner(target_col=target_col, features_dir=features_dir, register_from=register_from, features_settings=features_settings,
                                          unused_cols=unused_cols, folds_gen_factory_settings=folds_gen_factory_settings,
                                          model_factory_settings=model_factory_settings, trainer_factory_settings=trainer_factory_settings,
                                          evaluator_str=evaluator_str)

        self.train_path = train_path
        self.test_path = test_path

    def run(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        output = self.blocks_runner.run(train, test)
        return output
