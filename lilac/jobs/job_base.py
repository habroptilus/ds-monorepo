import pandas as pd

from lilac.core.blocks import BlocksRunner
from lilac.core.utils import stop_watch
from lilac.ensemble.stacking_runner import StackingRunner
from lilac.models import consts


class BasicSeedJob:
    """基本的なSeedJob.

    以下を実行する.
    :特徴量生成
    :学習/テストデータ生成
    :モデル学習
    :評価
    """

    def __init__(
        self,
        target_col,
        model_str,
        evaluator_str,
        train_path,
        test_path,
        features_dir,
        register_from=None,
        extra_class_names=None,
        features_settings=None,
        unused_cols=None,
        folds_gen_str="kfold",
        fold_num=5,
        group_key_col="group_key_col",
        trainer_str="basic",
        base_class=3,
        bagging_num=5,
        allow_less_than_base=True,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        min_child_samples=consts.min_child_samples,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
        class_weight=consts.class_weight,
        base_col=None,
        log_target_on_target_enc=False,
        target_enc_cols=None,
        model_dir=None,
        pred_only=False,
    ):
        features_settings = [] if features_settings is None else features_settings
        unused_cols = [] if unused_cols is None else unused_cols

        trainer_factory_settings = {
            "model_str": trainer_str,
            "params": {
                "target_col": target_col,
                "seed": seed,
                "base_class": base_class,
                "bagging_num": bagging_num,
                "allow_less_than_base": allow_less_than_base,
            },
        }

        folds_gen_factory_settings = {
            "model_str": folds_gen_str,  # kfold, stratified, group, stratified_group
            "params": {"fold_num": fold_num, "seed": seed, "target_col": target_col, "key_col": group_key_col},
        }

        model_factory_settings = {
            "model_str": model_str,
            "params": {
                # 共通
                "depth": depth,
                "n_estimators": n_estimators,
                "seed": seed,
                "early_stopping_rounds": early_stopping_rounds,
                # lgbm
                "verbose_eval": verbose_eval,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "subsample": subsample,
                "min_child_weight": min_child_weight,
                "min_child_samples": min_child_samples,
                # catb
                "learning_rate": learning_rate,
                "random_strength": random_strength,
                "bagging_temperature": bagging_temperature,
                "od_type": od_type,
                "od_wait": od_wait,
                # lgbm and catb's classifier
                "class_weight": class_weight,
                # diff model
                "base_col": base_col,
            },
        }

        self.blocks_runner = BlocksRunner(
            target_col=target_col,
            features_dir=features_dir,
            register_from=register_from,
            extra_class_names=extra_class_names,
            features_settings=features_settings,
            unused_cols=unused_cols,
            folds_gen_factory_settings=folds_gen_factory_settings,
            model_factory_settings=model_factory_settings,
            trainer_factory_settings=trainer_factory_settings,
            evaluator_str=evaluator_str,
            log_target_on_target_enc=log_target_on_target_enc,
            target_enc_cols=target_enc_cols,
            seed=seed,
            model_dir=model_dir,
            pred_only=pred_only,
        )

        self.train_path = train_path
        self.test_path = test_path

    @stop_watch
    def run(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return self.blocks_runner.run(train, test)


class StackingJob:
    """stackingを行う.パラメータを揃えてStackingRunnerを走らせるだけ.

    stackingの他の実装をしたくなることがなさそうなのでベースクラスはつくらずにこのまま.
    """

    def __init__(
        self,
        target_col,
        stacking_settings,
        train_path,
        test_path,
        unused_cols=None,
        trainer_str="basic",
        seed=None,
        base_class=3,
        bagging_num=5,
        allow_less_than_base=True,
        folds_gen_str="kfold",
        fold_num=5,
        group_key_col="group_key_col",
        use_original_cols=False,
    ):

        unused_cols = [] if unused_cols is None else unused_cols

        trainer_factory_settings = {
            "model_str": trainer_str,
            "params": {
                "target_col": target_col,
                "seed": seed,
                "base_class": base_class,
                "bagging_num": bagging_num,
                "allow_less_than_base": allow_less_than_base,
            },
        }

        folds_gen_factory_settings = {
            "model_str": folds_gen_str,  # kfold, stratified, group, stratified_group
            "params": {"fold_num": fold_num, "seed": seed, "target_col": target_col, "key_col": group_key_col},
        }
        shared = {
            "target_col": target_col,
            "unused_cols": unused_cols,
            "folds_gen_factory_settings": folds_gen_factory_settings,
            "trainer_factory_settings": trainer_factory_settings,
            "use_original_cols": use_original_cols,
        }

        self.stacking_runner = StackingRunner(stacking_settings, shared)

        self.train_path = train_path
        self.test_path = test_path

    def run(self, output_list):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return self.stacking_runner.run(output_list, train, test)
