import numpy as np

from lilac.core.data import Predictions
from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase, RegressorBase


class CrossValidationRunner:
    """Cross validationを行う.

    regression: oof_pred=予測値, oof_pred_proba=なし, predict=fold平均
    binary classification: oof_pred=予測クラス, oof_pred_proba=正例の確率, predict,predict_proba=fold平均
    multi classification: oof_pred=予測クラス, oof_pred_proba=全クラスの確率, predict=fold平均
    """

    def __init__(self, pred_oof, target_col, model_factory, model_params, trainer, evaluator):
        self.pred_oof = pred_oof
        self.target_col = target_col
        self.model_factory = model_factory
        self.model_params = model_params
        self.trainer = trainer
        self.evaluator = evaluator

    def run(self, labeled, folds):
        self.models = []
        if self.pred_oof:
            pred_valid_df = labeled.copy()
            pred_valid_df["oof_pred"] = None
            pred_valid_df["oof_raw_pred"] = None

        for i, (tdx, vdx) in enumerate(folds):
            print(f"Fold : {i+1}")
            # データ分割
            train, valid = labeled.iloc[tdx], labeled.iloc[vdx]

            # model作成
            model = self.model_factory.run(**self.model_params)

            # 訓練
            model = self.trainer.run(train, valid, self.model_factory, self.model_params)

            self.models.append(model)

            # validの予測値を出力
            if self.pred_oof:
                valid_pred = model.predict(valid)
                pred_valid_df.loc[vdx, "oof_pred"] = valid_pred

                valid_raw_pred = model.get_raw_pred(valid)
                if issubclass(model.__class__, MultiClassifierBase):
                    pred_valid_df.at[
                        vdx, [f"oof_raw_pred{i}" for i in range(valid_raw_pred.shape[1])]
                    ] = valid_raw_pred
                elif issubclass(model.__class__, RegressorBase) or issubclass(model.__class__, BinaryClassifierBase):
                    pred_valid_df.loc[vdx, "oof_raw_pred"] = valid_raw_pred
                else:
                    raise Exception(f"Invalid model class {model.__class__}")

        output = {}

        if self.pred_oof:
            output["oof_pred"] = pred_valid_df["oof_pred"].to_list()
            if issubclass(model.__class__, MultiClassifierBase):
                output["oof_raw_pred"] = pred_valid_df[
                    [f"oof_raw_pred{i}" for i in range(valid_raw_pred.shape[1])]
                ].values
            elif issubclass(model.__class__, RegressorBase) or issubclass(model.__class__, BinaryClassifierBase):
                output["oof_raw_pred"] = pred_valid_df["oof_raw_pred"].to_list()

        # 評価
        predictions = Predictions(pred=output["oof_pred"], raw_pred=output["oof_raw_pred"])
        output["evaluator"] = self.evaluator.return_flag()
        output["score"] = self.evaluator.run(labeled[self.target_col], predictions)

        return output

    def raw_output(self, test_df):
        """予測値の元になるscoreを出力する.
        regression : 予測値そのまま (レコード数,)
        binary class : 正例の確率 (レコード数,)
        multi class : 確率ベクトル (レコード数, クラス数)
        """
        preds = []
        for model in self.models:
            if issubclass(model.__class__, RegressorBase):
                pred = model.predict(test_df)
            elif issubclass(model.__class__, BinaryClassifierBase) or issubclass(model.__class__, MultiClassifierBase):
                pred = model.predict_proba(test_df)
            else:
                raise Exception(f"Invalid model class: {model.__class__}")
            preds.append(pred)
        return np.mean(preds, axis=0)

    def final_output(self, test_df):
        """予測値そのものを出力する.
        regression : 予測値そのまま (レコード数,)
        binary class : 0 or 1 (レコード数,)
        multi class : 0,...,num_class (レコード数,)
        """
        preds = self.raw_output(test_df)
        if len(preds.shape) == 1:
            if issubclass(self.models[0].__class__, RegressorBase):
                return preds
            elif issubclass(self.models[0].__class__, BinaryClassifierBase):
                return np.round(preds)
            else:
                raise Exception(f"Invalid model class: {self.models[0].__class__}")
        elif len(preds.shape) == 2:
            return np.argmax(preds, axis=1)
        else:
            raise Exception(f"Invalid output shape: {preds.shape}")

    def get_predictions(self, test_df):
        """raw_predとpredを作ってpredictionsを返す.

        内部的にはfinal_outputとraw_outputを呼び出すだけ.
        """
        return Predictions(list(self.final_output(test_df)), list(self.raw_output(test_df)))
