import glob
import pickle
from pathlib import Path

import numpy as np

from lilac.core.data import Predictions
from lilac.features.target_encoders.target_encoder import TargetEncoder
from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase, RegressorBase


def load_model(model_path):
    print(f"Loaded from {model_path}")
    return pickle.load(open(model_path, "rb"))


class CrossValidationRunner:
    """Cross validationを行う.

    regression: oof_pred=予測値, oof_pred_proba=なし, predict=fold平均
    binary classification: oof_pred=予測クラス, oof_pred_proba=正例の確率, predict,predict_proba=fold平均
    multi classification: oof_pred=予測クラス, oof_pred_proba=全クラスの確率, predict=fold平均
    """

    def __init__(
        self,
        pred_oof,
        target_col,
        model_factory,
        model_params,
        trainer,
        evaluator,
        folds_generator,
        unused_cols=None,
        target_enc_cols=None,
        seed=None,
        log_target_on_target_enc=False,
        model_dir=None,
    ):
        self.pred_oof = pred_oof
        self.target_col = target_col
        self.model_factory = model_factory
        self.model_params = model_params
        self.trainer = trainer
        self.evaluator = evaluator
        self.folds_generator = folds_generator
        self.unused_cols = unused_cols or []
        self.target_enc_cols = target_enc_cols or []
        self.seed = seed
        self.log_target_on_target_enc = log_target_on_target_enc
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.models = None
        self.encoders = None

    def run(self, df):
        self.models = []
        if self.pred_oof:
            pred_valid_df = df.copy()
            pred_valid_df["oof_pred"] = None
            pred_valid_df["oof_raw_pred"] = None

        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        folds = self.folds_generator.run(df)
        df = df.drop(self.unused_cols, axis=1)
        print(f"Data: {len(df)}")
        self.encoders = []
        additionals = []
        for fold, (tdx, vdx) in enumerate(folds):
            print(f"Fold : {fold+1}")
            # split
            train, valid = df.iloc[tdx].reset_index(drop=True), df.iloc[vdx].reset_index(drop=True)

            # Target encoding
            if self.target_enc_cols:
                target_encoder = TargetEncoder(
                    input_cols=self.target_enc_cols,
                    target_col=self.target_col,
                    random_state=self.seed,
                    log_target=self.log_target_on_target_enc,
                )
                train = target_encoder.fit_transform(train)
                valid = target_encoder.transform(valid)
                self.encoders.append(target_encoder)

            # Train
            model = self.trainer.run(train, valid, self.model_factory, self.model_params)
            self.models.append(model)
            additionals.append(model.get_additional())

            # predict for valid
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

            # save model
            if self.model_dir:
                model.save(filepath=f"{self.model_dir}/fold{fold+1}.model")
                if self.target_enc_cols:
                    target_encoder.save(filepath=f"{self.model_dir}/fold{fold+1}.te")
        if pred_valid_df["oof_pred"].isnull().sum() > 0:
            raise Exception(pred_valid_df["oof_pred"].isnull().sum())
        output = {}
        # add oof to output
        if self.pred_oof:
            output["oof_pred"] = pred_valid_df["oof_pred"].to_list()
            if issubclass(model.__class__, MultiClassifierBase):
                output["oof_raw_pred"] = pred_valid_df[
                    [f"oof_raw_pred{i}" for i in range(valid_raw_pred.shape[1])]
                ].values
            elif issubclass(model.__class__, RegressorBase) or issubclass(model.__class__, BinaryClassifierBase):
                output["oof_raw_pred"] = pred_valid_df["oof_raw_pred"].to_list()
            else:
                raise Exception(f"Invalid model class: {model.__class__}")

        # add evaluation to output
        predictions = Predictions(pred=output["oof_pred"], raw_pred=output["oof_raw_pred"])
        output["evaluator"] = self.evaluator.return_flag()
        if df[self.target_col].isnull().sum() > 0:
            raise Exception(df[self.target_col].isnull().sum())

        score = self.evaluator.run(df[self.target_col], predictions)
        print(f"CV: {score}")
        output["score"] = score
        output["additional"] = additionals
        return output

    def load_models(self):
        """modelとtarget encoderをloadする"""
        if self.models or self.encoders:
            raise Exception("Trained model already exists.")
        models_pathlist = glob.glob(f"{self.model_dir}/fold*.model")
        models = []
        for model_path in sorted(models_pathlist):
            model = load_model(model_path)
            models.append(model)
        self.models = models

        if not self.target_enc_cols:
            return

        encoders_pathlist = glob.glob(f"{self.model_dir}/fold*.te")
        encoders = []
        for encoder_path in sorted(encoders_pathlist):
            encoder = TargetEncoder(
                input_cols=self.target_enc_cols,
                target_col=self.target_col,
                random_state=self.seed,
                log_target=self.log_target_on_target_enc,
            )
            encoder.load(encoder_path)
            encoders.append(encoder)
        self.encoders = encoders

    def raw_output(self, test_df):
        """予測値の元になるscoreを出力する.
        regression : 予測値そのまま (レコード数,)
        binary class : 正例の確率 (レコード数,)
        multi class : 確率ベクトル (レコード数, クラス数)
        """
        test_df = test_df.drop(self.unused_cols, axis=1)

        preds = []
        for i, model in enumerate(self.models):
            if self.target_enc_cols:
                encoder = self.encoders[i]
                test_df = encoder.transform(test_df)
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
