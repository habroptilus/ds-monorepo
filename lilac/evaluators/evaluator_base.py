from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_recall_curve,
    roc_auc_score,
)

from lilac.core.data import Predictions


class EvaluatorBase(metaclass=ABCMeta):
    """評価計算クラスの基底."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, y: List, predictions: Predictions):
        raise Exception("Not implemented error.")

    def return_flag(self):
        # TODO そもそもこれいるのか問題を考える
        # 使っているのはoutput["evaluator"]に入れるところだが、他の実験条件と同列に扱うならoutputに入れるのはおかしい気もする
        # Evaluatorに限らず、一意性を表すflagの管理をどうするか問題は考える必要がある
        return self.flag


class RmsleEvaluator(EvaluatorBase):
    """RMSLEで評価する."""

    direction = "minimize"
    flag = "rmsle"

    def run(self, y, predictions):
        return np.sqrt(mean_squared_log_error(y, predictions.pred))


class RmseEvaluator(EvaluatorBase):
    """RMSEで評価する."""

    direction = "minimize"
    flag = "rmse"

    def run(self, y, predictions):
        return np.sqrt(mean_squared_error(y, predictions.pred))


class MaeEvaluator(EvaluatorBase):
    """MAEで評価する."""

    direction = "minimize"
    flag = "mae"

    def run(self, y, predictions):
        return mean_absolute_error(y, predictions.pred)


class AucEvaluator(EvaluatorBase):
    """AUCで評価する.ROC曲線のAUCの方."""

    direction = "maximize"
    flag = "auc"

    def run(self, y, predictions):
        return roc_auc_score(y, predictions.raw_pred)


class AccuracyEvaluator(EvaluatorBase):
    """Accuracyで評価する."""

    direction = "maximize"
    flag = "accuracy"

    def run(self, y, predictions):
        return accuracy_score(y, predictions.pred)


class MacroF1Evaluator(EvaluatorBase):
    """macro f1_scoreで評価する."""

    direction = "maximize"
    flag = "macro_f1"

    def run(self, y, predictions):
        return f1_score(y, predictions.pred, average="macro")


class PrAucEvaluator(EvaluatorBase):
    """PR AUCで評価する."""

    direction = "maximaize"
    flag = "prauc"

    def run(self, y: List, predictions: Predictions):
        precision, recall, _ = precision_recall_curve(y, predictions.raw_pred)
        return auc(recall, precision)


class MapeEvaluator(EvaluatorBase):
    """MAPEで評価する."""

    direction = "minimize"
    flag = "mape"

    def run(self, y: List, predictions: Predictions):
        return np.mean(np.abs((predictions.pred - y) / y)) * 100


class RmspeEvaluator(EvaluatorBase):
    """RMS`PEで評価する."""

    direction = "minimize"
    flag = "rmspe"

    def run(self, y: List, predictions: Predictions):
        return np.sqrt(np.mean(((predictions.pred - y) / y) ** 2)) * 100
