import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, roc_auc_score, accuracy_score, f1_score)
from tools.core.data import Predictions
from typing import List
from abc import ABCMeta, abstractmethod


class EvaluatorBase(metaclass=ABCMeta):
    """評価計算クラスの基底."""

    @abstractmethod
    def run(self, y: List, predictions: Predictions):
        raise Exception("Not implemented error.")

    def return_flag(self):
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
    """AUCで評価する."""
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

    def run(self, y,  predictions):
        return f1_score(y, predictions.pred, average='macro')
