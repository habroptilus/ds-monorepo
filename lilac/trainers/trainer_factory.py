from .basic_trainer import BasicTrainer
from .down_sampling_bagging_trainer import DownSamplingBaggingTrainer
from lilac.core.factory_base import FactoryBase


class TrainerFactory(FactoryBase):
    def __init__(self, register_from=None):
        str2model = {
            "basic": BasicTrainer,
            "down_sample_bagging":  DownSamplingBaggingTrainer
        }
        super().__init__(str2model, register_from)
