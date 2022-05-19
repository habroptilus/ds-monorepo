from .basic_trainer import BasicTrainer
from .down_sampling_bagging_trainer import DownSamplingBaggingTrainer
from tools.core.factory_base import FactoryBase


class TrainerFactory(FactoryBase):
    def __init__(self, custom_members=None):
        str2model = {
            "basic": BasicTrainer,
            "down_sample_bagging":  DownSamplingBaggingTrainer
        }
        super().__init__(str2model, custom_members)
