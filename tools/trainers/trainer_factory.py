from .basic_trainer import BasicTrainer
from .down_sampling_bagging_trainer import DownSamplingBaggingTrainer
from tools.core.factory_base import FactoryBase


class TrainerFactory(FactoryBase):
    def __init__(self, custom_members=None):
        # model側にデフォルトを移す
        self.default_params = {
            "bagging_num": 5,
            "base_class": 3,
            "allow_less_than_base": True
        }
        str2model = {
            "basic": BasicTrainer,
            "down_sample_bagging":  DownSamplingBaggingTrainer
        }
        super().__init__(str2model, custom_members)

    def run(self, model_str, target_col, params=None, seed=None):
        return super().run(model_str, target_col, params, seed)

    def get_params(self, target_col, params, seed):
        result = self.default_params.copy()
        if params:
            print(f"Params updated with {params}")
            result.update(params)
        result["target_col"] = target_col
        result["seed"] = seed
        return result
